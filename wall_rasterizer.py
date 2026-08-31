"""Independent Schema 3 wall reconstruction for triangleNearestCellCenter-v1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import h5py
import numpy as np
from shondy_hybrid_contract import CONTRACT_V2

RASTERIZATION_ALGORITHM_VERSION = str(CONTRACT_V2["wall"]["rasterizationAlgorithm"])
WALL_GRID_CHANNEL_NAMES = tuple(CONTRACT_V2["channels"]["wallGrid"])
MOTION_MODES = tuple(CONTRACT_V2["wall"]["motionModes"])


class WallContractError(ValueError):
    pass


def _text_attribute(group: h5py.Group, name: str) -> str:
    if name not in group.attrs:
        raise WallContractError(f"Missing HDF5 attribute {group.name}:{name}.")
    value = np.asarray(group.attrs[name])
    if value.size != 1:
        raise WallContractError(f"HDF5 attribute {group.name}:{name} must be scalar.")
    scalar = value.reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    raise WallContractError(f"HDF5 attribute {group.name}:{name} must be text.")


def _json_attribute(group: h5py.Group, name: str) -> Any:
    try:
        return json.loads(_text_attribute(group, name))
    except json.JSONDecodeError as error:
        raise WallContractError(
            f"HDF5 attribute {group.name}:{name} must contain valid JSON."
        ) from error


def _finite_array(group: h5py.Group, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise WallContractError(f"Missing HDF5 dataset {group.name}/{name}.")
    value = np.asarray(group[name])
    if value.shape != shape or not np.isfinite(value).all():
        raise WallContractError(
            f"HDF5 dataset {group.name}/{name} must have finite shape {shape}."
        )
    return value.astype(np.float64, copy=False)


def normalize_quaternion_wxyz(value: Any, *, require_unit: bool = True) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise WallContractError("WXYZ quaternion must be a finite length-4 vector.")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise WallContractError("WXYZ quaternion must have a finite non-zero norm.")
    if require_unit and not math.isclose(norm, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise WallContractError("WXYZ quaternion must be unit length.")
    return quaternion / norm


def quaternion_matrix_wxyz(value: Any) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply_wxyz(left: Any, right: Any) -> np.ndarray:
    lw, lx, ly, lz = normalize_quaternion_wxyz(left)
    rw, rx, ry, rz = normalize_quaternion_wxyz(right)
    return normalize_quaternion_wxyz(
        np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ]
        )
    )


def quaternion_slerp_wxyz(left: Any, right: Any, alpha: float) -> np.ndarray:
    if not math.isfinite(alpha):
        raise WallContractError("Quaternion interpolation alpha must be finite.")
    amount = min(1.0, max(0.0, float(alpha)))
    first = normalize_quaternion_wxyz(left)
    second = normalize_quaternion_wxyz(right)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1.0e-12:
        return normalize_quaternion_wxyz((1.0 - amount) * first + amount * second)
    theta = math.acos(dot)
    sine = math.sin(theta)
    return normalize_quaternion_wxyz(
        math.sin((1.0 - amount) * theta) / sine * first
        + math.sin(amount * theta) / sine * second
    )


@dataclass(frozen=True)
class RigidBodyReference:
    uuid: str
    motion_mode: str
    reference_vertices: np.ndarray
    triangle_connectivity: np.ndarray
    triangle_ids: np.ndarray
    reference_centroid: np.ndarray
    reference_translation: np.ndarray
    reference_rotation_wxyz: np.ndarray
    origin_time: float | None = None
    prescribed_linear_velocity: np.ndarray | None = None
    prescribed_angular_velocity: np.ndarray | None = None


@dataclass(frozen=True)
class WallGeometry:
    body_uuids: tuple[str, ...]
    bodies: tuple[RigidBodyReference, ...]

    @property
    def motion_modes(self) -> tuple[str, ...]:
        return tuple(body.motion_mode for body in self.bodies)

    @property
    def is_static(self) -> bool:
        return all(body.motion_mode == "static" for body in self.bodies)


@dataclass(frozen=True)
class ResolvedWallState:
    body_uuid: str
    body_index: int
    translation: np.ndarray
    rotation_wxyz: np.ndarray
    center_of_mass: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    vertices: np.ndarray
    triangle_connectivity: np.ndarray
    triangle_ids: np.ndarray


def load_wall_geometry(file: h5py.File) -> WallGeometry:
    if "wallGeometry" not in file or not isinstance(file["wallGeometry"], h5py.Group):
        raise WallContractError("Missing /wallGeometry group.")
    group = file["wallGeometry"]
    expected_attributes = {
        "geometryFormatVersion": CONTRACT_V2["wall"]["geometryFormatVersion"],
        "coordinateFrame": "world-right-handed",
        "quaternionOrder": "WXYZ",
        "quaternionConvention": CONTRACT_V2["wall"]["quaternionConvention"],
        "coordinateTransform": CONTRACT_V2["wall"]["coordinateTransform"],
        "timeInterpolation": CONTRACT_V2["wall"]["timeInterpolation"],
        "velocityDefinition": CONTRACT_V2["wall"]["velocityDefinition"],
        "rasterizationAlgorithm": RASTERIZATION_ALGORITHM_VERSION,
    }
    for name, expected in expected_attributes.items():
        if _text_attribute(group, name) != expected:
            raise WallContractError(f"wallGeometry {name} does not match contract v2.")
    uuid_value = _json_attribute(group, "bodyUuidsJson")
    if (
        not isinstance(uuid_value, list)
        or any(not isinstance(value, str) or not value for value in uuid_value)
        or len(set(uuid_value)) != len(uuid_value)
    ):
        raise WallContractError("bodyUuidsJson must be a unique ordered string list.")
    body_uuids = tuple(uuid_value)
    if "bodies" not in group or not isinstance(group["bodies"], h5py.Group):
        raise WallContractError("Missing /wallGeometry/bodies group.")
    body_groups = group["bodies"]
    expected_names = tuple(f"body-{index:06d}" for index in range(len(body_uuids)))
    if tuple(sorted(body_groups.keys())) != expected_names:
        raise WallContractError("Wall body groups do not match stable UUID order.")
    bodies: list[RigidBodyReference] = []
    for body_index, (uuid, body_name) in enumerate(zip(body_uuids, expected_names)):
        body = body_groups[body_name]
        if _text_attribute(body, "bodyType") != "rigid":
            raise WallContractError(f"Wall body {body_index} must have type rigid.")
        motion_mode = _text_attribute(body, "motionMode")
        if motion_mode not in MOTION_MODES:
            raise WallContractError(f"Unknown wall motion mode: {motion_mode}.")
        if (
            "referenceVertices" not in body
            or not isinstance(body["referenceVertices"], h5py.Dataset)
            or body["referenceVertices"].ndim != 2
            or body["referenceVertices"].shape[1] != 3
        ):
            raise WallContractError("referenceVertices must have shape [V,3].")
        vertices = np.asarray(body["referenceVertices"], dtype=np.float64)
        if vertices.shape[0] < 3 or not np.isfinite(vertices).all():
            raise WallContractError("referenceVertices must contain finite triangles.")
        if (
            "triangleConnectivity" not in body
            or not isinstance(body["triangleConnectivity"], h5py.Dataset)
            or body["triangleConnectivity"].ndim != 2
            or body["triangleConnectivity"].shape[1] != 3
            or body["triangleConnectivity"].dtype.kind not in "iu"
        ):
            raise WallContractError("triangleConnectivity must be integer [T,3].")
        connectivity = np.asarray(body["triangleConnectivity"], dtype=np.int64)
        if (
            connectivity.shape[0] == 0
            or np.any(connectivity < 0)
            or np.any(connectivity >= vertices.shape[0])
        ):
            raise WallContractError("triangleConnectivity contains invalid indices.")
        if (
            "triangleId" not in body
            or not isinstance(body["triangleId"], h5py.Dataset)
            or body["triangleId"].dtype.kind not in "iu"
        ):
            raise WallContractError("triangleId must be an integer dataset.")
        triangle_ids = np.asarray(body["triangleId"], dtype=np.int64)
        if (
            triangle_ids.shape != (connectivity.shape[0],)
            or np.any(triangle_ids < 0)
            or np.unique(triangle_ids).size != triangle_ids.size
        ):
            raise WallContractError("triangleId values must be stable and unique.")
        triangle_vertices = vertices[connectivity]
        normals = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        if np.any(np.linalg.norm(normals, axis=1) <= 1.0e-15):
            raise WallContractError("Wall topology contains a degenerate triangle.")
        centroid = _finite_array(body, "referenceCentroid", (3,))
        if not np.allclose(centroid, vertices.mean(axis=0), rtol=0.0, atol=1.0e-6):
            raise WallContractError("referenceCentroid disagrees with vertices.")
        pose = _finite_array(body, "referencePose", (7,))
        rotation = normalize_quaternion_wxyz(pose[3:])
        origin_time: float | None = None
        linear: np.ndarray | None = None
        angular: np.ndarray | None = None
        if motion_mode == "prescribed-law":
            law = _json_attribute(body, "motionLawJson")
            if not isinstance(law, dict):
                raise WallContractError("motionLawJson must contain an object.")
            origin_time = float(law.get("originTime", 0.0))
            linear = np.asarray(law.get("linearVelocity"), dtype=np.float64)
            angular = np.asarray(law.get("angularVelocity"), dtype=np.float64)
            if (
                not math.isfinite(origin_time)
                or linear.shape != (3,)
                or angular.shape != (3,)
                or not np.isfinite(linear).all()
                or not np.isfinite(angular).all()
            ):
                raise WallContractError("Prescribed motion law values must be finite.")
        bodies.append(
            RigidBodyReference(
                uuid=uuid,
                motion_mode=motion_mode,
                reference_vertices=vertices,
                triangle_connectivity=connectivity,
                triangle_ids=triangle_ids,
                reference_centroid=centroid,
                reference_translation=pose[:3],
                reference_rotation_wxyz=rotation,
                origin_time=origin_time,
                prescribed_linear_velocity=linear,
                prescribed_angular_velocity=angular,
            )
        )
    return WallGeometry(body_uuids=body_uuids, bodies=tuple(bodies))


def validate_frame_wall_state(frame: h5py.Group, wall_geometry: WallGeometry) -> None:
    if "bodyUuidsJson" in frame.attrs:
        raise WallContractError(
            "Body UUID topology must only appear under wallGeometry."
        )
    state = frame.get("wallState")
    needs_state = any(
        mode in ("sampled-state", "coupled-state")
        for mode in wall_geometry.motion_modes
    )
    if state is None:
        if needs_state:
            raise WallContractError(
                f"Missing wallState for sampled/coupled frame {frame.name}."
            )
        return
    if not isinstance(state, h5py.Group):
        raise WallContractError(f"{frame.name}/wallState must be a group.")
    if "bodyUuidsJson" in state.attrs:
        raise WallContractError(
            "Body UUID topology must only appear under wallGeometry."
        )
    if int(np.asarray(state.attrs.get("bodyCount", -1)).reshape(-1)[0]) != len(
        wall_geometry.bodies
    ):
        raise WallContractError("wallState bodyCount does not match root UUID order.")
    poses = _finite_array(state, "pose", (len(wall_geometry.bodies), 7))
    _finite_array(state, "linearVelocity", (len(wall_geometry.bodies), 3))
    _finite_array(state, "angularVelocity", (len(wall_geometry.bodies), 3))
    for pose in poses:
        normalize_quaternion_wxyz(pose[3:])


def _sampled_state(
    file: h5py.File, body_index: int, time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for frame_name in sorted(file["frames"].keys()):
        frame = file["frames"][frame_name]
        state = frame.get("wallState")
        if not isinstance(state, h5py.Group):
            continue
        sample_time = float(np.asarray(frame.attrs["timeStart"]).reshape(-1)[0])
        samples.append(
            (
                sample_time,
                np.asarray(state["pose"][body_index], dtype=np.float64),
                np.asarray(state["linearVelocity"][body_index], dtype=np.float64),
                np.asarray(state["angularVelocity"][body_index], dtype=np.float64),
            )
        )
    if not samples:
        raise WallContractError("sampled/coupled wall body has no state samples.")
    if time <= samples[0][0]:
        pose = samples[0][1]
        return (
            pose[:3],
            normalize_quaternion_wxyz(pose[3:]),
            samples[0][2],
            samples[0][3],
        )
    if time >= samples[-1][0]:
        pose = samples[-1][1]
        return (
            pose[:3],
            normalize_quaternion_wxyz(pose[3:]),
            samples[-1][2],
            samples[-1][3],
        )
    for left, right in pairwise(samples):
        if left[0] <= time <= right[0]:
            if right[0] <= left[0]:
                raise WallContractError("wallState sample times must be increasing.")
            alpha = (time - left[0]) / (right[0] - left[0])
            return (
                (1.0 - alpha) * left[1][:3] + alpha * right[1][:3],
                quaternion_slerp_wxyz(left[1][3:], right[1][3:], alpha),
                (1.0 - alpha) * left[2] + alpha * right[2],
                (1.0 - alpha) * left[3] + alpha * right[3],
            )
    raise WallContractError("Unable to bracket sampled wall state.")


def resolve_wall_states(
    file: h5py.File, wall_geometry: WallGeometry, time: float
) -> tuple[ResolvedWallState, ...]:
    if not math.isfinite(time):
        raise WallContractError("Wall reconstruction time must be finite.")
    result: list[ResolvedWallState] = []
    for body_index, body in enumerate(wall_geometry.bodies):
        if body.motion_mode == "static":
            translation = body.reference_translation
            rotation = body.reference_rotation_wxyz
            linear_velocity = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)
        elif body.motion_mode == "prescribed-law":
            assert body.origin_time is not None
            assert body.prescribed_linear_velocity is not None
            assert body.prescribed_angular_velocity is not None
            elapsed = time - body.origin_time
            linear_velocity = body.prescribed_linear_velocity
            angular_velocity = body.prescribed_angular_velocity
            translation = body.reference_translation + linear_velocity * elapsed
            angular_speed = float(np.linalg.norm(angular_velocity))
            if angular_speed == 0.0:
                rotation = body.reference_rotation_wxyz
            else:
                axis = angular_velocity / angular_speed
                half_angle = 0.5 * angular_speed * elapsed
                delta = np.asarray(
                    [math.cos(half_angle), *(math.sin(half_angle) * axis)]
                )
                rotation = quaternion_multiply_wxyz(delta, body.reference_rotation_wxyz)
        else:
            translation, rotation, linear_velocity, angular_velocity = _sampled_state(
                file, body_index, time
            )
        rotation_matrix = quaternion_matrix_wxyz(rotation)
        vertices = translation + body.reference_vertices @ rotation_matrix.T
        center = translation + rotation_matrix @ body.reference_centroid
        result.append(
            ResolvedWallState(
                body_uuid=body.uuid,
                body_index=body_index,
                translation=translation,
                rotation_wxyz=rotation,
                center_of_mass=center,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                vertices=vertices,
                triangle_connectivity=body.triangle_connectivity,
                triangle_ids=body.triangle_ids,
            )
        )
    return tuple(result)


def _closest_points_triangle(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = ap @ ab
    d2 = ap @ ac
    closest = np.empty_like(points)
    assigned = np.zeros(points.shape[0], dtype=np.bool_)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask] = a
    assigned |= mask

    bp = points - b
    d3 = bp @ ab
    d4 = bp @ ac
    mask = ~assigned & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = ~assigned & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denominator = d1[mask] - d3[mask]
    closest[mask] = a + (d1[mask] / denominator)[:, None] * ab
    assigned |= mask

    cp = points - c
    d5 = cp @ ab
    d6 = cp @ ac
    mask = ~assigned & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = ~assigned & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denominator = d2[mask] - d6[mask]
    closest[mask] = a + (d2[mask] / denominator)[:, None] * ac
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = ~assigned & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    numerator = d4[mask] - d3[mask]
    denominator = numerator + d5[mask] - d6[mask]
    closest[mask] = b + (numerator / denominator)[:, None] * (c - b)
    assigned |= mask

    mask = ~assigned
    denominator = va[mask] + vb[mask] + vc[mask]
    v = vb[mask] / denominator
    w = vc[mask] / denominator
    closest[mask] = a + v[:, None] * ab + w[:, None] * ac
    difference = points - closest
    return closest, np.einsum("ij,ij->i", difference, difference)


def rasterize_wall_grid(
    file: h5py.File, wall_geometry: WallGeometry, time: float
) -> np.ndarray:
    """Return float32 channels in strict ``(channel,z,y,x)`` layout."""

    grid = file["gridGeometry"]
    padded_min = _finite_array(grid, "paddedBoundsMin", (3,))
    counts_raw = np.asarray(grid["cellCounts"])
    if counts_raw.shape != (3,) or counts_raw.dtype.kind not in "iu":
        raise WallContractError("gridGeometry/cellCounts must be integer [3].")
    nx, ny, nz = (int(value) for value in counts_raw)
    cell_size = float(np.asarray(grid.attrs.get("cellSize", math.nan)).reshape(-1)[0])
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise WallContractError("Grid cellSize must be finite and positive.")
    x = padded_min[0] + (np.arange(nx, dtype=np.float64) + 0.5) * cell_size
    y = padded_min[1] + (np.arange(ny, dtype=np.float64) + 0.5) * cell_size
    z = padded_min[2] + (np.arange(nz, dtype=np.float64) + 0.5) * cell_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    count = points.shape[0]
    best_distance = np.full(count, np.inf, dtype=np.float64)
    best_triangle = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    best_body = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    best_point = np.zeros((count, 3), dtype=np.float64)
    best_normal = np.zeros((count, 3), dtype=np.float64)
    best_linear = np.zeros((count, 3), dtype=np.float64)
    best_angular = np.zeros((count, 3), dtype=np.float64)
    best_center = np.zeros((count, 3), dtype=np.float64)
    states = resolve_wall_states(file, wall_geometry, time)
    for state in states:
        for triangle_index, connectivity in enumerate(state.triangle_connectivity):
            a, b, c = state.vertices[connectivity]
            raw_normal = np.cross(b - a, c - a)
            normal = raw_normal / np.linalg.norm(raw_normal)
            closest, distance = _closest_points_triangle(points, a, b, c)
            triangle_id = int(state.triangle_ids[triangle_index])
            better = distance < best_distance
            tie = distance == best_distance
            better |= tie & (triangle_id < best_triangle)
            better |= (
                tie & (triangle_id == best_triangle) & (state.body_index < best_body)
            )
            if not np.any(better):
                continue
            best_distance[better] = distance[better]
            best_triangle[better] = triangle_id
            best_body[better] = state.body_index
            best_point[better] = closest[better]
            best_normal[better] = normal
            best_linear[better] = state.linear_velocity
            best_angular[better] = state.angular_velocity
            best_center[better] = state.center_of_mass
    if not np.isfinite(best_distance).all():
        raise WallContractError("Wall geometry does not contain a valid triangle.")
    band_distance = (
        float(CONTRACT_V2["wall"]["rasterizationBandWidthCells"]) * cell_size
    )
    in_band = best_distance <= band_distance * band_distance
    output = np.zeros((8, count), dtype=np.float32)
    output[0, in_band] = 1.0
    output[1:4, in_band] = best_normal[in_band].T.astype(np.float32)
    point_velocity = best_linear + np.cross(best_angular, best_point - best_center)
    output[4:7, in_band] = point_velocity[in_band].T.astype(np.float32)
    output[7, ~in_band] = 1.0
    return output.reshape(8, nz, ny, nx)
