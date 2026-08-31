"""PyTorch reference implementation for the dense-grid hybrid surrogate."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import random
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from shondy_hybrid_contract import CONTRACT_V2 as CONTRACT
from shondy_hybrid_contract import REGISTRY_V2_SHA256 as REGISTRY_SHA256
from shondy_hybrid_contract import validate_model_bundle_v3

from wall_rasterizer import (
    RASTERIZATION_ALGORITHM_VERSION,
    ResolvedWallState,
    WallContractError,
    WallGeometry,
    load_wall_geometry,
    rasterize_wall_grid,
    resolve_wall_states,
    validate_frame_wall_state,
)

TEACHER_SCHEMA_VERSION = int(CONTRACT["teacher"]["schemaVersion"])
MODEL_BUNDLE_SCHEMA_VERSION = int(CONTRACT["modelBundle"]["schemaVersion"])
NATIVE_PARTICLE_MLP_SCHEMA_VERSION = int(
    CONTRACT["modelBundle"]["nativeParticleMlp"]["schemaVersion"]
)
# Retained for callers that used the original Teacher-only constant.
SCHEMA_VERSION = TEACHER_SCHEMA_VERSION
TENSOR_LAYOUT = str(CONTRACT["teacher"]["tensorLayout"])
PADDING_CELLS = int(CONTRACT["grid"]["paddingCells"])
LOCAL_FEATURE_COUNT = len(CONTRACT["channels"]["local"])
DYNAMIC_QUANTITY_COUNT = len(CONTRACT["channels"]["dynamicQuantities"])
DYNAMIC_GRID_CHANNEL_COUNT = len(CONTRACT["channels"]["dynamicGrid"])
WALL_GRID_CHANNEL_COUNT = len(CONTRACT["channels"]["wallGrid"])
BASE_GRID_CHANNEL_COUNT = len(CONTRACT["channels"]["gridBase"])
LATENT_CHANNEL_COUNT = len(CONTRACT["channels"]["latent"])
TARGET_CHANNEL_COUNT = len(CONTRACT["channels"]["target"])
PARTICLE_MLP_WIDTHS = tuple(
    int(value) for value in CONTRACT["modelBundle"]["nativeParticleMlp"]["architecture"]
)
ONNX_CONSISTENCY_RTOL = 2.0e-2
ONNX_CONSISTENCY_ATOL = 5.0e-2
WALL_CACHE_RTOL = 1.0e-5
WALL_CACHE_ATOL = 1.0e-6

DEFAULT_MODEL_PROFILE = "compact-v1"
MODEL_PROFILES: dict[str, dict[str, tuple[int, ...]]] = {
    "compact-v1": {
        "encoderWidths": (32, 48, 64, 96),
        "decoderWidths": (64, 48, 32),
    },
    "large-v1": {
        "encoderWidths": (64, 96, 128, 192),
        "decoderWidths": (128, 96, 64),
    },
}

LOCAL_FEATURE_NAMES = tuple(CONTRACT["channels"]["local"])
DYNAMIC_QUANTITY_NAMES = tuple(CONTRACT["channels"]["dynamicQuantities"])
DYNAMIC_GRID_CHANNEL_NAMES = tuple(CONTRACT["channels"]["dynamicGrid"])
WALL_GRID_CHANNEL_NAMES = tuple(CONTRACT["channels"]["wallGrid"])
COORDINATE_CHANNEL_NAMES = tuple(CONTRACT["channels"]["coordinates"])
BASE_GRID_CHANNEL_NAMES = tuple(CONTRACT["channels"]["gridBase"])
LATENT_CHANNEL_NAMES = tuple(CONTRACT["channels"]["latent"])
TARGET_CHANNEL_NAMES = tuple(CONTRACT["channels"]["target"])


class ContractError(ValueError):
    """Raised when data does not satisfy the frozen hybrid contract."""


class IncompleteGridSupportError(ContractError):
    """Raised instead of clamping a particle with incomplete support."""


def particle_volume_from_radius(radius: torch.Tensor) -> torch.Tensor:
    """Match the native MPS particle volume: diameter cubed, or 8*r^3."""

    return 8.0 * radius * radius * radius


def _finite_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ContractError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class GridGeometry:
    physical_bounds_min: tuple[float, float, float]
    physical_bounds_max: tuple[float, float, float]
    padded_bounds_min: tuple[float, float, float]
    padded_bounds_max: tuple[float, float, float]
    cell_counts: tuple[int, int, int]
    cell_size: float
    padding_cells: int = PADDING_CELLS

    def __post_init__(self) -> None:
        physical_min = _finite_tuple(self.physical_bounds_min, "physical_bounds_min")
        physical_max = _finite_tuple(self.physical_bounds_max, "physical_bounds_max")
        padded_min = _finite_tuple(self.padded_bounds_min, "padded_bounds_min")
        padded_max = _finite_tuple(self.padded_bounds_max, "padded_bounds_max")
        counts = tuple(int(value) for value in self.cell_counts)
        if len(counts) != 3 or any(value < 2 for value in counts):
            raise ContractError("cell_counts must contain three values >= 2.")
        if not math.isfinite(self.cell_size) or self.cell_size <= 0.0:
            raise ContractError("cell_size must be finite and positive.")
        if self.padding_cells != PADDING_CELLS:
            raise ContractError("The frozen grid contract requires 3 padding cells.")
        tolerance = 64.0 * np.finfo(np.float64).eps
        for axis in range(3):
            if physical_max[axis] <= physical_min[axis]:
                raise ContractError("Physical bounds must have positive extent.")
            if (
                padded_min[axis] > physical_min[axis]
                or padded_max[axis] < physical_max[axis]
            ):
                raise ContractError("Padded bounds must contain physical bounds.")
            expected_min = physical_min[axis] - PADDING_CELLS * self.cell_size
            scale = max(1.0, abs(expected_min), abs(padded_min[axis]))
            if abs(expected_min - padded_min[axis]) > tolerance * scale:
                raise ContractError("Padded bounds must start with 3 full cells.")
            if padded_max[axis] < physical_max[axis] + PADDING_CELLS * self.cell_size:
                raise ContractError(
                    "Padded bounds must end with at least 3 full cells."
                )
            expected_max = padded_min[axis] + counts[axis] * self.cell_size
            scale = max(1.0, abs(expected_max), abs(padded_max[axis]))
            if abs(expected_max - padded_max[axis]) > tolerance * scale:
                raise ContractError("Padded extent must equal cell_counts * cell_size.")
        object.__setattr__(self, "physical_bounds_min", physical_min)
        object.__setattr__(self, "physical_bounds_max", physical_max)
        object.__setattr__(self, "padded_bounds_min", padded_min)
        object.__setattr__(self, "padded_bounds_max", padded_max)
        object.__setattr__(self, "cell_counts", counts)

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.cell_counts
        return nz, ny, nx

    @property
    def cell_volume(self) -> float:
        return self.cell_size**3

    @property
    def cell_count(self) -> int:
        nx, ny, nz = self.cell_counts
        return nx * ny * nz

    def cell_centers(
        self, *, dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        nx, ny, nz = self.cell_counts
        # Match the C++ contract implementation: geometry arithmetic is
        # performed in double and each completed center is then converted to
        # the requested scalar type. Doing the arithmetic directly in float32
        # can move a center by one ULP across a physical-boundary comparison.
        minimum = torch.tensor(
            self.padded_bounds_min, dtype=torch.float64, device=device
        )
        spacing = torch.as_tensor(self.cell_size, dtype=torch.float64, device=device)

        def centers(axis: int, count: int) -> torch.Tensor:
            index = torch.arange(count, dtype=torch.float64, device=device)
            return (minimum[axis] + (index + 0.5) * spacing).to(dtype=dtype)

        x = centers(0, nx)
        y = centers(1, ny)
        z = centers(2, nz)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0).contiguous()

    def coordinate_channels(
        self, *, dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        centers = self.cell_centers(dtype=dtype, device=device)
        minimum = torch.tensor(self.padded_bounds_min, dtype=dtype, device=device).view(
            3, 1, 1, 1
        )
        maximum = torch.tensor(self.padded_bounds_max, dtype=dtype, device=device).view(
            3, 1, 1, 1
        )
        return (2.0 * (centers - minimum) / (maximum - minimum) - 1.0).contiguous()

    def valid_domain_mask(
        self, *, dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        centers = self.cell_centers(dtype=dtype, device=device)
        minimum = torch.tensor(
            self.physical_bounds_min, dtype=dtype, device=device
        ).view(3, 1, 1, 1)
        maximum = torch.tensor(
            self.physical_bounds_max, dtype=dtype, device=device
        ).view(3, 1, 1, 1)
        inside = torch.logical_and(centers >= minimum, centers <= maximum).all(dim=0)
        return inside.to(dtype=dtype).unsqueeze(0).contiguous()

    def to_metadata(self) -> dict[str, object]:
        return {
            "physicalBoundsMin": list(self.physical_bounds_min),
            "physicalBoundsMax": list(self.physical_bounds_max),
            "paddedBoundsMin": list(self.padded_bounds_min),
            "paddedBoundsMax": list(self.padded_bounds_max),
            "cellCounts": list(self.cell_counts),
            "cellSize": self.cell_size,
            "paddingCells": self.padding_cells,
            "cellCentered": True,
            "tensorLayout": TENSOR_LAYOUT,
        }


@dataclass(frozen=True)
class WallState:
    body_uuids: tuple[str, ...]
    center_of_mass: np.ndarray
    rotation_wxyz: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class TeacherFrame:
    source_path: Path
    case_id: str
    trajectory_id: str
    macro_step_index: int
    time_start: float
    time_end: float
    ai_delta_time: float
    particle_diameter: float
    certification_profile: str
    substep_count: int
    valid_grid_support: bool
    geometry: GridGeometry
    condition_names: tuple[str, ...]
    conditions: tuple[float, ...]
    static_id: np.ndarray
    position_start: np.ndarray
    velocity_start: np.ndarray
    radius_start: np.ndarray
    local_feature_start: np.ndarray
    position_end: np.ndarray
    velocity_end: np.ndarray
    gravity_reference_velocity_end: np.ndarray
    target_residual_acceleration: np.ndarray
    valid: np.ndarray
    wall_start: WallState
    wall_end: WallState
    wall_channels_start: np.ndarray

    @property
    def trajectory_key(self) -> tuple[str, str]:
        return self.case_id, self.trajectory_id

    @property
    def frame_key(self) -> tuple[str, str, int]:
        return self.case_id, self.trajectory_id, self.macro_step_index

    @property
    def particle_count(self) -> int:
        return int(self.static_id.shape[0])


@dataclass(frozen=True)
class TeacherFrameIndex:
    source_path: Path
    frame_name: str
    case_id: str
    trajectory_id: str
    macro_step_index: int
    time_start: float
    time_end: float
    ai_delta_time: float
    particle_diameter: float
    certification_profile: str
    substep_count: int
    valid_grid_support: bool
    geometry: GridGeometry
    condition_names: tuple[str, ...]
    conditions: tuple[float, ...]
    particle_count: int
    wall_is_static: bool

    @property
    def trajectory_key(self) -> tuple[str, str]:
        return self.case_id, self.trajectory_id

    @property
    def frame_key(self) -> tuple[str, str, int]:
        return self.case_id, self.trajectory_id, self.macro_step_index


@dataclass(frozen=True)
class TrainingFrameData:
    """Minimal frame payload used after the Teacher file has been validated."""

    index: TeacherFrameIndex
    position_start: np.ndarray
    local_feature_start: np.ndarray
    target_residual_acceleration: np.ndarray
    valid: np.ndarray
    radius_start: np.ndarray | None = None
    wall_channels_start: np.ndarray | None = None

    @property
    def frame_key(self) -> tuple[str, str, int]:
        return self.index.frame_key

    @property
    def geometry(self) -> GridGeometry:
        return self.index.geometry

    @property
    def nbytes(self) -> int:
        arrays = (
            self.position_start,
            self.local_feature_start,
            self.target_residual_acceleration,
            self.valid,
            self.radius_start,
            self.wall_channels_start,
        )
        return sum(value.nbytes for value in arrays if value is not None)


def _attribute_text(value: object, name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ContractError(f"HDF5 attribute {name} must be text.")


def _required_attribute(group: h5py.Group | h5py.File, name: str) -> object:
    if name not in group.attrs:
        raise ContractError(f"Missing HDF5 attribute {group.name}:{name}.")
    value = np.asarray(group.attrs[name])
    if value.size != 1:
        raise ContractError(f"HDF5 attribute {group.name}:{name} must be scalar.")
    scalar = value.reshape(-1)[0]
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _required_array(group: h5py.Group, name: str) -> np.ndarray:
    if name not in group:
        raise ContractError(f"Missing HDF5 dataset {group.name}/{name}.")
    return np.asarray(group[name])


def _load_wall_state(group: h5py.Group) -> WallState:
    body_uuids_raw = _attribute_text(
        _required_attribute(group, "bodyUuidsJson"), "bodyUuidsJson"
    )
    body_uuids_value = json.loads(body_uuids_raw)
    if not isinstance(body_uuids_value, list) or not all(
        isinstance(value, str) for value in body_uuids_value
    ):
        raise ContractError("bodyUuidsJson must contain a string list.")
    body_uuids = tuple(body_uuids_value)
    if len(set(body_uuids)) != len(body_uuids):
        raise ContractError("Wall body UUID values must be unique.")
    arrays = {
        "center_of_mass": _required_array(group, "centerOfMass"),
        "rotation_wxyz": _required_array(group, "rotationWxyz"),
        "linear_velocity": _required_array(group, "linearVelocity"),
        "angular_velocity": _required_array(group, "angularVelocity"),
    }
    expected = {
        "center_of_mass": (len(body_uuids), 3),
        "rotation_wxyz": (len(body_uuids), 4),
        "linear_velocity": (len(body_uuids), 3),
        "angular_velocity": (len(body_uuids), 3),
    }
    for name, value in arrays.items():
        if value.shape != expected[name] or not np.isfinite(value).all():
            raise ContractError(f"Invalid wall state array {group.name}:{name}.")
    return WallState(body_uuids=body_uuids, **arrays)


def _load_geometry(file: h5py.File) -> GridGeometry:
    if "gridGeometry" not in file:
        raise ContractError("Missing /gridGeometry group.")
    group = file["gridGeometry"]
    if int(_required_attribute(group, "cellCentered")) != 1:
        raise ContractError("The ML grid must be cell-centered.")
    return GridGeometry(
        physical_bounds_min=tuple(
            float(value) for value in _required_array(group, "physicalBoundsMin")
        ),
        physical_bounds_max=tuple(
            float(value) for value in _required_array(group, "physicalBoundsMax")
        ),
        padded_bounds_min=tuple(
            float(value) for value in _required_array(group, "paddedBoundsMin")
        ),
        padded_bounds_max=tuple(
            float(value) for value in _required_array(group, "paddedBoundsMax")
        ),
        cell_counts=tuple(int(value) for value in _required_array(group, "cellCounts")),
        cell_size=float(_required_attribute(group, "cellSize")),
        padding_cells=int(_required_attribute(group, "paddingCells")),
    )


def _validate_particle_arrays(frame: TeacherFrame) -> None:
    count = frame.particle_count
    expected = {
        "position_start": (count, 3),
        "velocity_start": (count, 3),
        "radius_start": (count,),
        "local_feature_start": (count, LOCAL_FEATURE_COUNT),
        "position_end": (count, 3),
        "velocity_end": (count, 3),
        "gravity_reference_velocity_end": (count, 3),
        "target_residual_acceleration": (count, TARGET_CHANNEL_COUNT),
        "valid": (count,),
    }
    for name, shape in expected.items():
        if getattr(frame, name).shape != shape:
            raise ContractError(f"Invalid particle array shape: {name}.")
    if count == 0:
        raise ContractError("A Teacher frame must contain particles.")
    ids = frame.static_id.astype(np.int64, copy=False)
    if ids.shape != (count,) or np.any(ids < 0) or np.unique(ids).size != count:
        raise ContractError("staticId values must be non-negative and unique.")
    finite_start = (
        np.isfinite(frame.position_start).all()
        and np.isfinite(frame.velocity_start).all()
        and np.isfinite(frame.radius_start).all()
        and np.isfinite(frame.local_feature_start).all()
    )
    if not finite_start or np.any(frame.radius_start <= 0.0):
        raise ContractError("Start particle state must be finite with positive radii.")
    if not np.allclose(
        frame.local_feature_start[:, 0:3],
        frame.velocity_start,
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ContractError("local18 velocity does not match velocityStart.")
    valid = frame.valid
    if valid.dtype != np.bool_:
        raise ContractError("Internal valid mask must have boolean dtype.")
    if valid.any():
        finite_valid = (
            np.isfinite(frame.position_end[valid]).all()
            and np.isfinite(frame.velocity_end[valid]).all()
            and np.isfinite(frame.gravity_reference_velocity_end[valid]).all()
            and np.isfinite(frame.target_residual_acceleration[valid]).all()
        )
        if not finite_valid:
            raise ContractError("Valid particles require finite endpoint and target.")
    if (~valid).any() and np.isfinite(frame.target_residual_acceleration[~valid]).any():
        raise ContractError("Invalid particle targets must remain non-finite.")
    if frame.wall_start.body_uuids != frame.wall_end.body_uuids:
        raise ContractError("wallStart and wallEnd body UUID order must match.")
    wall_channels = torch.as_tensor(frame.wall_channels_start)
    validate_wall_channels(wall_channels, frame.geometry)


type TeacherFileContract = tuple[
    str,
    str,
    float,
    float,
    str,
    GridGeometry,
    tuple[str, ...],
    tuple[float, ...],
    WallGeometry,
]


def _teacher_file_contract(file: h5py.File) -> TeacherFileContract:
    teacher_contract = CONTRACT["teacher"]
    grid_contract = CONTRACT["grid"]
    if (
        _attribute_text(_required_attribute(file, "contractName"), "contractName")
        != teacher_contract["contractName"]
        or _attribute_text(
            _required_attribute(file, "contractVersion"), "contractVersion"
        )
        != CONTRACT["contractPackage"]["version"]
        or _attribute_text(
            _required_attribute(file, "contractRegistrySha256"),
            "contractRegistrySha256",
        )
        != REGISTRY_SHA256
    ):
        raise ContractError("Unsupported Teacher contract identity.")
    if int(_required_attribute(file, "schemaVersion")) != TEACHER_SCHEMA_VERSION:
        raise ContractError("Unsupported Teacher frame schemaVersion.")
    tensor_layout = _attribute_text(
        _required_attribute(file, "tensorLayout"), "tensorLayout"
    )
    if tensor_layout != TENSOR_LAYOUT:
        raise ContractError(f"Expected tensor layout {TENSOR_LAYOUT}.")
    case_id = _attribute_text(_required_attribute(file, "caseId"), "caseId")
    trajectory_id = _attribute_text(
        _required_attribute(file, "trajectoryId"), "trajectoryId"
    )
    ai_delta_time = float(_required_attribute(file, "aiDeltaTime"))
    if not math.isfinite(ai_delta_time) or ai_delta_time <= 0.0:
        raise ContractError("aiDeltaTime must be finite and strictly positive.")
    particle_diameter = float(_required_attribute(file, "particleDiameter"))
    if not math.isfinite(particle_diameter) or particle_diameter <= 0.0:
        raise ContractError("particleDiameter must be finite and positive.")
    exact_text_attributes = {
        "particleVolumeDefinition": teacher_contract["particleVolumeDefinition"],
        "targetDefinition": teacher_contract["targetDefinition"],
        "previousMacroResidualHistoryDefinition": teacher_contract[
            "previousMacroResidualHistoryDefinition"
        ],
        "previousMacroResidualInitialValue": teacher_contract[
            "previousMacroResidualInitialValue"
        ],
        "macroTickDefinition": teacher_contract["macroTickDefinition"],
        "teacherFrameStrideSemantics": teacher_contract["teacherFrameStrideSemantics"],
        "gridInterpolation": grid_contract["interpolation"],
        "coordinateTransform": CONTRACT["wall"]["coordinateTransform"],
        "quaternionConvention": CONTRACT["wall"]["quaternionConvention"],
        "timeInterpolation": CONTRACT["wall"]["timeInterpolation"],
        "velocityDefinition": CONTRACT["wall"]["velocityDefinition"],
        "wallRasterizationAlgorithm": CONTRACT["wall"]["rasterizationAlgorithm"],
    }
    for name, expected in exact_text_attributes.items():
        actual = _attribute_text(_required_attribute(file, name), name)
        if actual != expected:
            raise ContractError(f"Teacher contract attribute {name} mismatch.")
    exact_integer_attributes = {
        "targetIncludesTeacherCollision": 1,
        "targetMacroStepSpan": teacher_contract["targetMacroStepSpan"],
        "gridStencilCellCount": grid_contract["stencilCellCount"],
        "gridClampOutOfBounds": int(grid_contract["clampOutOfBounds"]),
    }
    for name, expected in exact_integer_attributes.items():
        if int(_required_attribute(file, name)) != expected:
            raise ContractError(f"Teacher contract attribute {name} mismatch.")
    if int(_required_attribute(file, "teacherFrameStride")) <= 0:
        raise ContractError("Teacher frame stride must be positive.")
    units = json.loads(
        _attribute_text(_required_attribute(file, "unitsJson"), "unitsJson")
    )
    if units != dict(CONTRACT["units"]):
        raise ContractError("Teacher units do not match the published contract.")
    channel_attributes = {
        "localFeatureNamesJson": "local",
        "dynamicQuantityNamesJson": "dynamicQuantities",
        "gridBaseChannelNamesJson": "gridBase",
        "wallGridChannelNamesJson": "wallGrid",
        "targetChannelNamesJson": "target",
    }
    for attribute_name, registry_name in channel_attributes.items():
        actual = json.loads(
            _attribute_text(_required_attribute(file, attribute_name), attribute_name)
        )
        if actual != list(CONTRACT["channels"][registry_name]):
            raise ContractError(
                f"Teacher {attribute_name} names/order do not match the contract."
            )
    conditions_raw = _attribute_text(
        _required_attribute(file, "conditionsJson"), "conditionsJson"
    )
    conditions_value = json.loads(conditions_raw)
    if not isinstance(conditions_value, dict) or not all(
        isinstance(name, str)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for name, value in conditions_value.items()
    ):
        raise ContractError("conditionsJson must contain finite named scalars.")
    condition_names = tuple(conditions_value.keys())
    encoded_condition_names = json.loads(
        _attribute_text(
            _required_attribute(file, "conditionNamesJson"), "conditionNamesJson"
        )
    )
    if encoded_condition_names != list(condition_names):
        raise ContractError("Teacher condition names/order metadata disagree.")
    conditions = tuple(float(conditions_value[name]) for name in condition_names)
    profile = _attribute_text(
        _required_attribute(file, "certificationProfile"), "certificationProfile"
    )
    profiles = CONTRACT["certificationProfiles"]
    if profile not in tuple(profiles.values()):
        raise ContractError("Unknown Teacher certification profile.")
    geometry = _load_geometry(file)
    try:
        wall_geometry = load_wall_geometry(file)
    except WallContractError as error:
        raise ContractError(str(error)) from error
    if "frames" not in file or not isinstance(file["frames"], h5py.Group):
        raise ContractError("Missing /frames group.")
    return (
        case_id,
        trajectory_id,
        ai_delta_time,
        particle_diameter,
        profile,
        geometry,
        condition_names,
        conditions,
        wall_geometry,
    )


def _index_teacher_frame(
    source_path: Path,
    frame_name: str,
    group: h5py.Group,
    file_contract: TeacherFileContract,
) -> TeacherFrameIndex:
    (
        case_id,
        trajectory_id,
        ai_delta_time,
        particle_diameter,
        profile,
        geometry,
        condition_names,
        conditions,
        wall_geometry,
    ) = file_contract
    if "particle" not in group or "grid" not in group:
        raise ContractError(f"Invalid Teacher frame group: {frame_name}.")
    macro_step_index = int(_required_attribute(group, "macroStepIndex"))
    if frame_name != f"{macro_step_index:012d}":
        raise ContractError("Frame name and macroStepIndex disagree.")
    frame_delta_time = float(_required_attribute(group, "aiDeltaTime"))
    if frame_delta_time != ai_delta_time:
        raise ContractError("Frame and trajectory aiDeltaTime disagree.")
    support_raw = int(_required_attribute(group, "validGridSupport"))
    if support_raw not in (0, 1):
        raise ContractError("validGridSupport must be 0 or 1.")
    time_start = float(_required_attribute(group, "timeStart"))
    time_end = float(_required_attribute(group, "timeEnd"))
    substep_count = int(_required_attribute(group, "substepCount"))
    if substep_count <= 0:
        raise ContractError("substepCount must be positive.")
    if not all(math.isfinite(value) for value in (time_start, time_end)):
        raise ContractError("Teacher frame times must be finite.")
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(time_end), abs(macro_step_index * ai_delta_time))
    )
    if abs((time_end - time_start) - ai_delta_time) > tolerance:
        raise ContractError("Teacher frame duration must equal aiDeltaTime.")
    if (
        abs(time_start - macro_step_index * ai_delta_time) > tolerance
        or abs(time_end - (macro_step_index + 1) * ai_delta_time) > tolerance
    ):
        raise ContractError("Teacher frame times must use integer macro ticks.")
    particle = group["particle"]
    if not isinstance(particle, h5py.Group) or "staticId" not in particle:
        raise ContractError(f"Missing particle datasets in frame {frame_name}.")
    static_id = particle["staticId"]
    if not isinstance(static_id, h5py.Dataset) or len(static_id.shape) != 1:
        raise ContractError("particle/staticId must be a rank-1 dataset.")
    required_paths = (
        "positionStart",
        "velocityStart",
        "radiusStart",
        "localFeatureStart",
        "positionEnd",
        "velocityEnd",
        "gravityReferenceVelocityEnd",
        "targetResidualAcceleration",
        "valid",
    )
    if any(name not in particle for name in required_paths):
        raise ContractError(f"Missing particle datasets in frame {frame_name}.")
    for duplicate_name in ("wallStart", "wallEnd", "wallGeometry"):
        if duplicate_name in group:
            raise ContractError(
                f"Frame {frame_name} repeats root wall topology in {duplicate_name}."
            )
    try:
        validate_frame_wall_state(group, wall_geometry)
    except WallContractError as error:
        raise ContractError(str(error)) from error
    return TeacherFrameIndex(
        source_path=source_path,
        frame_name=frame_name,
        case_id=case_id,
        trajectory_id=trajectory_id,
        macro_step_index=macro_step_index,
        time_start=time_start,
        time_end=time_end,
        ai_delta_time=frame_delta_time,
        particle_diameter=particle_diameter,
        certification_profile=profile,
        substep_count=substep_count,
        valid_grid_support=bool(support_raw),
        geometry=geometry,
        condition_names=condition_names,
        conditions=conditions,
        particle_count=int(static_id.shape[0]),
        wall_is_static=wall_geometry.is_static,
    )


def index_teacher_trajectory(path: str | Path) -> tuple[TeacherFrameIndex, ...]:
    """Index one trajectory without loading particle or grid arrays."""

    source_path = Path(path).resolve()
    with h5py.File(source_path, "r") as file:
        file_contract = _teacher_file_contract(file)
        indexes = tuple(
            _index_teacher_frame(
                source_path, frame_name, file["frames"][frame_name], file_contract
            )
            for frame_name in sorted(file["frames"].keys())
        )
    if not indexes:
        raise ContractError("Teacher trajectory contains no frames.")
    return indexes


def _resolved_wall_state(states: Sequence[ResolvedWallState]) -> WallState:
    return WallState(
        body_uuids=tuple(state.body_uuid for state in states),
        center_of_mass=np.stack([state.center_of_mass for state in states], axis=0)
        if states
        else np.empty((0, 3), dtype=np.float64),
        rotation_wxyz=np.stack([state.rotation_wxyz for state in states], axis=0)
        if states
        else np.empty((0, 4), dtype=np.float64),
        linear_velocity=np.stack([state.linear_velocity for state in states], axis=0)
        if states
        else np.empty((0, 3), dtype=np.float64),
        angular_velocity=np.stack([state.angular_velocity for state in states], axis=0)
        if states
        else np.empty((0, 3), dtype=np.float64),
    )


def _load_teacher_frame_from_file(
    index: TeacherFrameIndex,
    file: h5py.File,
    file_contract: TeacherFileContract,
    wall_channels: np.ndarray | None = None,
) -> TeacherFrame:
    current_index = _index_teacher_frame(
        index.source_path,
        index.frame_name,
        file["frames"][index.frame_name],
        file_contract,
    )
    if current_index != index:
        raise ContractError(f"Teacher frame metadata changed: {index.frame_key}.")
    group = file["frames"][index.frame_name]
    particle = group["particle"]
    valid_raw = _required_array(particle, "valid")
    if not np.isin(valid_raw, (0, 1)).all():
        raise ContractError("Particle valid mask must contain only 0 or 1.")
    wall_geometry = file_contract[-1]
    try:
        if wall_channels is None:
            wall_channels = rasterize_wall_grid(file, wall_geometry, index.time_start)
        start_states = resolve_wall_states(file, wall_geometry, index.time_start)
        end_states = resolve_wall_states(file, wall_geometry, index.time_end)
    except WallContractError as error:
        raise ContractError(str(error)) from error
    if "wallStart" in group["grid"]:
        cached_wall = _required_array(group["grid"], "wallStart")
        if (
            cached_wall.shape != wall_channels.shape
            or cached_wall.dtype != np.float32
            or not np.allclose(
                cached_wall,
                wall_channels,
                rtol=WALL_CACHE_RTOL,
                atol=WALL_CACHE_ATOL,
            )
        ):
            raise ContractError(
                "Optional dense wallStart cache disagrees with Schema 3 reconstruction."
            )
    frame = TeacherFrame(
        source_path=index.source_path,
        case_id=index.case_id,
        trajectory_id=index.trajectory_id,
        macro_step_index=index.macro_step_index,
        time_start=index.time_start,
        time_end=index.time_end,
        ai_delta_time=index.ai_delta_time,
        particle_diameter=index.particle_diameter,
        certification_profile=index.certification_profile,
        substep_count=index.substep_count,
        valid_grid_support=index.valid_grid_support,
        geometry=index.geometry,
        condition_names=index.condition_names,
        conditions=index.conditions,
        static_id=_required_array(particle, "staticId"),
        position_start=_required_array(particle, "positionStart"),
        velocity_start=_required_array(particle, "velocityStart"),
        radius_start=_required_array(particle, "radiusStart"),
        local_feature_start=_required_array(particle, "localFeatureStart"),
        position_end=_required_array(particle, "positionEnd"),
        velocity_end=_required_array(particle, "velocityEnd"),
        gravity_reference_velocity_end=_required_array(
            particle, "gravityReferenceVelocityEnd"
        ),
        target_residual_acceleration=_required_array(
            particle, "targetResidualAcceleration"
        ),
        valid=valid_raw.astype(np.bool_),
        wall_start=_resolved_wall_state(start_states),
        wall_end=_resolved_wall_state(end_states),
        wall_channels_start=wall_channels,
    )
    _validate_particle_arrays(frame)
    if not np.allclose(
        frame.radius_start,
        0.5 * frame.particle_diameter,
        rtol=1.0e-6,
        atol=0.0,
    ):
        raise ContractError("Teacher radiusStart and particleDiameter disagree.")
    return frame


def load_teacher_frame(index: TeacherFrameIndex) -> TeacherFrame:
    """Load and validate exactly one indexed Teacher frame."""

    with h5py.File(index.source_path, "r") as file:
        return _load_teacher_frame_from_file(index, file, _teacher_file_contract(file))


class _TeacherFrameReader:
    """Keep one validated HDF5 handle open per Teacher trajectory."""

    def __init__(self) -> None:
        self._files: dict[Path, h5py.File] = {}
        self._contracts: dict[Path, TeacherFileContract] = {}
        self._static_wall_grids: dict[Path, np.ndarray] = {}

    def _file(self, index: TeacherFrameIndex) -> tuple[h5py.File, TeacherFileContract]:
        path = index.source_path
        if path not in self._files:
            file = h5py.File(path, "r")
            try:
                contract = _teacher_file_contract(file)
            except Exception:
                file.close()
                raise
            self._files[path] = file
            self._contracts[path] = contract
        return self._files[path], self._contracts[path]

    def load_validated(self, index: TeacherFrameIndex) -> TeacherFrame:
        file, contract = self._file(index)
        return _load_teacher_frame_from_file(
            index, file, contract, self._wall_grid(index, file, contract)
        )

    def _wall_grid(
        self,
        index: TeacherFrameIndex,
        file: h5py.File,
        contract: TeacherFileContract,
    ) -> np.ndarray:
        if index.wall_is_static and index.source_path in self._static_wall_grids:
            return self._static_wall_grids[index.source_path]
        try:
            wall = rasterize_wall_grid(file, contract[-1], index.time_start)
        except WallContractError as error:
            raise ContractError(str(error)) from error
        if index.wall_is_static:
            self._static_wall_grids[index.source_path] = wall
        return wall

    def load_training(
        self,
        index: TeacherFrameIndex,
        *,
        include_radius: bool,
        include_wall: bool,
    ) -> TrainingFrameData:
        file, contract = self._file(index)
        current_index = _index_teacher_frame(
            index.source_path,
            index.frame_name,
            file["frames"][index.frame_name],
            contract,
        )
        if current_index != index:
            raise ContractError(f"Teacher frame metadata changed: {index.frame_key}.")
        group = file["frames"][index.frame_name]
        particle = group["particle"]
        valid_raw = _required_array(particle, "valid")
        if (
            valid_raw.shape != (index.particle_count,)
            or not np.isin(valid_raw, (0, 1)).all()
        ):
            raise ContractError("Particle valid mask must contain only 0 or 1.")
        if not np.any(valid_raw):
            raise ContractError("MSE requires at least one valid finite target.")
        position = _required_array(particle, "positionStart")
        local = _required_array(particle, "localFeatureStart")
        target = _required_array(particle, "targetResidualAcceleration")
        if position.shape != (index.particle_count, 3):
            raise ContractError("positionStart must have shape [P,3].")
        if local.shape != (index.particle_count, LOCAL_FEATURE_COUNT):
            raise ContractError("localFeatureStart must have shape [P,18].")
        if target.shape != (index.particle_count, TARGET_CHANNEL_COUNT):
            raise ContractError("targetResidualAcceleration must have shape [P,3].")
        radius = _required_array(particle, "radiusStart") if include_radius else None
        if radius is not None and radius.shape != (index.particle_count,):
            raise ContractError("radiusStart must have shape [P].")
        wall = self._wall_grid(index, file, contract) if include_wall else None
        if wall is not None and wall.shape != (
            WALL_GRID_CHANNEL_COUNT,
            *index.geometry.tensor_shape,
        ):
            raise ContractError("wallStart grid has an invalid shape.")
        return TrainingFrameData(
            index=index,
            position_start=position,
            radius_start=radius,
            local_feature_start=local,
            target_residual_acceleration=target,
            valid=valid_raw.astype(np.bool_),
            wall_channels_start=wall,
        )

    def load_radii(self, index: TeacherFrameIndex) -> np.ndarray:
        file, contract = self._file(index)
        current_index = _index_teacher_frame(
            index.source_path,
            index.frame_name,
            file["frames"][index.frame_name],
            contract,
        )
        if current_index != index:
            raise ContractError(f"Teacher frame metadata changed: {index.frame_key}.")
        radii = _required_array(
            file["frames"][index.frame_name]["particle"], "radiusStart"
        )
        if radii.shape != (index.particle_count,):
            raise ContractError("radiusStart must have shape [P].")
        return radii

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()
        self._contracts.clear()
        self._static_wall_grids.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


_PREFETCH_DONE = object()


@dataclass(frozen=True)
class _PrefetchFailure:
    error: Exception


def _iter_with_reader[LoadedFrame](
    indexes: Sequence[TeacherFrameIndex],
    loader: Callable[[_TeacherFrameReader, TeacherFrameIndex], LoadedFrame],
    *,
    prefetch_frames: int,
) -> Iterable[LoadedFrame]:
    """Load frames in order, optionally overlapping I/O with device work."""

    if prefetch_frames < 0:
        raise ContractError("prefetch_frames must be non-negative.")
    if prefetch_frames == 0:
        with _TeacherFrameReader() as reader:
            for index in indexes:
                yield loader(reader, index)
        return

    pending: queue.Queue[object] = queue.Queue(maxsize=prefetch_frames)
    stop = threading.Event()

    def enqueue(value: object) -> bool:
        while not stop.is_set():
            try:
                pending.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            with _TeacherFrameReader() as reader:
                for index in indexes:
                    if not enqueue(loader(reader, index)):
                        return
        except Exception as error:  # noqa: BLE001 - forward worker failures verbatim.
            enqueue(_PrefetchFailure(error))
        finally:
            enqueue(_PREFETCH_DONE)

    worker = threading.Thread(
        target=produce, name="shondy-teacher-prefetch", daemon=True
    )
    worker.start()
    try:
        while True:
            value = pending.get()
            if value is _PREFETCH_DONE:
                break
            if isinstance(value, _PrefetchFailure):
                raise value.error
            yield cast(LoadedFrame, value)
    finally:
        stop.set()
        worker.join()


def iter_validated_teacher_frames(
    indexes: Sequence[TeacherFrameIndex], *, prefetch_frames: int = 0
) -> Iterable[TeacherFrame]:
    return _iter_with_reader(
        indexes,
        lambda reader, index: reader.load_validated(index),
        prefetch_frames=prefetch_frames,
    )


def iter_training_frame_data(
    indexes: Sequence[TeacherFrameIndex],
    *,
    dynamic_grid_cache: Mapping[tuple[str, str, int], torch.Tensor],
    wall_grid_cache: Mapping[Path, torch.Tensor],
    training_frame_cache: Mapping[tuple[str, str, int], TrainingFrameData],
    prefetch_frames: int = 0,
) -> Iterable[TrainingFrameData]:
    def load(
        reader: _TeacherFrameReader, index: TeacherFrameIndex
    ) -> TrainingFrameData:
        cached = training_frame_cache.get(index.frame_key)
        if cached is not None:
            return cached
        return reader.load_training(
            index,
            include_radius=index.frame_key not in dynamic_grid_cache,
            include_wall=index.source_path not in wall_grid_cache,
        )

    return _iter_with_reader(indexes, load, prefetch_frames=prefetch_frames)


def load_teacher_trajectory(path: str | Path) -> tuple[TeacherFrame, ...]:
    """Eager compatibility loader; training uses the streaming index API."""

    return tuple(load_teacher_frame(index) for index in index_teacher_trajectory(path))


@dataclass(frozen=True)
class TrilinearStencil:
    linear_indices: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor

    def require_complete(self) -> None:
        invalid = torch.nonzero(~self.valid, as_tuple=False).flatten()
        if invalid.numel() != 0:
            sample = invalid[:8].cpu().tolist()
            raise IncompleteGridSupportError(
                f"Particles lack complete 8-cell support: {sample}."
            )


def trilinear_stencil(
    positions: torch.Tensor, geometry: GridGeometry, *, validate: bool = True
) -> TrilinearStencil:
    """Build the deterministic 8-cell stencil without clamping."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ContractError("positions must have shape [P,3].")
    if not positions.is_floating_point() or (
        validate and not torch.isfinite(positions).all()
    ):
        raise ContractError("positions must be finite floating-point values.")
    minimum = torch.tensor(
        geometry.padded_bounds_min,
        dtype=positions.dtype,
        device=positions.device,
    )
    cell_coordinate = (positions - minimum) / geometry.cell_size - 0.5
    lower = torch.floor(cell_coordinate).to(torch.int64)
    fraction = cell_coordinate - lower.to(dtype=positions.dtype)
    counts = torch.tensor(
        geometry.cell_counts, dtype=torch.int64, device=positions.device
    )
    valid = torch.logical_and(lower >= 0, lower + 1 < counts).all(dim=1)
    offsets = torch.tensor(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ),
        dtype=torch.int64,
        device=positions.device,
    )
    xyz = lower[:, None, :] + offsets[None, :, :]
    axis_weights = torch.where(
        offsets[None, :, :] == 0,
        1.0 - fraction[:, None, :],
        fraction[:, None, :],
    )
    weights = axis_weights.prod(dim=2)
    nx, ny, _ = geometry.cell_counts
    linear = xyz[:, :, 2] * (ny * nx) + xyz[:, :, 1] * nx + xyz[:, :, 0]
    linear = torch.where(valid[:, None], linear, torch.full_like(linear, -1))
    return TrilinearStencil(
        linear_indices=linear.contiguous(),
        weights=weights.contiguous(),
        valid=valid.contiguous(),
    )


@dataclass(frozen=True)
class P2GResult:
    dynamic_grid: torch.Tensor
    cell_weight: torch.Tensor
    stencil: TrilinearStencil


def reference_p2g(
    positions: torch.Tensor,
    radii: torch.Tensor,
    velocity: torch.Tensor,
    normalized_number_density: torch.Tensor,
    previous_residual_acceleration: torch.Tensor,
    geometry: GridGeometry,
    *,
    eps: float = 1.0e-12,
    validate: bool = True,
) -> P2GResult:
    """Volume-weight particle quantities onto a dense x-contiguous grid."""

    particle_count = positions.shape[0]
    expected = {
        "radii": (particle_count,),
        "velocity": (particle_count, 3),
        "normalized_number_density": (particle_count,),
        "previous_residual_acceleration": (particle_count, 3),
    }
    values = {
        "radii": radii,
        "velocity": velocity,
        "normalized_number_density": normalized_number_density,
        "previous_residual_acceleration": previous_residual_acceleration,
    }
    if not math.isfinite(eps) or eps <= 0.0:
        raise ContractError("eps must be finite and positive.")
    for name, value in values.items():
        if value.shape != expected[name]:
            raise ContractError(f"{name} has invalid shape.")
        if value.device != positions.device or value.dtype != positions.dtype:
            raise ContractError(f"{name} must match position dtype and device.")
        if validate and not torch.isfinite(value).all():
            raise ContractError(f"{name} must be finite.")
    if validate and torch.any(radii <= 0.0):
        raise ContractError("Particle radii must be positive.")

    stencil = trilinear_stencil(positions, geometry, validate=validate)
    if validate:
        stencil.require_complete()
    volume = particle_volume_from_radius(radii)
    weighted_volume = volume[:, None] * stencil.weights
    linear = stencil.linear_indices.reshape(-1)
    weighted_volume_flat = weighted_volume.reshape(-1)

    cell_weight = torch.zeros(
        geometry.cell_count, dtype=positions.dtype, device=positions.device
    )
    cell_weight.scatter_add_(0, linear, weighted_volume_flat)
    quantities = torch.cat(
        (
            velocity,
            normalized_number_density[:, None],
            previous_residual_acceleration,
        ),
        dim=1,
    )
    numerator = torch.zeros(
        (geometry.cell_count, DYNAMIC_QUANTITY_COUNT),
        dtype=positions.dtype,
        device=positions.device,
    )
    quantity_contribution = (
        weighted_volume[:, :, None] * quantities[:, None, :]
    ).reshape(-1, DYNAMIC_QUANTITY_COUNT)
    numerator.scatter_add_(
        0,
        linear[:, None].expand(-1, DYNAMIC_QUANTITY_COUNT),
        quantity_contribution,
    )
    averaged = numerator / (cell_weight[:, None] + eps)
    occupancy = cell_weight / geometry.cell_volume
    nz, ny, nx = geometry.tensor_shape
    dynamic_grid = torch.cat((averaged, occupancy[:, None]), dim=1)
    dynamic_grid = dynamic_grid.transpose(0, 1).reshape(
        DYNAMIC_GRID_CHANNEL_COUNT, nz, ny, nx
    )
    return P2GResult(
        dynamic_grid=dynamic_grid.contiguous(),
        cell_weight=cell_weight.reshape(nz, ny, nx).contiguous(),
        stencil=stencil,
    )


def reference_g2p(
    grid_latent: torch.Tensor,
    positions: torch.Tensor,
    geometry: GridGeometry,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Interpolate a dense latent grid to particles with the P2G stencil."""

    if grid_latent.ndim == 5:
        if grid_latent.shape[0] != 1:
            raise ContractError("Reference G2P accepts one frame at a time.")
        grid_latent = grid_latent[0]
    if grid_latent.ndim != 4 or tuple(grid_latent.shape[1:]) != geometry.tensor_shape:
        raise ContractError("grid_latent must have shape [C,Nz,Ny,Nx].")
    if grid_latent.device != positions.device or grid_latent.dtype != positions.dtype:
        raise ContractError("grid_latent must match position dtype and device.")
    stencil = trilinear_stencil(positions, geometry, validate=validate)
    if validate:
        stencil.require_complete()
    channels = grid_latent.shape[0]
    flat = grid_latent.reshape(channels, geometry.cell_count)
    gathered = flat[:, stencil.linear_indices]
    return torch.sum(
        gathered.permute(1, 0, 2) * stencil.weights[:, None, :], dim=2
    ).contiguous()


@dataclass(frozen=True)
class FeatureStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    count: int
    constant_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ContractError("Statistics must be one-dimensional and aligned.")
        if self.constant_mask.shape != self.mean.shape:
            raise ContractError("constant_mask must match statistics shape.")
        if self.count <= 0:
            raise ContractError("Statistics require at least one sample.")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ContractError("Statistics must be finite.")
        if torch.any(self.std <= 0.0):
            raise ContractError("Statistics std must be positive.")

    @property
    def feature_count(self) -> int:
        return int(self.mean.numel())

    def to(
        self, *, dtype: torch.dtype, device: torch.device | str
    ) -> FeatureStatistics:
        return FeatureStatistics(
            mean=self.mean.to(dtype=dtype, device=device),
            std=self.std.to(dtype=dtype, device=device),
            count=self.count,
            constant_mask=self.constant_mask.to(device=device),
        )

    def to_metadata(self, names: Sequence[str]) -> dict[str, object]:
        if len(names) != self.feature_count:
            raise ContractError("Statistic names do not match feature count.")
        return {
            "names": list(names),
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "count": self.count,
            "constantMask": self.constant_mask.detach().cpu().tolist(),
        }


class _RunningStatistics:
    def __init__(self, feature_count: int) -> None:
        self.feature_count = feature_count
        self.count = 0
        self.mean = torch.zeros(feature_count, dtype=torch.float64)
        self.m2 = torch.zeros(feature_count, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.feature_count:
            raise ContractError("Statistics input has invalid shape.")
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise ContractError("Statistics input must be finite.")
        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0)
        batch_m2 = torch.sum((values - batch_mean) ** 2, dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2 + batch_m2 + delta * delta * (self.count * batch_count / total)
        )
        self.count = total

    def finish(self, *, constant_epsilon: float = 1.0e-12) -> FeatureStatistics:
        if self.count == 0:
            raise ContractError("Cannot compute statistics from zero samples.")
        variance = self.m2 / self.count
        raw_std = torch.sqrt(torch.clamp(variance, min=0.0))
        constant_mask = raw_std <= constant_epsilon
        std = torch.where(constant_mask, torch.ones_like(raw_std), raw_std)
        return FeatureStatistics(
            mean=self.mean,
            std=std,
            count=self.count,
            constant_mask=constant_mask,
        )


def compute_feature_statistics(
    batches: Iterable[torch.Tensor], feature_count: int
) -> FeatureStatistics:
    running = _RunningStatistics(feature_count)
    for batch in batches:
        running.update(batch)
    return running.finish()


def normalize_dynamic_grid(
    dynamic_grid: torch.Tensor,
    statistics: FeatureStatistics,
    *,
    occupancy_epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Normalize deposited quantities while preserving exact empty-cell zero."""

    if dynamic_grid.ndim not in (4, 5):
        raise ContractError("dynamic_grid must be [8,D,H,W] or [N,8,D,H,W].")
    channel_axis = 0 if dynamic_grid.ndim == 4 else 1
    if dynamic_grid.shape[channel_axis] != DYNAMIC_GRID_CHANNEL_COUNT:
        raise ContractError("dynamic_grid must contain 8 channels.")
    if statistics.feature_count != DYNAMIC_QUANTITY_COUNT:
        raise ContractError("Dynamic statistics must contain 7 quantities.")
    if occupancy_epsilon <= 0.0 or not math.isfinite(occupancy_epsilon):
        raise ContractError("occupancy_epsilon must be finite and positive.")
    stats = statistics.to(dtype=dynamic_grid.dtype, device=dynamic_grid.device)
    if dynamic_grid.ndim == 4:
        quantity = dynamic_grid[:DYNAMIC_QUANTITY_COUNT]
        occupancy = dynamic_grid[DYNAMIC_QUANTITY_COUNT:]
        shape = (DYNAMIC_QUANTITY_COUNT, 1, 1, 1)
    else:
        quantity = dynamic_grid[:, :DYNAMIC_QUANTITY_COUNT]
        occupancy = dynamic_grid[:, DYNAMIC_QUANTITY_COUNT:]
        shape = (1, DYNAMIC_QUANTITY_COUNT, 1, 1, 1)
    normalized = (quantity - stats.mean.view(shape)) / stats.std.view(shape)
    normalized = torch.where(
        occupancy > occupancy_epsilon, normalized, torch.zeros_like(normalized)
    )
    return torch.cat((normalized, occupancy), dim=channel_axis).contiguous()


def validate_wall_channels(wall_channels: torch.Tensor, geometry: GridGeometry) -> None:
    if wall_channels.ndim != 4 or wall_channels.shape != (
        WALL_GRID_CHANNEL_COUNT,
        *geometry.tensor_shape,
    ):
        raise ContractError("wall_channels must have shape [8,Nz,Ny,Nx].")
    if not torch.isfinite(wall_channels).all():
        raise ContractError("wall_channels must be finite.")
    wall_band = wall_channels[0]
    if torch.any(wall_band < 0.0) or torch.any(wall_band > 1.0):
        raise ContractError("wallBand must be in [0,1].")
    outside = wall_band <= 0.0
    if torch.any(wall_channels[1:7, outside] != 0.0):
        raise ContractError("Wall normal and velocity must be zero outside wallBand.")
    expected_mask = (wall_band <= 0.0).to(dtype=wall_channels.dtype)
    if not torch.equal(wall_channels[7], expected_mask):
        raise ContractError("validDomainMask must be one outside the wall band.")


def assemble_grid_input(
    normalized_dynamic_grid: torch.Tensor,
    wall_channels: torch.Tensor,
    geometry: GridGeometry,
    condition_values: torch.Tensor,
    condition_statistics: FeatureStatistics | None,
    *,
    validate_wall: bool = True,
) -> torch.Tensor:
    """Assemble one `[1,19+K,Nz,Ny,Nx]` grid encoder input."""

    if normalized_dynamic_grid.shape != (
        DYNAMIC_GRID_CHANNEL_COUNT,
        *geometry.tensor_shape,
    ):
        raise ContractError("normalized_dynamic_grid has invalid shape.")
    if validate_wall:
        validate_wall_channels(wall_channels, geometry)
    if normalized_dynamic_grid.dtype != wall_channels.dtype or (
        normalized_dynamic_grid.device != wall_channels.device
    ):
        raise ContractError("Dynamic and wall grids must share dtype and device.")
    if condition_values.ndim != 1:
        raise ContractError("condition_values must be one-dimensional.")
    if condition_values.device != wall_channels.device:
        condition_values = condition_values.to(device=wall_channels.device)
    condition_values = condition_values.to(dtype=wall_channels.dtype)
    if condition_values.numel() == 0:
        if condition_statistics is not None:
            raise ContractError("Empty condition schema must not have statistics.")
        normalized_conditions = condition_values
    else:
        if condition_statistics is None or (
            condition_statistics.feature_count != condition_values.numel()
        ):
            raise ContractError("Condition statistics do not match schema.")
        stats = condition_statistics.to(
            dtype=condition_values.dtype, device=condition_values.device
        )
        normalized_conditions = (condition_values - stats.mean) / stats.std
    coordinate_channels = geometry.coordinate_channels(
        dtype=wall_channels.dtype, device=wall_channels.device
    )
    condition_channels = normalized_conditions.view(-1, 1, 1, 1).expand(
        -1, *geometry.tensor_shape
    )
    result = torch.cat(
        (
            normalized_dynamic_grid,
            wall_channels,
            coordinate_channels,
            condition_channels,
        ),
        dim=0,
    )
    return result.unsqueeze(0).contiguous()


@dataclass(frozen=True)
class HybridFrameSample:
    frame: TeacherFrame
    wall_channels: torch.Tensor


@dataclass(frozen=True)
class TrainingStatistics:
    dynamic: FeatureStatistics
    condition: FeatureStatistics | None
    target: FeatureStatistics
    latent: FeatureStatistics | None = None


@dataclass(frozen=True)
class PreparedFrame:
    frame_key: tuple[str, str, int]
    geometry: GridGeometry
    grid_input: torch.Tensor
    positions: torch.Tensor
    local_features: torch.Tensor
    standardized_target: torch.Tensor
    valid: torch.Tensor


def _frame_tensors(frame: TeacherFrame) -> dict[str, torch.Tensor]:
    dtype = torch.float64 if frame.position_start.dtype == np.float64 else torch.float32
    return {
        "positions": torch.as_tensor(frame.position_start, dtype=dtype),
        "radii": torch.as_tensor(frame.radius_start, dtype=dtype),
        "velocity": torch.as_tensor(frame.velocity_start, dtype=dtype),
        "normalized_density": torch.as_tensor(
            frame.local_feature_start[:, 3], dtype=dtype
        ),
        "previous_residual": torch.as_tensor(
            frame.local_feature_start[:, 8:11], dtype=dtype
        ),
        "local_features": torch.as_tensor(frame.local_feature_start, dtype=dtype),
        "target": torch.as_tensor(frame.target_residual_acceleration, dtype=dtype),
        "valid": torch.as_tensor(frame.valid, dtype=torch.bool),
        "conditions": torch.as_tensor(frame.conditions, dtype=dtype),
    }


def _training_data_tensors(
    frame: TrainingFrameData,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    if dtype is None:
        dtype = (
            torch.float64 if frame.position_start.dtype == np.float64 else torch.float32
        )

    def transfer(value: np.ndarray, *, tensor_dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=tensor_dtype).to(
            device=device, non_blocking=True
        )

    local = transfer(frame.local_feature_start, tensor_dtype=dtype)
    result = {
        "positions": transfer(frame.position_start, tensor_dtype=dtype),
        "velocity": local[:, 0:3],
        "normalized_density": local[:, 3],
        "previous_residual": local[:, 8:11],
        "local_features": local,
        "target": transfer(frame.target_residual_acceleration, tensor_dtype=dtype),
        "valid": transfer(frame.valid, tensor_dtype=torch.bool),
        "conditions": torch.as_tensor(frame.index.conditions, dtype=dtype).to(
            device=device, non_blocking=True
        ),
    }
    if frame.radius_start is not None:
        result["radii"] = transfer(frame.radius_start, tensor_dtype=dtype)
    return result


def require_common_dataset_contract(samples: Sequence[HybridFrameSample]) -> None:
    if not samples:
        raise ContractError("At least one frame sample is required.")
    first = samples[0].frame
    for sample in samples:
        frame = sample.frame
        if frame.geometry != first.geometry:
            raise ContractError("All frames for one model must share grid geometry.")
        if frame.condition_names != first.condition_names:
            raise ContractError("All frames must share the named condition schema.")
        if frame.ai_delta_time != first.ai_delta_time:
            raise ContractError("All frames must share aiDeltaTime.")


def require_common_frame_contract(
    frames: Sequence[TeacherFrame | TeacherFrameIndex],
) -> None:
    if not frames:
        raise ContractError("At least one Teacher frame index is required.")
    first = frames[0]
    frame_keys: set[tuple[str, str, int]] = set()
    for frame in frames:
        if frame.geometry != first.geometry:
            raise ContractError("All frames for one model must share grid geometry.")
        if frame.condition_names != first.condition_names:
            raise ContractError("All frames must share the named condition schema.")
        if frame.ai_delta_time != first.ai_delta_time:
            raise ContractError("All frames must share aiDeltaTime.")
        if frame.frame_key in frame_keys:
            raise ContractError("Duplicate frame key across Teacher trajectories.")
        frame_keys.add(frame.frame_key)


def compute_base_training_statistics(
    samples: Sequence[HybridFrameSample],
    split: TrajectorySplit,
) -> TrainingStatistics:
    """Compute all non-latent statistics from valid training trajectories."""

    require_common_dataset_contract(samples)
    dynamic_running = _RunningStatistics(DYNAMIC_QUANTITY_COUNT)
    condition_count = len(samples[0].frame.condition_names)
    condition_running = (
        _RunningStatistics(condition_count) if condition_count != 0 else None
    )
    target_running = _RunningStatistics(TARGET_CHANNEL_COUNT)
    accepted = 0
    for sample in samples:
        frame = sample.frame
        if split.assignment(frame) != "training":
            continue
        if not frame.valid_grid_support:
            continue
        tensors = _frame_tensors(frame)
        p2g = reference_p2g(
            tensors["positions"],
            tensors["radii"],
            tensors["velocity"],
            tensors["normalized_density"],
            tensors["previous_residual"],
            frame.geometry,
        )
        occupied = p2g.dynamic_grid[7] > 1.0e-12
        quantities = p2g.dynamic_grid[:7, occupied].transpose(0, 1)
        dynamic_running.update(quantities)
        if condition_running is not None:
            condition_running.update(tensors["conditions"].view(1, -1))
        target_running.update(tensors["target"][tensors["valid"]])
        accepted += 1
    if accepted == 0:
        raise ContractError("Training split has no frame with valid grid support.")
    return TrainingStatistics(
        dynamic=dynamic_running.finish(),
        condition=(condition_running.finish() if condition_running else None),
        target=target_running.finish(),
    )


def compute_base_training_statistics_streaming(
    indexes: Sequence[TeacherFrameIndex],
    split: TrajectorySplit,
    *,
    device: torch.device | str = "cpu",
    dynamic_grid_cache: MutableMapping[tuple[str, str, int], torch.Tensor]
    | None = None,
    dynamic_grid_cache_max_bytes: int = 0,
    wall_grid_cache: MutableMapping[Path, torch.Tensor] | None = None,
    training_frame_cache: MutableMapping[tuple[str, str, int], TrainingFrameData]
    | None = None,
    training_frame_cache_max_bytes: int = 0,
    prefetch_frames: int = 0,
) -> TrainingStatistics:
    """First pass: compute training-only statistics one indexed frame at a time."""

    require_common_frame_contract(indexes)
    if dynamic_grid_cache_max_bytes < 0 or training_frame_cache_max_bytes < 0:
        raise ContractError("Training cache byte limits must be non-negative.")
    dynamic_running = _RunningStatistics(DYNAMIC_QUANTITY_COUNT)
    condition_count = len(indexes[0].condition_names)
    condition_running = (
        _RunningStatistics(condition_count) if condition_count != 0 else None
    )
    target_running = _RunningStatistics(TARGET_CHANNEL_COUNT)
    selected_indexes = [
        index
        for index in indexes
        if split.assignment(index) == "training" and index.valid_grid_support
    ]
    indexes_by_key = {index.frame_key: index for index in selected_indexes}
    accepted = 0
    cached_bytes = (
        sum(
            value.numel() * value.element_size()
            for value in dynamic_grid_cache.values()
        )
        if dynamic_grid_cache is not None
        else 0
    )
    cached_training_bytes = (
        sum(value.nbytes for value in training_frame_cache.values())
        if training_frame_cache is not None
        else 0
    )
    for frame in iter_validated_teacher_frames(
        selected_indexes, prefetch_frames=prefetch_frames
    ):
        if not np.any(frame.valid):
            raise ContractError("MSE requires at least one valid finite target.")
        tensors = _frame_tensors(frame)
        tensors = {
            name: value.to(device=device, non_blocking=True)
            for name, value in tensors.items()
        }
        p2g = reference_p2g(
            tensors["positions"],
            tensors["radii"],
            tensors["velocity"],
            tensors["normalized_density"],
            tensors["previous_residual"],
            frame.geometry,
        )
        occupied = p2g.dynamic_grid[7] > 1.0e-12
        dynamic_running.update(p2g.dynamic_grid[:7, occupied].transpose(0, 1))
        if dynamic_grid_cache is not None and frame.frame_key not in dynamic_grid_cache:
            grid_bytes = p2g.dynamic_grid.numel() * p2g.dynamic_grid.element_size()
            if cached_bytes + grid_bytes <= dynamic_grid_cache_max_bytes:
                dynamic_grid_cache[frame.frame_key] = (
                    p2g.dynamic_grid.detach().to(device="cpu").contiguous()
                )
                cached_bytes += grid_bytes
        if (
            wall_grid_cache is not None
            and indexes_by_key[frame.frame_key].wall_is_static
            and frame.source_path not in wall_grid_cache
        ):
            wall_grid_cache[frame.source_path] = (
                torch.as_tensor(frame.wall_channels_start).clone().contiguous()
            )
        if (
            training_frame_cache is not None
            and frame.frame_key not in training_frame_cache
        ):
            cached_frame = TrainingFrameData(
                index=indexes_by_key[frame.frame_key],
                position_start=frame.position_start,
                radius_start=(
                    None
                    if dynamic_grid_cache is not None
                    and frame.frame_key in dynamic_grid_cache
                    else frame.radius_start
                ),
                local_feature_start=frame.local_feature_start,
                target_residual_acceleration=frame.target_residual_acceleration,
                valid=frame.valid,
                wall_channels_start=(
                    None
                    if wall_grid_cache is not None
                    and frame.source_path in wall_grid_cache
                    else frame.wall_channels_start
                ),
            )
            if (
                cached_training_bytes + cached_frame.nbytes
                <= training_frame_cache_max_bytes
            ):
                training_frame_cache[frame.frame_key] = cached_frame
                cached_training_bytes += cached_frame.nbytes
        if condition_running is not None:
            condition_running.update(tensors["conditions"].view(1, -1))
        target_running.update(tensors["target"][tensors["valid"]])
        accepted += 1
    if accepted == 0:
        raise ContractError("Training split has no frame with valid grid support.")
    return TrainingStatistics(
        dynamic=dynamic_running.finish(),
        condition=(condition_running.finish() if condition_running else None),
        target=target_running.finish(),
    )


def prepare_indexed_frame(
    index: TeacherFrameIndex,
    statistics: TrainingStatistics,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PreparedFrame:
    frame = load_teacher_frame(index)
    return prepare_frame(
        HybridFrameSample(
            frame=frame,
            wall_channels=torch.as_tensor(frame.wall_channels_start),
        ),
        statistics,
        device=device,
        dtype=dtype,
    )


def prepare_frame(
    sample: HybridFrameSample,
    statistics: TrainingStatistics,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PreparedFrame:
    frame = sample.frame
    if not frame.valid_grid_support:
        raise IncompleteGridSupportError(
            f"Frame {frame.frame_key} is marked invalid for grid support."
        )
    tensors = _frame_tensors(frame)
    if device is not None:
        tensors = {
            name: value.to(
                device=device,
                dtype=(dtype if value.is_floating_point() else value.dtype),
                non_blocking=True,
            )
            for name, value in tensors.items()
        }
    wall_channels = sample.wall_channels.to(
        dtype=tensors["positions"].dtype, device=tensors["positions"].device
    )
    p2g = reference_p2g(
        tensors["positions"],
        tensors["radii"],
        tensors["velocity"],
        tensors["normalized_density"],
        tensors["previous_residual"],
        frame.geometry,
    )
    normalized_dynamic = normalize_dynamic_grid(p2g.dynamic_grid, statistics.dynamic)
    grid_input = assemble_grid_input(
        normalized_dynamic,
        wall_channels,
        frame.geometry,
        tensors["conditions"],
        statistics.condition,
    )
    target_stats = statistics.target.to(
        dtype=tensors["target"].dtype, device=tensors["target"].device
    )
    standardized_target = torch.full_like(tensors["target"], float("nan"))
    valid = tensors["valid"]
    standardized_target[valid] = (
        tensors["target"][valid] - target_stats.mean
    ) / target_stats.std
    return PreparedFrame(
        frame_key=frame.frame_key,
        geometry=frame.geometry,
        grid_input=grid_input,
        positions=tensors["positions"],
        local_features=tensors["local_features"],
        standardized_target=standardized_target,
        valid=valid,
    )


def prepare_training_frame(
    frame: TrainingFrameData,
    statistics: TrainingStatistics,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None,
    dynamic_grid: torch.Tensor | None,
    wall_channels: torch.Tensor | None,
) -> PreparedFrame:
    """Prepare an already validated frame without reloading unused Teacher fields."""

    if not frame.index.valid_grid_support:
        raise IncompleteGridSupportError(
            f"Frame {frame.frame_key} is marked invalid for grid support."
        )
    tensors = _training_data_tensors(frame, device=device, dtype=dtype)
    if dynamic_grid is None:
        radii = tensors.get("radii")
        if radii is None:
            raise ContractError("Uncached training frames require radiusStart.")
        dynamic_grid = reference_p2g(
            tensors["positions"],
            radii,
            tensors["velocity"],
            tensors["normalized_density"],
            tensors["previous_residual"],
            frame.geometry,
            validate=False,
        ).dynamic_grid
    else:
        dynamic_grid = dynamic_grid.to(
            device=device, dtype=tensors["positions"].dtype, non_blocking=True
        )
    normalized_dynamic = normalize_dynamic_grid(dynamic_grid, statistics.dynamic)
    if wall_channels is None:
        if frame.wall_channels_start is None:
            raise ContractError("Uncached training frames require wallStart.")
        wall_channels = torch.as_tensor(frame.wall_channels_start).to(
            device=device, dtype=tensors["positions"].dtype, non_blocking=True
        )
    else:
        wall_channels = wall_channels.to(
            device=device, dtype=tensors["positions"].dtype, non_blocking=True
        )
    grid_input = assemble_grid_input(
        normalized_dynamic,
        wall_channels,
        frame.geometry,
        tensors["conditions"],
        statistics.condition,
        validate_wall=False,
    )
    target_stats = statistics.target.to(
        dtype=tensors["target"].dtype, device=tensors["target"].device
    )
    standardized_target = torch.full_like(tensors["target"], float("nan"))
    valid = tensors["valid"]
    standardized_target[valid] = (
        tensors["target"][valid] - target_stats.mean
    ) / target_stats.std
    return PreparedFrame(
        frame_key=frame.frame_key,
        geometry=frame.geometry,
        grid_input=grid_input,
        positions=tensors["positions"],
        local_features=tensors["local_features"],
        standardized_target=standardized_target,
        valid=valid,
    )


def iter_prepared_training_frames(
    indexes: Sequence[TeacherFrameIndex],
    statistics: TrainingStatistics,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    dynamic_grid_cache: Mapping[tuple[str, str, int], torch.Tensor] | None = None,
    wall_grid_cache: Mapping[Path, torch.Tensor] | None = None,
    training_frame_cache: Mapping[tuple[str, str, int], TrainingFrameData]
    | None = None,
    prefetch_frames: int = 0,
    validated_indexes: bool = False,
) -> Iterable[PreparedFrame]:
    dynamic_cache = dynamic_grid_cache or {}
    wall_cache = wall_grid_cache or {}
    frame_cache = training_frame_cache or {}
    if not validated_indexes:
        for frame in iter_validated_teacher_frames(
            indexes, prefetch_frames=prefetch_frames
        ):
            yield prepare_frame(
                HybridFrameSample(
                    frame=frame,
                    wall_channels=torch.as_tensor(frame.wall_channels_start),
                ),
                statistics,
                device=device,
                dtype=dtype,
            )
        return
    for frame in iter_training_frame_data(
        indexes,
        dynamic_grid_cache=dynamic_cache,
        wall_grid_cache=wall_cache,
        training_frame_cache=frame_cache,
        prefetch_frames=prefetch_frames,
    ):
        yield prepare_training_frame(
            frame,
            statistics,
            device=device,
            dtype=dtype,
            dynamic_grid=dynamic_cache.get(frame.frame_key),
            wall_channels=wall_cache.get(frame.index.source_path),
        )


@dataclass(frozen=True)
class TrajectorySplit:
    training: tuple[tuple[str, str], ...]
    validation: tuple[tuple[str, str], ...]
    test: tuple[tuple[str, str], ...]

    def assignment_key(self, key: tuple[str, str]) -> str:
        matches = [
            name
            for name, values in (
                ("training", self.training),
                ("validation", self.validation),
                ("test", self.test),
            )
            if key in values
        ]
        if len(matches) != 1:
            raise ContractError(f"Trajectory {key} has invalid split assignment.")
        return matches[0]

    def assignment(self, frame: TeacherFrame | TeacherFrameIndex) -> str:
        return self.assignment_key(frame.trajectory_key)

    def to_metadata(
        self, frames: Sequence[TeacherFrame | TeacherFrameIndex]
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for split_name in ("training", "validation", "test"):
            keys = getattr(self, split_name)
            result[split_name] = {
                "trajectories": [
                    {"caseId": case_id, "trajectoryId": trajectory_id}
                    for case_id, trajectory_id in keys
                ],
                "frames": [
                    {
                        "caseId": frame.case_id,
                        "trajectoryId": frame.trajectory_id,
                        "macroStepIndex": frame.macro_step_index,
                    }
                    for frame in frames
                    if frame.trajectory_key in keys
                ],
            }
        return result


def split_trajectories(
    frames: Sequence[TeacherFrame | TeacherFrameIndex],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 1729,
) -> TrajectorySplit:
    """Assign whole trajectories, never particles or frames, to a split."""

    if len(fractions) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in fractions
    ):
        raise ContractError("Split fractions must be three finite non-negative values.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ContractError("Split fractions must sum to one.")
    keys = sorted({frame.trajectory_key for frame in frames})
    if not keys:
        raise ContractError("Cannot split an empty frame collection.")
    frame_keys = [frame.frame_key for frame in frames]
    if len(set(frame_keys)) != len(frame_keys):
        raise ContractError("Duplicate frame key across Teacher trajectories.")
    random.Random(seed).shuffle(keys)
    raw_counts = [len(keys) * value for value in fractions]
    counts = [math.floor(value) for value in raw_counts]
    remainder = len(keys) - sum(counts)
    priority = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in priority[:remainder]:
        counts[index] += 1
    first = counts[0]
    second = first + counts[1]
    result = TrajectorySplit(
        training=tuple(sorted(keys[:first])),
        validation=tuple(sorted(keys[first:second])),
        test=tuple(sorted(keys[second:])),
    )
    for frame in frames:
        result.assignment(frame)
    return result


class ContractGroupNorm3d(torch.nn.Module):
    """Explicit GroupNorm arithmetic with a stable ONNX representation."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ContractError("GroupNorm channels must be divisible by groups.")
        self.channels = channels
        self.groups = groups
        self.eps = 1.0e-5
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.bias = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.channels:
            raise RuntimeError("GroupNorm input must have shape [N,C,D,H,W].")
        shape = value.shape
        grouped = value.reshape(shape[0], self.groups, -1)
        mean = grouped.mean(dim=2, keepdim=True)
        centered = grouped - mean
        variance = (centered * centered).mean(dim=2, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(shape)
        return normalized * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(
            1, -1, 1, 1, 1
        )


class DoubleConv3d(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, output_channels, 3, padding=1),
            ContractGroupNorm3d(output_channels, 8),
            torch.nn.ReLU(),
            torch.nn.Conv3d(output_channels, output_channels, 3, padding=1),
            ContractGroupNorm3d(output_channels, 8),
            torch.nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class DownStage3d(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.downsample = torch.nn.Conv3d(
            input_channels, output_channels, 3, stride=2, padding=1
        )
        self.block = DoubleConv3d(output_channels, output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(self.downsample(value))


class UpStage3d(torch.nn.Module):
    def __init__(
        self, input_channels: int, skip_channels: int, output_channels: int
    ) -> None:
        super().__init__()
        self.project = torch.nn.Conv3d(input_channels, output_channels, 3, padding=1)
        self.block = DoubleConv3d(output_channels + skip_channels, output_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value, size=skip.shape[-3:], mode="trilinear", align_corners=False
        )
        value = self.project(value)
        return self.block(torch.cat((value, skip), dim=1))


class ProfiledGridEncoder(torch.nn.Module):
    """Profile-selected 3D U-Net with an explicit linear 16-channel head."""

    def __init__(self, input_channels: int, model_profile: str) -> None:
        super().__init__()
        if input_channels < BASE_GRID_CHANNEL_COUNT:
            raise ContractError("Grid encoder requires at least 19 input channels.")
        if model_profile not in MODEL_PROFILES:
            raise ContractError(f"Unknown model profile: {model_profile}.")
        profile = MODEL_PROFILES[model_profile]
        encoder = profile["encoderWidths"]
        decoder = profile["decoderWidths"]
        if len(encoder) != 4 or len(decoder) != 3:
            raise ContractError(
                "Grid encoder profile must define four down and three up widths."
            )
        self.input_channels = input_channels
        self.model_profile = model_profile
        self.encoder_widths = encoder
        self.decoder_widths = decoder
        self.full = DoubleConv3d(input_channels, encoder[0])
        self.down_half = DownStage3d(encoder[0], encoder[1])
        self.down_quarter = DownStage3d(encoder[1], encoder[2])
        self.down_eighth = DownStage3d(encoder[2], encoder[3])
        self.up_quarter = UpStage3d(encoder[3], encoder[2], decoder[0])
        self.up_half = UpStage3d(decoder[0], encoder[1], decoder[1])
        self.up_full = UpStage3d(decoder[1], encoder[0], decoder[2])
        self.output = torch.nn.Conv3d(decoder[2], LATENT_CHANNEL_COUNT, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.input_channels:
            raise RuntimeError("Grid input must have shape [N,C,Nz,Ny,Nx].")
        original_depth = value.shape[-3]
        original_height = value.shape[-2]
        original_width = value.shape[-1]
        pad_depth = (8 - original_depth % 8) % 8
        pad_height = (8 - original_height % 8) % 8
        pad_width = (8 - original_width % 8) % 8
        value = F.pad(value, (0, pad_width, 0, pad_height, 0, pad_depth))
        full = self.full(value)
        half = self.down_half(full)
        quarter = self.down_quarter(half)
        eighth = self.down_eighth(quarter)
        value = self.up_quarter(eighth, quarter)
        value = self.up_half(value, half)
        value = self.up_full(value, full)
        value = self.output(value)
        return value[
            :, :, :original_depth, :original_height, :original_width
        ].contiguous()


class CompactGridEncoder(ProfiledGridEncoder):
    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels, "compact-v1")


class LargeGridEncoder(ProfiledGridEncoder):
    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels, "large-v1")


class ParticleMLP(torch.nn.Module):
    """Frozen 34 -> 128 -> 128 -> 64 -> 3 particle branch."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(PARTICLE_MLP_WIDTHS[0], PARTICLE_MLP_WIDTHS[1])
        self.layer2 = torch.nn.Linear(PARTICLE_MLP_WIDTHS[1], PARTICLE_MLP_WIDTHS[2])
        self.layer3 = torch.nn.Linear(PARTICLE_MLP_WIDTHS[2], PARTICLE_MLP_WIDTHS[3])
        self.output = torch.nn.Linear(PARTICLE_MLP_WIDTHS[3], PARTICLE_MLP_WIDTHS[4])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != 34:
            raise RuntimeError("Particle MLP input must have shape [P,34].")
        value = torch.relu(self.layer1(value))
        value = torch.relu(self.layer2(value))
        value = torch.relu(self.layer3(value))
        return self.output(value)


class HybridReferenceModel(torch.nn.Module):
    def __init__(
        self,
        condition_count: int,
        model_profile: str = DEFAULT_MODEL_PROFILE,
    ) -> None:
        super().__init__()
        if condition_count < 0:
            raise ContractError("condition_count must be non-negative.")
        if model_profile not in MODEL_PROFILES:
            raise ContractError(f"Unknown model profile: {model_profile}.")
        self.condition_count = condition_count
        self.model_profile = model_profile
        self.grid_encoder = ProfiledGridEncoder(
            BASE_GRID_CHANNEL_COUNT + condition_count, model_profile
        )
        self.particle_mlp = ParticleMLP()
        self.register_buffer("latent_mean", torch.zeros(LATENT_CHANNEL_COUNT))
        self.register_buffer("latent_std", torch.ones(LATENT_CHANNEL_COUNT))

    def raw_grid_latent(self, grid_input: torch.Tensor) -> torch.Tensor:
        return self.grid_encoder(grid_input)

    def standardized_grid_latent(self, grid_input: torch.Tensor) -> torch.Tensor:
        raw = self.raw_grid_latent(grid_input)
        mean = self.latent_mean.to(dtype=raw.dtype).view(1, -1, 1, 1, 1)
        std = self.latent_std.to(dtype=raw.dtype).view(1, -1, 1, 1, 1)
        return (raw - mean) / std

    def forward(
        self,
        grid_input: torch.Tensor,
        positions: torch.Tensor,
        local_features: torch.Tensor,
        geometry: GridGeometry,
        validate_inputs: bool = True,
    ) -> torch.Tensor:
        if local_features.shape != (positions.shape[0], LOCAL_FEATURE_COUNT):
            raise ContractError("local_features must have shape [P,18].")
        grid_latent = self.standardized_grid_latent(grid_input)
        particle_latent = reference_g2p(
            grid_latent, positions, geometry, validate=validate_inputs
        )
        return self.particle_mlp(torch.cat((local_features, particle_latent), dim=1))


def model_parameter_counts(model: HybridReferenceModel) -> dict[str, int]:
    grid_count = sum(parameter.numel() for parameter in model.grid_encoder.parameters())
    particle_count = sum(
        parameter.numel() for parameter in model.particle_mlp.parameters()
    )
    return {
        "grid": grid_count,
        "particleMlp": particle_count,
        "total": grid_count + particle_count,
    }


def model_architecture(model: HybridReferenceModel) -> dict[str, object]:
    counts = model_parameter_counts(model)
    profile = MODEL_PROFILES[model.model_profile]
    return {
        "modelProfile": model.model_profile,
        "gridEncoderWidths": list(profile["encoderWidths"]),
        "decoderWidths": list(profile["decoderWidths"]),
        "latentWidth": LATENT_CHANNEL_COUNT,
        "particleMlpWidths": list(PARTICLE_MLP_WIDTHS),
        "conditionWidth": model.condition_count,
        "gridParameterCount": counts["grid"],
        "particleMlpParameterCount": counts["particleMlp"],
        "totalParameterCount": counts["total"],
    }


def standardized_target_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    validated: bool = False,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != TARGET_CHANNEL_COUNT:
        raise ContractError("Prediction and target must have aligned [P,3] shape.")
    if valid.shape != (prediction.shape[0],) or valid.dtype != torch.bool:
        raise ContractError("valid must be a boolean particle mask.")
    selected = valid
    if not validated:
        selected = torch.logical_and(valid, torch.isfinite(target).all(dim=1))
        if not torch.any(selected):
            raise ContractError("MSE requires at least one valid finite target.")
    return F.mse_loss(prediction[selected], target[selected])


def normalized_position_error(
    position_start: torch.Tensor,
    velocity_start: torch.Tensor,
    velocity_end: torch.Tensor,
    position_teacher_end: torch.Tensor,
    radii: torch.Tensor,
    ai_delta_time: float,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen particle-diameter-normalized validation metric."""

    particle_count = position_start.shape[0]
    vector_shape = (particle_count, 3)
    if any(
        value.shape != vector_shape
        for value in (
            position_start,
            velocity_start,
            velocity_end,
            position_teacher_end,
        )
    ):
        raise ContractError("Position metric vector arrays must have shape [P,3].")
    if radii.shape != (particle_count,) or valid.shape != (particle_count,):
        raise ContractError("Position metric radius/mask arrays have invalid shape.")
    if valid.dtype != torch.bool or not math.isfinite(ai_delta_time):
        raise ContractError("Position metric requires a boolean mask and finite dt.")
    if ai_delta_time <= 0.0 or torch.any(radii <= 0.0):
        raise ContractError("Position metric requires positive dt and radii.")
    selected = torch.logical_and(
        valid,
        torch.isfinite(position_teacher_end).all(dim=1)
        & torch.isfinite(velocity_end).all(dim=1),
    )
    if not torch.any(selected):
        raise ContractError("Position metric requires at least one valid particle.")
    predicted_position = position_start[selected] + 0.5 * ai_delta_time * (
        velocity_start[selected] + velocity_end[selected]
    )
    error = torch.linalg.vector_norm(
        position_teacher_end[selected] - predicted_position, dim=1
    )
    return error / (2.0 * radii[selected])


def compute_particle_latent_statistics(
    model: HybridReferenceModel,
    samples: Sequence[PreparedFrame],
    split: TrajectorySplit,
) -> FeatureStatistics:
    """Measure raw G2P latent on real training-split particle positions."""

    if not samples:
        raise ContractError("Latent statistics require training frames.")
    running = _RunningStatistics(LATENT_CHANNEL_COUNT)
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sample in samples:
            if split.assignment_key(sample.frame_key[:2]) != "training":
                continue
            grid_input = sample.grid_input.to(
                device=parameter.device, dtype=parameter.dtype
            )
            positions = sample.positions.to(
                device=parameter.device, dtype=parameter.dtype
            )
            raw_grid = model.raw_grid_latent(grid_input)
            running.update(reference_g2p(raw_grid, positions, sample.geometry))
    model.train(was_training)
    return running.finish()


def compute_particle_latent_statistics_streaming(
    model: HybridReferenceModel,
    indexes: Sequence[TeacherFrameIndex],
    statistics: TrainingStatistics,
    split: TrajectorySplit,
    *,
    dynamic_grid_cache: Mapping[tuple[str, str, int], torch.Tensor] | None = None,
    wall_grid_cache: Mapping[Path, torch.Tensor] | None = None,
    training_frame_cache: Mapping[tuple[str, str, int], TrainingFrameData]
    | None = None,
    prefetch_frames: int = 0,
    validated_indexes: bool = False,
) -> FeatureStatistics:
    """Measure training latent statistics while retaining one prepared frame."""

    if not indexes:
        raise ContractError("Latent statistics require training frames.")
    running = _RunningStatistics(LATENT_CHANNEL_COUNT)
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    accepted = 0
    with torch.no_grad():
        selected_indexes = [
            index
            for index in indexes
            if split.assignment(index) == "training" and index.valid_grid_support
        ]
        for frame in iter_prepared_training_frames(
            selected_indexes,
            statistics,
            device=parameter.device,
            dtype=parameter.dtype,
            dynamic_grid_cache=dynamic_grid_cache,
            wall_grid_cache=wall_grid_cache,
            training_frame_cache=training_frame_cache,
            prefetch_frames=prefetch_frames,
            validated_indexes=validated_indexes,
        ):
            raw_grid = model.raw_grid_latent(frame.grid_input)
            running.update(
                reference_g2p(
                    raw_grid,
                    frame.positions,
                    frame.geometry,
                    validate=not validated_indexes,
                )
            )
            accepted += 1
    model.train(was_training)
    if accepted == 0:
        raise ContractError("Latent statistics require a valid training frame.")
    return running.finish()


def calibrate_latent_standardization(
    model: HybridReferenceModel, statistics: FeatureStatistics
) -> None:
    """Install exact training latent stats without changing model predictions."""

    if statistics.feature_count != LATENT_CHANNEL_COUNT:
        raise ContractError("Latent statistics must contain 16 channels.")
    device = model.latent_mean.device
    dtype = model.latent_mean.dtype
    new_mean = statistics.mean.to(device=device, dtype=dtype)
    new_std = statistics.std.to(device=device, dtype=dtype)
    with torch.no_grad():
        old_mean = model.latent_mean.clone()
        old_std = model.latent_std.clone()
        latent_weights = model.particle_mlp.layer1.weight[
            :, LOCAL_FEATURE_COUNT:
        ].clone()
        scale = new_std / old_std
        shift = (new_mean - old_mean) / old_std
        model.particle_mlp.layer1.weight[:, LOCAL_FEATURE_COUNT:] = (
            latent_weights * scale[None, :]
        )
        model.particle_mlp.layer1.bias.add_(latent_weights @ shift)
        model.latent_mean.copy_(new_mean)
        model.latent_std.copy_(new_std)


@dataclass(frozen=True)
class TrainingResult:
    epoch_losses: tuple[float, ...]
    latent_statistics: FeatureStatistics
    epoch_seconds: tuple[float, ...]


def train_reference_model(
    model: HybridReferenceModel,
    training_frames: Sequence[PreparedFrame],
    split: TrajectorySplit,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    device: torch.device | str = "cuda",
    epoch_callback: Callable[[int, float, float], None] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> TrainingResult:
    """Jointly train the reference encoder and MLP one complete frame at a time."""

    if not training_frames:
        raise ContractError("Training requires at least one prepared frame.")
    if epochs <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ContractError("Invalid training hyperparameters.")
    for frame in training_frames:
        if split.assignment_key(frame.frame_key[:2]) != "training":
            raise ContractError("Training received a non-training frame.")
    model.to(device=device)
    initial_latent = compute_particle_latent_statistics(model, training_frames, split)
    calibrate_latent_standardization(model, initial_latent)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[float] = []
    epoch_seconds: list[float] = []
    for epoch_index in range(epochs):
        model.train()
        parameter = next(model.parameters())
        if progress_callback is not None:
            progress_callback(epoch_index + 1, 0, len(training_frames))
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        for frame_number, frame in enumerate(training_frames, start=1):
            grid_input = frame.grid_input.to(
                device=parameter.device, dtype=parameter.dtype
            )
            positions = frame.positions.to(
                device=parameter.device, dtype=parameter.dtype
            )
            local = frame.local_features.to(
                device=parameter.device, dtype=parameter.dtype
            )
            target = frame.standardized_target.to(
                device=parameter.device, dtype=parameter.dtype
            )
            valid = frame.valid.to(device=parameter.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(grid_input, positions, local, frame.geometry)
            loss = standardized_target_mse(prediction, target, valid)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            if progress_callback is not None:
                progress_callback(epoch_index + 1, frame_number, len(training_frames))
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)
        elapsed = time.perf_counter() - epoch_started
        epoch_loss = loss_sum / len(training_frames)
        history.append(epoch_loss)
        epoch_seconds.append(elapsed)
        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, epoch_loss, elapsed)
    final_latent = compute_particle_latent_statistics(model, training_frames, split)
    calibrate_latent_standardization(model, final_latent)
    return TrainingResult(tuple(history), final_latent, tuple(epoch_seconds))


def train_reference_model_streaming(
    model: HybridReferenceModel,
    training_indexes: Sequence[TeacherFrameIndex],
    statistics: TrainingStatistics,
    split: TrajectorySplit,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    device: torch.device | str = "cuda",
    seed: int = 1729,
    epoch_callback: Callable[[int, float, float], None] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    dynamic_grid_cache: Mapping[tuple[str, str, int], torch.Tensor] | None = None,
    wall_grid_cache: Mapping[Path, torch.Tensor] | None = None,
    training_frame_cache: Mapping[tuple[str, str, int], TrainingFrameData]
    | None = None,
    prefetch_frames: int = 0,
    validated_indexes: bool = False,
) -> TrainingResult:
    """Train from shuffled frame keys without retaining prepared frames."""

    valid_indexes = [
        index
        for index in training_indexes
        if index.valid_grid_support and split.assignment(index) == "training"
    ]
    if not valid_indexes:
        raise ContractError("Training requires at least one valid indexed frame.")
    if len(valid_indexes) != len(training_indexes):
        for index in training_indexes:
            if split.assignment(index) != "training":
                raise ContractError("Training received a non-training frame.")
    if epochs <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ContractError("Invalid training hyperparameters.")
    model.to(device=device)
    initial_latent = compute_particle_latent_statistics_streaming(
        model,
        valid_indexes,
        statistics,
        split,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=prefetch_frames,
        validated_indexes=validated_indexes,
    )
    calibrate_latent_standardization(model, initial_latent)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    random_generator = random.Random(seed)
    order = sorted(valid_indexes, key=lambda index: index.frame_key)
    history: list[float] = []
    epoch_seconds: list[float] = []
    for epoch_index in range(epochs):
        model.train()
        random_generator.shuffle(order)
        parameter = next(model.parameters())
        if progress_callback is not None:
            progress_callback(epoch_index + 1, 0, len(order))
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)
        epoch_started = time.perf_counter()
        loss_sum = torch.zeros((), dtype=torch.float64, device=parameter.device)
        prepared_frames = iter_prepared_training_frames(
            order,
            statistics,
            device=parameter.device,
            dtype=parameter.dtype,
            dynamic_grid_cache=dynamic_grid_cache,
            wall_grid_cache=wall_grid_cache,
            training_frame_cache=training_frame_cache,
            prefetch_frames=prefetch_frames,
            validated_indexes=validated_indexes,
        )
        for frame_number, frame in enumerate(prepared_frames, start=1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                frame.grid_input,
                frame.positions,
                frame.local_features,
                frame.geometry,
                validate_inputs=not validated_indexes,
            )
            loss = standardized_target_mse(
                prediction,
                frame.standardized_target,
                frame.valid,
                validated=validated_indexes,
            )
            loss.backward()
            optimizer.step()
            loss_sum.add_(loss.detach().to(dtype=torch.float64))
            if progress_callback is not None:
                progress_callback(epoch_index + 1, frame_number, len(order))
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)
        elapsed = time.perf_counter() - epoch_started
        epoch_loss = float((loss_sum / len(order)).cpu())
        history.append(epoch_loss)
        epoch_seconds.append(elapsed)
        if epoch_callback is not None:
            epoch_callback(epoch_index + 1, epoch_loss, elapsed)
    final_latent = compute_particle_latent_statistics_streaming(
        model,
        valid_indexes,
        statistics,
        split,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=prefetch_frames,
        validated_indexes=validated_indexes,
    )
    calibrate_latent_standardization(model, final_latent)
    return TrainingResult(tuple(history), final_latent, tuple(epoch_seconds))


class _ExportGridEncoder(torch.nn.Module):
    def __init__(self, model: HybridReferenceModel) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(model.grid_encoder)
        self.register_buffer("mean", model.latent_mean.detach().clone())
        self.register_buffer("std", model.latent_std.detach().clone())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raw = self.encoder(value)
        return (raw - self.mean.view(1, -1, 1, 1, 1)) / self.std.view(1, -1, 1, 1, 1)


class _ExportParticleMLP(torch.nn.Module):
    def __init__(
        self, model: HybridReferenceModel, target_statistics: FeatureStatistics
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(model.particle_mlp)
        self.register_buffer("target_mean", target_statistics.mean.to(torch.float32))
        self.register_buffer("target_std", target_statistics.std.to(torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value) * self.target_std + self.target_mean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torchscript_save(module: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.jit.script(module.eval()).save(str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_onnx_export(
    module: torch.nn.Module,
    example: torch.Tensor,
    path: Path,
    *,
    input_name: str,
    output_name: str,
    opset_version: int,
    dynamic_batch: bool,
) -> None:
    """Export an ONNX graph through the stable Torch exporter atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        dynamic_axes = (
            {input_name: {0: "batch"}, output_name: {0: "batch"}}
            if dynamic_batch
            else None
        )
        # The legacy exporter is intentional here: it is available with the
        # pinned PyTorch environment even when onnxscript is not installed.
        torch.onnx.export(
            module.eval(),
            (example,),
            str(temporary),
            opset_version=opset_version,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_onnx_cuda_consistency(
    module: torch.nn.Module,
    path: str | Path,
    example: torch.Tensor,
    *,
    input_name: str,
    rtol: float = ONNX_CONSISTENCY_RTOL,
    atol: float = ONNX_CONSISTENCY_ATOL,
) -> dict[str, object]:
    """Run one ONNX Runtime session and require CUDA as its first provider."""

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ContractError(
            "ONNX Runtime GPU is required for stage 4 export."
        ) from error
    available = tuple(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise ContractError(
            "ONNX Runtime CUDAExecutionProvider is unavailable; refusing CPU fallback."
        )
    session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    providers = tuple(session.get_providers())
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise ContractError(
            "ONNX Runtime did not select CUDAExecutionProvider as the active provider."
        )
    cuda_module = copy.deepcopy(module).to(device="cuda").eval()
    cuda_example = example.to(device="cuda")
    with torch.no_grad():
        expected = cuda_module(cuda_example).detach().cpu().numpy()
    actual = session.run(None, {input_name: example.detach().cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    difference = np.abs(actual - expected)
    max_absolute = float(difference.max()) if difference.size else 0.0
    relative = difference / np.maximum(np.abs(expected), atol)
    max_relative = float(relative.max()) if relative.size else 0.0
    return {
        "provider": providers[0],
        "availableProviders": list(available),
        "maxAbsoluteError": max_absolute,
        "maxRelativeError": max_relative,
        "rtol": rtol,
        "atol": atol,
    }


def _atomic_json_save(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _native_particle_parameters(
    model: HybridReferenceModel, target_statistics: FeatureStatistics
) -> list[float]:
    particle = model.particle_mlp
    target_mean = target_statistics.mean.to(dtype=torch.float32)
    target_std = target_statistics.std.to(dtype=torch.float32)
    tensors = (
        particle.layer1.weight,
        particle.layer1.bias,
        particle.layer2.weight,
        particle.layer2.bias,
        particle.layer3.weight,
        particle.layer3.bias,
        particle.output.weight * target_std[:, None],
        particle.output.bias * target_std + target_mean,
    )
    return torch.cat(
        [
            value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            for value in tensors
        ]
    ).tolist()


def validate_exported_model_profile(
    bundle: str | Path, *, expected_profile: str | None = None
) -> dict[str, object]:
    """Cross-check profile metadata against exported trainable parameter tensors."""

    bundle_path = Path(bundle)
    metadata_path = bundle_path / "model-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("Model bundle metadata is not valid ASCII JSON.") from error
    grid_metadata = metadata.get("architecture", {}).get("gridEncoder", {})
    profile_name = grid_metadata.get("profile")
    if profile_name not in MODEL_PROFILES:
        raise ContractError(f"Unknown model bundle profile: {profile_name}.")
    if expected_profile is not None and profile_name != expected_profile:
        raise ContractError(
            f"Model bundle profile {profile_name} does not match {expected_profile}."
        )
    condition_names = metadata.get("channels", {}).get("conditions", [])
    if not isinstance(condition_names, list):
        raise ContractError("Model bundle condition channel metadata is invalid.")
    expected_model = HybridReferenceModel(len(condition_names), profile_name)
    expected_architecture = model_architecture(expected_model)
    profile = MODEL_PROFILES[profile_name]
    if grid_metadata.get("encoderWidths") != list(profile["encoderWidths"]):
        raise ContractError("Model profile and encoder widths disagree.")
    if grid_metadata.get("decoderWidths") != list(profile["decoderWidths"]):
        raise ContractError("Model profile and decoder widths disagree.")
    if grid_metadata.get("inputChannels") != BASE_GRID_CHANNEL_COUNT + len(
        condition_names
    ):
        raise ContractError("Model profile and grid input width disagree.")
    if grid_metadata.get("outputChannels") != LATENT_CHANNEL_COUNT:
        raise ContractError("Model profile and latent width disagree.")
    if (
        grid_metadata.get("parameterCount")
        != expected_architecture["gridParameterCount"]
    ):
        raise ContractError("Model profile and grid parameter count disagree.")
    grid_script_path = bundle_path / "grid-encoder.pt"
    particle_script_path = bundle_path / "particle-mlp.pt"
    if not grid_script_path.is_file() or not particle_script_path.is_file():
        raise ContractError(
            "Training bundle profile validation requires TorchScript compatibility artifacts."
        )
    grid_script = torch.jit.load(str(grid_script_path), map_location="cpu")
    particle_script = torch.jit.load(str(particle_script_path), map_location="cpu")
    actual_grid_count = sum(value.numel() for value in grid_script.parameters())
    actual_particle_count = sum(value.numel() for value in particle_script.parameters())
    if actual_grid_count != expected_architecture["gridParameterCount"]:
        raise ContractError("Exported grid layers do not match the declared profile.")
    if actual_particle_count != expected_architecture["particleMlpParameterCount"]:
        raise ContractError("Exported Particle MLP layers do not match its ABI.")
    native = json.loads(
        (bundle_path / "particle-mlp-native.json").read_text(encoding="ascii")
    )
    if (
        native.get("architecture") != list(PARTICLE_MLP_WIDTHS)
        or len(native.get("parameters", []))
        != expected_architecture["particleMlpParameterCount"]
    ):
        raise ContractError("Native Particle MLP artifact does not match its ABI.")
    return {
        **expected_architecture,
        "actualGridParameterCount": actual_grid_count,
        "actualParticleMlpParameterCount": actual_particle_count,
    }


def export_reference_artifacts(
    output_directory: str | Path,
    model: HybridReferenceModel,
    statistics: TrainingStatistics,
    frames: Sequence[TeacherFrame | TeacherFrameIndex],
    split: TrajectorySplit,
    *,
    collision_post_process: str = "full",
    onnx_opset: int = 18,
    validate_cuda: bool = True,
) -> dict[str, object]:
    """Export separate TorchScript modules plus shared frozen-contract metadata."""

    if statistics.latent is None:
        raise ContractError("Export requires final training-split latent statistics.")
    if collision_post_process not in ("full", "guard", "none"):
        raise ContractError("Invalid collision post-process policy.")
    if not frames:
        raise ContractError("Export metadata requires dataset frames.")
    frame_keys = [frame.frame_key for frame in frames]
    if len(set(frame_keys)) != len(frame_keys):
        raise ContractError("Export metadata contains duplicate frame keys.")
    geometry = frames[0].geometry
    condition_names = frames[0].condition_names
    ai_delta_time = frames[0].ai_delta_time
    particle_diameter = frames[0].particle_diameter
    certification_profiles = {frame.certification_profile for frame in frames}
    if len(certification_profiles) != 1:
        raise ContractError(
            "Schema 3 cannot represent a bundle with mixed certification profiles."
        )
    certification_profile = next(iter(certification_profiles))
    for frame in frames:
        split.assignment(frame)
        if (
            frame.geometry != geometry
            or frame.condition_names != condition_names
            or frame.ai_delta_time != ai_delta_time
            or frame.particle_diameter != particle_diameter
        ):
            raise ContractError("Export frames do not share one model contract.")
    particle_radius: float | None = None
    with _TeacherFrameReader() as reader:
        for frame_reference in frames:
            radii = (
                frame_reference.radius_start
                if isinstance(frame_reference, TeacherFrame)
                else reader.load_radii(frame_reference)
            )
            if radii.size == 0 or not np.isfinite(radii).all() or np.any(radii <= 0.0):
                raise ContractError(
                    "Teacher particle radii must be finite and positive."
                )
            if not np.allclose(radii, radii[0], rtol=1.0e-6, atol=0.0):
                raise ContractError(
                    "The first baseline requires one particle resolution."
                )
            if particle_radius is None:
                particle_radius = float(radii[0])
            elif not math.isclose(
                float(radii[0]), particle_radius, rel_tol=1.0e-6, abs_tol=0.0
            ):
                raise ContractError(
                    "The first baseline requires one particle resolution."
                )
            if not math.isclose(
                2.0 * float(radii[0]),
                particle_diameter,
                rel_tol=1.0e-6,
                abs_tol=0.0,
            ):
                raise ContractError(
                    "Teacher radius and particleDiameter metadata disagree."
                )
    if particle_radius is None:
        raise ContractError("Export metadata requires particle radius data.")

    output_directory = Path(output_directory)
    grid_path = output_directory / "grid-encoder.pt"
    particle_path = output_directory / "particle-mlp.pt"
    grid_onnx_path = output_directory / "grid-encoder.onnx"
    particle_onnx_path = output_directory / "particle-mlp.onnx"
    particle_native_path = output_directory / "particle-mlp-native.json"
    metadata_path = output_directory / "model-metadata.json"
    export_model = copy.deepcopy(model).to(device="cpu", dtype=torch.float32).eval()
    calibrate_latent_standardization(export_model, statistics.latent)
    export_grid = _ExportGridEncoder(export_model)
    export_particle = _ExportParticleMLP(export_model, statistics.target)
    _atomic_torchscript_save(export_grid, grid_path)
    _atomic_torchscript_save(export_particle, particle_path)
    grid_example = torch.zeros(
        (
            1,
            BASE_GRID_CHANNEL_COUNT + len(condition_names),
            *geometry.tensor_shape,
        ),
        dtype=torch.float32,
    )
    particle_example = torch.zeros((1, PARTICLE_MLP_WIDTHS[0]), dtype=torch.float32)
    _atomic_onnx_export(
        export_grid,
        grid_example,
        grid_onnx_path,
        input_name="gridInput",
        output_name="gridLatent",
        opset_version=onnx_opset,
        dynamic_batch=False,
    )
    _atomic_onnx_export(
        export_particle,
        particle_example,
        particle_onnx_path,
        input_name="particleInput",
        output_name="residualAcceleration",
        opset_version=onnx_opset,
        dynamic_batch=True,
    )
    _atomic_json_save(
        {
            "schemaVersion": NATIVE_PARTICLE_MLP_SCHEMA_VERSION,
            "architecture": list(PARTICLE_MLP_WIDTHS),
            "layout": "rowMajorWeightsThenBiasPerLayer",
            "output": "physicalResidualAcceleration3",
            "parameters": _native_particle_parameters(export_model, statistics.target),
        },
        particle_native_path,
    )
    if validate_cuda:
        grid_consistency = validate_onnx_cuda_consistency(
            export_grid,
            grid_onnx_path,
            grid_example,
            input_name="gridInput",
        )
        particle_consistency = validate_onnx_cuda_consistency(
            export_particle,
            particle_onnx_path,
            particle_example,
            input_name="particleInput",
        )
    else:
        grid_consistency = {"status": "notRun", "reason": "validate_cuda=false"}
        particle_consistency = {"status": "notRun", "reason": "validate_cuda=false"}
    architecture = model_architecture(export_model)
    metadata: dict[str, object] = {
        "contractName": CONTRACT["modelBundle"]["contractName"],
        "contractVersion": CONTRACT["contractPackage"]["version"],
        "contractRegistrySha256": REGISTRY_SHA256,
        "schemaVersion": MODEL_BUNDLE_SCHEMA_VERSION,
        "certificationProfile": certification_profile,
        "modelFamily": CONTRACT["modelBundle"]["modelFamily"],
        "aiDeltaTime": ai_delta_time,
        "particleDiameter": particle_diameter,
        "particleVolumeDefinition": CONTRACT["teacher"]["particleVolumeDefinition"],
        "dtype": CONTRACT["modelBundle"]["dtype"],
        "units": dict(CONTRACT["units"]),
        "grid": geometry.to_metadata(),
        "gridInterpolation": {
            "name": CONTRACT["grid"]["interpolation"],
            "stencilCellCount": CONTRACT["grid"]["stencilCellCount"],
            "clampOutOfBounds": CONTRACT["grid"]["clampOutOfBounds"],
        },
        "wall": {
            "geometryFormatVersion": CONTRACT["wall"]["geometryFormatVersion"],
            "quaternionOrder": CONTRACT["wall"]["quaternionOrder"],
            "quaternionConvention": CONTRACT["wall"]["quaternionConvention"],
            "coordinateTransform": CONTRACT["wall"]["coordinateTransform"],
            "timeInterpolation": CONTRACT["wall"]["timeInterpolation"],
            "velocityDefinition": CONTRACT["wall"]["velocityDefinition"],
            "rasterizationAlgorithm": RASTERIZATION_ALGORITHM_VERSION,
        },
        "channels": {
            "gridBase": list(BASE_GRID_CHANNEL_NAMES),
            "conditions": list(condition_names),
            "local": list(LOCAL_FEATURE_NAMES),
            "latent": list(LATENT_CHANNEL_NAMES),
            "target": list(TARGET_CHANNEL_NAMES),
        },
        "architecture": {
            "gridEncoder": {
                "profile": model.model_profile,
                "inputChannels": BASE_GRID_CHANNEL_COUNT + len(condition_names),
                "encoderWidths": architecture["gridEncoderWidths"],
                "decoderWidths": architecture["decoderWidths"],
                "outputChannels": LATENT_CHANNEL_COUNT,
                "parameterCount": architecture["gridParameterCount"],
            },
            "particleMlp": list(PARTICLE_MLP_WIDTHS),
        },
        "normalization": {
            "sourceSplit": "training",
            "dynamicQuantities": statistics.dynamic.to_metadata(DYNAMIC_QUANTITY_NAMES),
            "conditions": (
                statistics.condition.to_metadata(condition_names)
                if statistics.condition is not None
                else None
            ),
            "target": statistics.target.to_metadata(TARGET_CHANNEL_NAMES),
            "particleG2pLatent": statistics.latent.to_metadata(LATENT_CHANNEL_NAMES),
        },
        "target": {
            "definition": CONTRACT["teacher"]["targetDefinition"],
            "targetIncludesTeacherCollision": CONTRACT["teacher"][
                "targetIncludesTeacherCollision"
            ],
        },
        "previousMacroResidualHistory": {
            "definition": CONTRACT["teacher"]["previousMacroResidualHistoryDefinition"],
            "initialValue": CONTRACT["teacher"]["previousMacroResidualInitialValue"],
        },
        "artifacts": {
            "gridEncoder": {
                "name": grid_onnx_path.name,
                "format": "onnx",
                "sha256": _sha256(grid_onnx_path),
            },
            "particleMlp": {
                "name": particle_onnx_path.name,
                "format": "onnx",
                "sha256": _sha256(particle_onnx_path),
            },
            "particleMlpNative": {
                "name": particle_native_path.name,
                "format": "json-float32-row-major-v1",
                "sha256": _sha256(particle_native_path),
            },
        },
        "runtime": {
            "onnxOpset": onnx_opset,
            "expectedOnnxRuntimeProvider": CONTRACT["modelBundle"][
                "expectedOnnxRuntimeProvider"
            ],
        },
    }
    _atomic_json_save(metadata, metadata_path)
    validate_model_bundle_v3(output_directory)
    profile_validation = validate_exported_model_profile(
        output_directory, expected_profile=model.model_profile
    )
    _atomic_json_save(
        {
            "modelProfile": model.model_profile,
            "collisionPostProcess": collision_post_process,
            "pytorchVersion": torch.__version__,
            "torchscript": {
                "gridEncoderSha256": _sha256(grid_path),
                "particleMlpSha256": _sha256(particle_path),
            },
            "onnxConsistencyTolerance": {
                "rtol": ONNX_CONSISTENCY_RTOL,
                "atol": ONNX_CONSISTENCY_ATOL,
            },
            "gridEncoderConsistency": grid_consistency,
            "particleMlpConsistency": particle_consistency,
            "profileValidation": profile_validation,
            "contractValidator": "passed",
        },
        output_directory / "export-validation.json",
    )
    return metadata
