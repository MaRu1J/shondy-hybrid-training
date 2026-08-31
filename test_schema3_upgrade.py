from __future__ import annotations

import io
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from shondy_hybrid_contract import ContractViolation, validate_teacher_hdf5_v3
from shondy_hybrid_contract import (
    rasterize_wall_grid as contract_rasterize_wall_grid,
)

import hybrid_reference_model as hybrid
import train_hybrid_model as trainer
from test_hybrid_reference_model import write_teacher_file
from wall_rasterizer import (
    RASTERIZATION_ALGORITHM_VERSION,
    WALL_GRID_CHANNEL_NAMES,
    load_wall_geometry,
    rasterize_wall_grid,
    resolve_wall_states,
)

CONTRACT_ROOT = Path("/home/ruijin/project/shondy-hybrid-contract")
CONTRACT_FIXTURES = CONTRACT_ROOT / "fixtures-v3"
REAL_TEACHER = Path(
    "/home/ruijin/project/shondy-teacher-writer/applicationTests/generated/"
    "03_damBreak3D-schema3-ai1e3-0400-0900/"
    "ai-teacher-frames-schema3-ai1e3-0400-0900.h5"
)


def _add_second_frame(path: Path, delta: float = 1.0e-3) -> None:
    with h5py.File(path, "r+") as file:
        frames = file["frames"]
        file.copy(frames["000000000000"], frames, name="000000000001")
        frame = frames["000000000001"]
        frame.attrs["macroStepIndex"] = np.uint64(1)
        frame.attrs["timeStart"] = delta
        frame.attrs["timeEnd"] = 2.0 * delta
        frame.attrs["aiDeltaTime"] = delta
        angle = 0.5 * delta
        frame["wallState/pose"][0] = np.asarray(
            [
                0.0,
                0.0,
                0.25 * delta,
                math.cos(angle / 2.0),
                0.0,
                0.0,
                math.sin(angle / 2.0),
            ]
        )
        frame["wallState/linearVelocity"][0] = [0.0, 0.0, 0.25]
        frame["wallState/angularVelocity"][0] = [0.0, 0.0, 0.5]


@pytest.mark.parametrize("delta", [1.0e-4, 1.0e-3, 0.037])
def test_positive_artifact_owned_ai_delta_time(tmp_path: Path, delta: float) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path, ai_delta_time=delta)
    index = hybrid.index_teacher_trajectory(path)[0]
    assert index.ai_delta_time == delta
    assert index.time_end == delta


@pytest.mark.parametrize("delta", [0.0, -1.0e-3, math.nan, math.inf])
def test_invalid_root_ai_delta_time_is_rejected(tmp_path: Path, delta: float) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file.attrs["aiDeltaTime"] = delta
    with pytest.raises(hybrid.ContractError, match="aiDeltaTime"):
        hybrid.index_teacher_trajectory(path)


def test_frame_root_ai_delta_time_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["frames/000000000000"].attrs["aiDeltaTime"] = 1.0e-3
    with pytest.raises(hybrid.ContractError, match="aiDeltaTime"):
        hybrid.index_teacher_trajectory(path)


@pytest.mark.parametrize(
    ("name", "value"),
    [("timeStart", 0.5), ("timeEnd", 0.5), ("macroStepIndex", 2)],
)
def test_non_integer_macro_tick_relation_is_rejected(
    tmp_path: Path, name: str, value: float
) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["frames/000000000000"].attrs[name] = value
    with pytest.raises(hybrid.ContractError, match="Frame name|duration|macro ticks"):
        hybrid.index_teacher_trajectory(path)


def test_multiple_trajectory_delta_mismatch_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    write_teacher_file(first, trajectory_id="first", ai_delta_time=1.0e-4)
    write_teacher_file(second, trajectory_id="second", ai_delta_time=1.0e-3)
    with pytest.raises(hybrid.ContractError, match="share aiDeltaTime"):
        trainer.load_dataset((first, second))


def test_model_teacher_delta_mismatch_is_rejected_by_contract(tmp_path: Path) -> None:
    teacher = CONTRACT_FIXTURES / "teacher-v3-fixed-wall.h5"
    metadata = json.loads(
        (
            CONTRACT_FIXTURES / "model-bundle-v3-fixed-wall/model-metadata.json"
        ).read_text(encoding="ascii")
    )
    metadata["aiDeltaTime"] = 1.0e-3
    with pytest.raises(ContractViolation, match="aiDeltaTime"):
        validate_teacher_hdf5_v3(teacher, metadata)


@pytest.mark.parametrize(
    ("fixture", "time"),
    [
        ("teacher-v3-fixed-wall.h5", 0.0),
        ("teacher-v3-prescribed-wall.h5", 1.0e-3),
        ("teacher-v3-sampled-wall.h5", 5.0e-4),
    ],
)
def test_local_wall_rasterizer_matches_contract_reference(
    fixture: str, time: float
) -> None:
    path = CONTRACT_FIXTURES / fixture
    with h5py.File(path, "r") as file:
        actual = rasterize_wall_grid(file, load_wall_geometry(file), time)
    expected = contract_rasterize_wall_grid(path, time)
    assert WALL_GRID_CHANNEL_NAMES == tuple(hybrid.CONTRACT["channels"]["wallGrid"])
    assert RASTERIZATION_ALGORITHM_VERSION == "triangleNearestCellCenter-v1"
    assert actual.shape == expected.shape == (8, 8, 8, 8)
    assert actual.dtype == expected.dtype == np.float32
    difference = np.abs(actual - expected)
    assert float(difference.max()) <= 1.0e-6
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_sampled_state_endpoints_and_midpoint_slerp(tmp_path: Path) -> None:
    path = tmp_path / "sampled.h5"
    delta = 1.0e-3
    write_teacher_file(path, ai_delta_time=delta, motion_mode="sampled-state")
    _add_second_frame(path, delta)
    with h5py.File(path, "r") as file:
        geometry = load_wall_geometry(file)
        start = resolve_wall_states(file, geometry, 0.0)[0]
        midpoint = resolve_wall_states(file, geometry, 0.5 * delta)[0]
        end = resolve_wall_states(file, geometry, delta)[0]
    np.testing.assert_allclose(start.translation, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(midpoint.translation, [0.0, 0.0, 0.125 * delta])
    np.testing.assert_allclose(end.translation, [0.0, 0.0, 0.25 * delta])
    assert midpoint.rotation_wxyz[0] < 1.0
    np.testing.assert_allclose(midpoint.linear_velocity, [0.0, 0.0, 0.125])


def test_prescribed_reference_pose_origin_and_velocities(tmp_path: Path) -> None:
    path = tmp_path / "prescribed.h5"
    write_teacher_file(path, motion_mode="prescribed-law")
    with h5py.File(path, "r") as file:
        geometry = load_wall_geometry(file)
        state = resolve_wall_states(file, geometry, 2.0)[0]
    np.testing.assert_allclose(state.translation, [0.0, 0.0, 0.5])
    np.testing.assert_allclose(state.linear_velocity, [0.0, 0.0, 0.25])
    np.testing.assert_allclose(state.angular_velocity, [0.0, 0.0, 0.5])


def test_multi_body_uuid_order_tie_break_and_mixed_motion(tmp_path: Path) -> None:
    path = tmp_path / "multi.h5"
    write_teacher_file(path, motion_mode="sampled-state")
    with h5py.File(path, "r+") as file:
        wall = file["wallGeometry"]
        wall.attrs["bodyUuidsJson"] = json.dumps(["uuid-first", "uuid-second"])
        bodies = wall["bodies"]
        file.copy(bodies["body-000000"], bodies, name="body-000001")
        bodies["body-000001"].attrs["motionMode"] = "static"
        state = file["frames/000000000000/wallState"]
        pose = np.repeat(np.asarray(state["pose"]), 2, axis=0)
        del state["pose"]
        state.create_dataset("pose", data=pose)
        del state["linearVelocity"]
        state.create_dataset(
            "linearVelocity", data=np.asarray([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        )
        angular = np.zeros((2, 3))
        del state["angularVelocity"]
        state.create_dataset("angularVelocity", data=angular)
        state.attrs["bodyCount"] = 2
    indexes = hybrid.index_teacher_trajectory(path)
    assert indexes[0].wall_is_static is False
    with h5py.File(path, "r") as file:
        geometry = load_wall_geometry(file)
        assert geometry.body_uuids == ("uuid-first", "uuid-second")
        assert geometry.motion_modes == ("sampled-state", "static")
        grid = rasterize_wall_grid(file, geometry, 0.0)
    velocities = grid[4:7, grid[0].astype(bool)]
    assert np.all(velocities[0] == 1.0)
    assert np.all(velocities[1:] == 0.0)


def test_coupled_state_uses_sampled_state_semantics(tmp_path: Path) -> None:
    path = tmp_path / "coupled.h5"
    write_teacher_file(path, motion_mode="coupled-state")
    _add_second_frame(path, 1.0e-3)
    with h5py.File(path, "r") as file:
        geometry = load_wall_geometry(file)
        state = resolve_wall_states(file, geometry, 5.0e-4)[0]
    np.testing.assert_allclose(state.translation, [0.0, 0.0, 0.125e-3])


def test_duplicate_body_uuid_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["wallGeometry"].attrs["bodyUuidsJson"] = json.dumps(["same", "same"])
    with pytest.raises(hybrid.ContractError, match="unique ordered"):
        hybrid.index_teacher_trajectory(path)


def test_invalid_motion_mode_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["wallGeometry/bodies/body-000000"].attrs["motionMode"] = "teleport"
    with pytest.raises(hybrid.ContractError, match="motion mode"):
        hybrid.index_teacher_trajectory(path)


@pytest.mark.parametrize("quaternion", [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
def test_invalid_or_non_unit_quaternion_is_rejected(
    tmp_path: Path, quaternion: list[float]
) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["wallGeometry/bodies/body-000000/referencePose"][3:] = quaternion
    with pytest.raises(hybrid.ContractError, match="quaternion"):
        hybrid.index_teacher_trajectory(path)


def test_out_of_range_triangle_connectivity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["wallGeometry/bodies/body-000000/triangleConnectivity"][0, 0] = 999
    with pytest.raises(hybrid.ContractError, match="[Cc]onnectivity"):
        hybrid.index_teacher_trajectory(path)


def test_repeated_frame_wall_topology_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        file["frames/000000000000"].create_group("wallGeometry")
    with pytest.raises(hybrid.ContractError, match="repeats root wall topology"):
        hybrid.index_teacher_trajectory(path)


def test_optional_dense_wall_cache_is_parity_checked_not_used_as_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "teacher.h5"
    write_teacher_file(path)
    with h5py.File(path, "r+") as file:
        expected = rasterize_wall_grid(file, load_wall_geometry(file), 0.0)
        file["frames/000000000000/grid"].create_dataset("wallStart", data=expected)
    frame = hybrid.load_teacher_trajectory(path)[0]
    np.testing.assert_array_equal(frame.wall_channels_start, expected)
    with h5py.File(path, "r+") as file:
        file["frames/000000000000/grid/wallStart"][0, 0, 0, 0] = 1.0
    with pytest.raises(hybrid.ContractError, match="dense wallStart cache"):
        hybrid.load_teacher_trajectory(path)


def test_profiles_have_expected_widths_and_actual_parameter_counts() -> None:
    compact = hybrid.HybridReferenceModel(0, "compact-v1")
    large = hybrid.HybridReferenceModel(0, "large-v1")
    compact_architecture = hybrid.model_architecture(compact)
    large_architecture = hybrid.model_architecture(large)
    assert compact_architecture["decoderWidths"] == [64, 48, 32]
    assert compact_architecture["totalParameterCount"] == 2_101_907
    assert large_architecture["gridEncoderWidths"] == [64, 96, 128, 192]
    assert large_architecture["decoderWidths"] == [128, 96, 64]
    assert large_architecture["totalParameterCount"] == 8_280_083


def test_checkpoint_profile_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    compact = hybrid.HybridReferenceModel(0, "compact-v1")
    trainer._write_checkpoint(checkpoint, compact, epoch=1, training_loss=1.0)
    large = hybrid.HybridReferenceModel(0, "large-v1")
    with pytest.raises(hybrid.ContractError, match="profile"):
        trainer._load_checkpoint(checkpoint, large)


def test_epoch_progress_reports_tty_bar_and_structured_log() -> None:
    structured = io.StringIO()
    reporter = trainer.EpochProgressReporter(2, 1.0, stream=structured)
    reporter(1, 0, 10)
    reporter(1, 10, 10)
    entries = [json.loads(line) for line in structured.getvalue().splitlines()]
    assert [entry["progressFrames"] for entry in entries] == [0, 10]
    assert entries[-1]["progressFraction"] == 1.0

    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    tty = TtyBuffer()
    tty_reporter = trainer.EpochProgressReporter(2, 1.0, stream=tty)
    tty_reporter(1, 5, 10)
    assert "Epoch 1/2 [###############...............]" in tty.getvalue()
    assert "5/10" in tty.getvalue()


@pytest.mark.skipif(
    not REAL_TEACHER.is_file(), reason="Writer integration artifact absent"
)
def test_real_schema3_500_macro_steps_wall_and_training_pipeline() -> None:
    indexes = hybrid.index_teacher_trajectory(REAL_TEACHER)
    assert len(indexes) == 500
    assert [indexes[0].macro_step_index, indexes[-1].macro_step_index] == [400, 899]
    assert indexes[0].ai_delta_time == 1.0e-3
    assert indexes[0].time_start == 0.4
    assert indexes[-1].time_end == 0.9
    frame = hybrid.load_teacher_frame(indexes[0])
    assert frame.wall_channels_start.shape == (8, 26, 30, 65)
    assert frame.wall_channels_start.dtype == np.float32
    assert frame.wall_channels_start[0].sum() > 0
    statistics = hybrid.TrainingStatistics(
        dynamic=hybrid.FeatureStatistics(
            torch.zeros(7), torch.ones(7), 1, torch.zeros(7, dtype=torch.bool)
        ),
        condition=None,
        target=hybrid.FeatureStatistics(
            torch.zeros(3), torch.ones(3), 1, torch.zeros(3, dtype=torch.bool)
        ),
    )
    prepared = hybrid.prepare_frame(
        hybrid.HybridFrameSample(frame, torch.as_tensor(frame.wall_channels_start)),
        statistics,
    )
    assert prepared.grid_input.shape == (1, 19, 26, 30, 65)
    assert prepared.positions.shape == (108540, 3)
