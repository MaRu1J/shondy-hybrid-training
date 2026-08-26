import copy
import dataclasses
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import onnx
import onnxruntime as ort
import torch

import hybrid_reference_model as hybrid
import train_hybrid_model as hybrid_trainer


def make_geometry() -> hybrid.GridGeometry:
    return hybrid.GridGeometry(
        physical_bounds_min=(0.0, 0.0, 0.0),
        physical_bounds_max=(2.0, 2.0, 2.0),
        padded_bounds_min=(-3.0, -3.0, -3.0),
        padded_bounds_max=(5.0, 5.0, 5.0),
        cell_counts=(8, 8, 8),
        cell_size=1.0,
    )


def make_statistics(
    mean: list[float], std: list[float], count: int = 4
) -> hybrid.FeatureStatistics:
    return hybrid.FeatureStatistics(
        mean=torch.tensor(mean, dtype=torch.float64),
        std=torch.tensor(std, dtype=torch.float64),
        count=count,
        constant_mask=torch.zeros(len(mean), dtype=torch.bool),
    )


def make_wall_channels(
    geometry: hybrid.GridGeometry, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    result = torch.zeros(
        hybrid.WALL_GRID_CHANNEL_COUNT, *geometry.tensor_shape, dtype=dtype
    )
    result[7] = geometry.valid_domain_mask(dtype=dtype, device="cpu")[0]
    result[0, 3, 3, 3] = 1.0
    result[1, 3, 3, 3] = 1.0
    result[4, 3, 3, 3] = 2.0
    return result


def write_teacher_file(
    path: Path,
    *,
    trajectory_id: str = "trajectory-0",
    duplicate_static_id: bool = False,
) -> None:
    geometry = make_geometry()
    positions = np.array([[0.5, 0.5, 0.5], [1.25, 1.5, 1.75]], dtype=np.float64)
    velocity = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    local = np.zeros((2, hybrid.LOCAL_FEATURE_COUNT), dtype=np.float64)
    local[:, 0:3] = velocity
    local[:, 3] = (1.0, 0.8)
    local[:, 8:11] = ((0.1, 0.2, 0.3), (0.0, 0.0, 0.0))
    static_id = np.array([10, 10 if duplicate_static_id else 20], dtype=np.int32)
    valid = np.array([1, 0], dtype=np.uint8)
    position_end = positions.copy()
    velocity_end = velocity.copy()
    target = np.array([[4.0, 5.0, 6.0], [np.nan, np.nan, np.nan]])
    position_end[1] = np.nan
    velocity_end[1] = np.nan

    with h5py.File(path, "w") as file:
        text_type = h5py.string_dtype(encoding="utf-8")
        teacher_contract = hybrid.CONTRACT["teacher"]
        grid_contract = hybrid.CONTRACT["grid"]
        file.attrs["contractName"] = np.array(
            [teacher_contract["contractName"]], dtype=text_type
        )
        file.attrs["contractVersion"] = np.array(
            [hybrid.CONTRACT["contractPackage"]["version"]], dtype=text_type
        )
        file.attrs["contractRegistrySha256"] = np.array(
            [hybrid.REGISTRY_SHA256], dtype=text_type
        )
        file.attrs["schemaVersion"] = np.array([hybrid.SCHEMA_VERSION])
        file.attrs["certificationProfile"] = np.array(
            [hybrid.CONTRACT["certificationProfiles"]["extendedUnverified"]],
            dtype=text_type,
        )
        file.attrs["caseId"] = np.array(["synthetic-case"], dtype=text_type)
        file.attrs["trajectoryId"] = np.array([trajectory_id], dtype=text_type)
        file.attrs["tensorLayout"] = np.array([hybrid.TENSOR_LAYOUT], dtype=text_type)
        file.attrs["aiDeltaTime"] = np.array([1.0e-4])
        file.attrs["particleDiameter"] = np.array([1.0])
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
            "teacherFrameStrideSemantics": teacher_contract[
                "teacherFrameStrideSemantics"
            ],
            "gridInterpolation": grid_contract["interpolation"],
        }
        for name, value in exact_text_attributes.items():
            file.attrs[name] = np.array([value], dtype=text_type)
        file.attrs["targetIncludesTeacherCollision"] = np.array([1])
        file.attrs["targetMacroStepSpan"] = np.array(
            [teacher_contract["targetMacroStepSpan"]]
        )
        file.attrs["gridStencilCellCount"] = np.array(
            [grid_contract["stencilCellCount"]]
        )
        file.attrs["gridClampOutOfBounds"] = np.array(
            [int(grid_contract["clampOutOfBounds"])]
        )
        file.attrs["teacherFrameStride"] = np.array([1])
        file.attrs["unitsJson"] = np.array(
            [json.dumps(dict(hybrid.CONTRACT["units"]))], dtype=text_type
        )
        file.attrs["conditionsJson"] = np.array(
            [json.dumps({"omegaStart": 1.0, "omegaEnd": 2.0})], dtype=text_type
        )
        file.attrs["conditionNamesJson"] = np.array(
            [json.dumps(["omegaStart", "omegaEnd"])], dtype=text_type
        )
        for attribute, registry_name in (
            ("localFeatureNamesJson", "local"),
            ("dynamicQuantityNamesJson", "dynamicQuantities"),
            ("gridBaseChannelNamesJson", "gridBase"),
            ("wallGridChannelNamesJson", "wallGrid"),
            ("targetChannelNamesJson", "target"),
        ):
            file.attrs[attribute] = np.array(
                [json.dumps(list(hybrid.CONTRACT["channels"][registry_name]))],
                dtype=text_type,
            )
        file.attrs["fixedWallGridDeduplicated"] = np.array([0])
        grid = file.create_group("gridGeometry")
        grid.attrs["cellCentered"] = np.array([1])
        grid.attrs["paddingCells"] = np.array([geometry.padding_cells])
        grid.attrs["cellSize"] = np.array([geometry.cell_size])
        grid.create_dataset("physicalBoundsMin", data=geometry.physical_bounds_min)
        grid.create_dataset("physicalBoundsMax", data=geometry.physical_bounds_max)
        grid.create_dataset("paddedBoundsMin", data=geometry.padded_bounds_min)
        grid.create_dataset("paddedBoundsMax", data=geometry.padded_bounds_max)
        grid.create_dataset("cellCounts", data=geometry.cell_counts)
        frames = file.create_group("frames")
        frame = frames.create_group("000000000000")
        frame.attrs["macroStepIndex"] = np.uint64(0)
        frame.attrs["timeStart"] = np.array([0.0])
        frame.attrs["timeEnd"] = np.array([1.0e-4])
        frame.attrs["aiDeltaTime"] = np.array([1.0e-4])
        frame.attrs["substepCount"] = np.array([2])
        frame.attrs["validGridSupport"] = np.array([1])
        particle = frame.create_group("particle")
        particle.create_dataset("staticId", data=static_id)
        particle.create_dataset("positionStart", data=positions)
        particle.create_dataset("velocityStart", data=velocity)
        particle.create_dataset("radiusStart", data=np.array([0.5, 0.5]))
        particle.create_dataset("localFeatureStart", data=local)
        particle.create_dataset("positionEnd", data=position_end)
        particle.create_dataset("velocityEnd", data=velocity_end)
        particle.create_dataset(
            "gravityReferenceVelocityEnd", data=velocity + [0.0, -0.001, 0.0]
        )
        particle.create_dataset("targetResidualAcceleration", data=target)
        particle.create_dataset("valid", data=valid)
        frame_grid = frame.create_group("grid")
        frame_grid.create_dataset(
            "wallStart", data=make_wall_channels(geometry).numpy()
        )
        for name in ("wallStart", "wallEnd"):
            wall = frame.create_group(name)
            wall.attrs["bodyUuidsJson"] = np.array(["[]"], dtype=text_type)
            wall.create_dataset("centerOfMass", data=np.empty((0, 3)))
            wall.create_dataset("rotationWxyz", data=np.empty((0, 4)))
            wall.create_dataset("linearVelocity", data=np.empty((0, 3)))
            wall.create_dataset("angularVelocity", data=np.empty((0, 3)))


class GeometryAndTransferTests(unittest.TestCase):
    def test_geometry_coordinates_mask_and_x_contiguous_layout(self) -> None:
        geometry = make_geometry()
        centers = geometry.cell_centers(dtype=torch.float64, device="cpu")
        coordinates = geometry.coordinate_channels(dtype=torch.float64, device="cpu")
        mask = geometry.valid_domain_mask(dtype=torch.float64, device="cpu")

        self.assertEqual(tuple(centers.shape), (3, 8, 8, 8))
        self.assertEqual(centers[0, 0, 0, 0].item(), -2.5)
        self.assertEqual(centers[0, 0, 0, 1].item(), -1.5)
        self.assertAlmostEqual(coordinates[0, 0, 0, 0].item(), -0.875)
        self.assertEqual(mask.sum().item(), 8.0)
        self.assertTrue(centers.is_contiguous())
        self.assertEqual(centers.stride()[-1], 1)

    def test_float32_centers_match_cpp_boundary_rounding(self) -> None:
        geometry = hybrid.GridGeometry(
            physical_bounds_min=(-0.7274999618530273, -0.44999998807907104,
                                 -0.30000001192092896),
            physical_bounds_max=(3.947499990463257, 1.4500000476837158,
                                 1.2313003540039062),
            padded_bounds_min=(-0.8774999640882015, -0.5999999903142452,
                               -0.45000001415610313),
            padded_bounds_max=(4.1225001104176044, 1.6500000432133675,
                               1.400000013411045),
            cell_counts=(100, 45, 37),
            cell_size=0.05000000074505806,
            padding_cells=3,
        )
        mask = geometry.valid_domain_mask(dtype=torch.float32, device="cpu")

        self.assertEqual(mask.sum().item(), 110732.0)
        self.assertTrue(mask[0, 3, 3, 96].item())

    def test_stencil_weight_sum_cell_center_and_no_clamp(self) -> None:
        geometry = make_geometry()
        positions = torch.tensor(
            [[0.5, 0.5, 0.5], [0.25, 0.75, 1.25]], dtype=torch.float64
        )
        stencil = hybrid.trilinear_stencil(positions, geometry)

        torch.testing.assert_close(
            stencil.weights.sum(dim=1), torch.ones(2, dtype=torch.float64)
        )
        self.assertEqual(stencil.weights[0, 0].item(), 1.0)
        self.assertEqual(torch.count_nonzero(stencil.weights[0]).item(), 1)
        self.assertTrue(stencil.valid.all())

        outside = torch.tensor([[4.5, 0.5, 0.5]], dtype=torch.float64)
        invalid = hybrid.trilinear_stencil(outside, geometry)
        self.assertFalse(invalid.valid.item())
        self.assertTrue(torch.all(invalid.linear_indices == -1))
        with self.assertRaises(hybrid.IncompleteGridSupportError):
            hybrid.reference_g2p(
                torch.zeros(16, *geometry.tensor_shape, dtype=torch.float64),
                outside,
                geometry,
            )

    def test_volume_weighted_p2g_and_constant_g2p(self) -> None:
        geometry = make_geometry()
        positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
        result = hybrid.reference_p2g(
            positions,
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
            torch.tensor([0.75], dtype=torch.float64),
            torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float64),
            geometry,
        )

        expected = torch.tensor(
            [1.0, 2.0, 3.0, 0.75, 4.0, 5.0, 6.0], dtype=torch.float64
        )
        torch.testing.assert_close(
            result.dynamic_grid[:7, 3, 3, 3], expected, rtol=1.0e-10, atol=1.0e-10
        )
        self.assertEqual(result.dynamic_grid[7, 3, 3, 3].item(), 1.0)
        self.assertEqual(result.dynamic_grid[7].count_nonzero().item(), 1)
        self.assertTrue(result.dynamic_grid.is_contiguous())
        self.assertEqual(result.dynamic_grid.stride()[-1], 1)

        channel_values = torch.arange(16, dtype=torch.float64)
        latent = channel_values[:, None, None, None].expand(-1, *geometry.tensor_shape)
        particle_latent = hybrid.reference_g2p(latent, positions, geometry)
        torch.testing.assert_close(particle_latent[0], channel_values)


class DataAndNormalizationTests(unittest.TestCase):
    def test_hdf5_loader_preserves_valid_nan_and_named_conditions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-loader-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            frames = hybrid.load_teacher_trajectory(path)

        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame.condition_names, ("omegaStart", "omegaEnd"))
        self.assertEqual(frame.conditions, (1.0, 2.0))
        self.assertEqual(frame.static_id.tolist(), [10, 20])
        self.assertEqual(frame.valid.tolist(), [True, False])
        self.assertTrue(np.isnan(frame.target_residual_acceleration[1]).all())
        self.assertEqual(frame.geometry.tensor_shape, (8, 8, 8))
        np.testing.assert_array_equal(
            frame.wall_channels_start, make_wall_channels(frame.geometry).numpy()
        )

    def test_hdf5_loader_rejects_duplicate_static_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-loader-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path, duplicate_static_id=True)
            with self.assertRaisesRegex(hybrid.ContractError, "staticId"):
                hybrid.load_teacher_trajectory(path)

    def test_training_cli_indexes_then_streams_stored_wall_grid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-cli-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            indexes = hybrid_trainer.load_dataset((path,))
            self.assertEqual(len(indexes), 1)
            self.assertIsInstance(indexes[0], hybrid.TeacherFrameIndex)
            self.assertFalse(hasattr(indexes[0], "static_id"))
            frame = hybrid.load_teacher_frame(indexes[0])

        np.testing.assert_array_equal(
            frame.wall_channels_start,
            make_wall_channels(frame.geometry).numpy(),
        )

    def test_streaming_statistics_match_eager_statistics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-stream-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            indexes = hybrid.index_teacher_trajectory(path)
            frame = hybrid.load_teacher_frame(indexes[0])
            sample = hybrid.HybridFrameSample(
                frame=frame,
                wall_channels=torch.as_tensor(frame.wall_channels_start),
            )
            split = hybrid.split_trajectories(indexes, fractions=(1.0, 0.0, 0.0))
            eager = hybrid.compute_base_training_statistics([sample], split)
            streaming = hybrid.compute_base_training_statistics_streaming(
                indexes, split
            )
            prepared = hybrid.prepare_indexed_frame(indexes[0], streaming)

        torch.testing.assert_close(streaming.dynamic.mean, eager.dynamic.mean)
        torch.testing.assert_close(streaming.dynamic.std, eager.dynamic.std)
        torch.testing.assert_close(streaming.target.mean, eager.target.mean)
        torch.testing.assert_close(streaming.target.std, eager.target.std)
        self.assertEqual(streaming.dynamic.count, eager.dynamic.count)
        self.assertEqual(streaming.target.count, eager.target.count)
        self.assertEqual(tuple(prepared.grid_input.shape), (1, 21, 8, 8, 8))

    def test_evenly_spaced_frame_subset_is_deterministic_and_includes_ends(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-subset-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            base = hybrid.index_teacher_trajectory(path)[0]
        indexes = tuple(
            dataclasses.replace(
                base,
                frame_name=f"{tick:012d}",
                macro_step_index=tick,
                time_start=tick * base.ai_delta_time,
                time_end=(tick + 1) * base.ai_delta_time,
            )
            for tick in range(4000, 4300)
        )

        selected = hybrid_trainer.select_frame_subset(indexes, 20)

        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[0].macro_step_index, 4000)
        self.assertEqual(selected[-1].macro_step_index, 4299)
        self.assertEqual(
            [index.macro_step_index for index in selected],
            [4000 + index * 299 // 19 for index in range(20)],
        )

    def test_frame_subset_rejects_multiple_trajectories_and_invalid_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-subset-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            base = hybrid.index_teacher_trajectory(path)[0]
        indexes = (
            base,
            dataclasses.replace(
                base,
                trajectory_id="second",
                macro_step_index=base.macro_step_index + 1,
            ),
        )

        with self.assertRaises(hybrid.ContractError):
            hybrid_trainer.select_frame_subset(indexes, 1)
        with self.assertRaises(hybrid.ContractError):
            hybrid_trainer.select_frame_subset((base,), 0)
        with self.assertRaises(hybrid.ContractError):
            hybrid_trainer.select_frame_subset((base,), 2)

    def test_training_checkpoint_contains_cpu_model_state(self) -> None:
        model = hybrid.HybridReferenceModel(condition_count=0)
        with tempfile.TemporaryDirectory(
            prefix="shondy-hybrid-checkpoint-"
        ) as directory:
            path = Path(directory) / "checkpoint.pt"
            hybrid_trainer._write_checkpoint(path, model, epoch=20, training_loss=0.25)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["epoch"], 20)
        self.assertEqual(checkpoint["trainingStandardizedMse"], 0.25)
        self.assertTrue(checkpoint["modelState"])
        self.assertTrue(
            all(
                value.device.type == "cpu"
                for value in checkpoint["modelState"].values()
            )
        )

    def test_fixed_weight_training_diagnostics_are_streamed_per_frame(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-evaluate-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            index = hybrid.index_teacher_trajectory(path)[0]
            split = hybrid.split_trajectories([index], fractions=(1.0, 0.0, 0.0))
            statistics = hybrid.compute_base_training_statistics_streaming(
                [index], split
            )
            model = hybrid.HybridReferenceModel(condition_count=2)

            diagnostics = hybrid_trainer.evaluate_standardized_mse_diagnostics(
                model, [index], statistics, device="cpu"
            )
            scalar = hybrid_trainer.evaluate_standardized_mse(
                model, [index], statistics, device="cpu"
            )

        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertEqual(diagnostics["evaluatedFrames"], 1)
        self.assertEqual(diagnostics["validParticles"], 1)
        self.assertEqual(len(diagnostics["componentStandardizedMse"]), 3)
        self.assertEqual(len(diagnostics["perFrame"]), 1)
        self.assertEqual(diagnostics["perFrame"][0]["validParticles"], 1)
        self.assertEqual(diagnostics["perFrame"][0]["collisionCandidateParticles"], 0)
        self.assertEqual(diagnostics["collisionCandidateParticles"], 0)
        self.assertIsNone(diagnostics["collisionCandidateStandardizedMse"])
        self.assertIn("1.0", diagnostics["absoluteStandardizedTargetThresholds"])
        self.assertAlmostEqual(
            scalar, diagnostics["frameMeanStandardizedMse"], places=7
        )
        self.assertAlmostEqual(
            diagnostics["particleWeightedStandardizedMse"],
            diagnostics["frameMeanStandardizedMse"],
            places=7,
        )

    def test_occupancy_aware_normalization_and_grid_assembly(self) -> None:
        geometry = make_geometry()
        dynamic = torch.zeros(8, *geometry.tensor_shape, dtype=torch.float64)
        dynamic[:7, 3, 3, 3] = torch.arange(1, 8, dtype=torch.float64)
        dynamic[7, 3, 3, 3] = 0.5
        statistics = make_statistics([10.0] * 7, [2.0] * 7)
        normalized = hybrid.normalize_dynamic_grid(dynamic, statistics)

        self.assertTrue(torch.all(normalized[:7, 0, 0, 0] == 0.0))
        torch.testing.assert_close(
            normalized[:7, 3, 3, 3],
            (torch.arange(1, 8, dtype=torch.float64) - 10.0) / 2.0,
        )
        self.assertEqual(normalized[7, 3, 3, 3].item(), 0.5)

        wall = make_wall_channels(geometry)
        condition_stats = make_statistics([1.0, 3.0], [2.0, 4.0])
        grid_input = hybrid.assemble_grid_input(
            normalized,
            wall,
            geometry,
            torch.tensor([3.0, -1.0], dtype=torch.float64),
            condition_stats,
        )
        self.assertEqual(tuple(grid_input.shape), (1, 21, 8, 8, 8))
        self.assertTrue(grid_input.is_contiguous())
        self.assertEqual(grid_input.stride()[-1], 1)
        self.assertEqual(grid_input[0, 19, 0, 0, 0].item(), 1.0)
        self.assertEqual(grid_input[0, 20, 0, 0, 0].item(), -1.0)

    def test_training_statistics_and_prepare_frame_use_only_valid_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-prepare-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            frame = hybrid.load_teacher_trajectory(path)[0]
        sample = hybrid.HybridFrameSample(
            frame=frame, wall_channels=make_wall_channels(frame.geometry)
        )
        split = hybrid.split_trajectories([frame])
        statistics = hybrid.compute_base_training_statistics([sample], split)
        prepared = hybrid.prepare_frame(sample, statistics)

        self.assertEqual(statistics.target.count, 1)
        torch.testing.assert_close(
            statistics.target.mean,
            torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
        )
        self.assertTrue(statistics.target.constant_mask.all())
        self.assertEqual(tuple(prepared.grid_input.shape), (1, 21, 8, 8, 8))
        self.assertTrue(torch.isfinite(prepared.standardized_target[0]).all())
        self.assertTrue(torch.isnan(prepared.standardized_target[1]).all())

    def test_invalid_grid_support_is_skipped_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-support-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            base = hybrid.load_teacher_trajectory(path)[0]

        valid_frame = dataclasses.replace(base, trajectory_id="valid-trajectory")
        invalid_frame = dataclasses.replace(
            base,
            trajectory_id="invalid-trajectory",
            valid_grid_support=False,
            target_residual_acceleration=np.full_like(
                base.target_residual_acceleration, 1.0e9
            ),
        )
        valid_sample = hybrid.HybridFrameSample(
            frame=valid_frame,
            wall_channels=make_wall_channels(valid_frame.geometry),
        )
        invalid_sample = hybrid.HybridFrameSample(
            frame=invalid_frame,
            wall_channels=make_wall_channels(invalid_frame.geometry),
        )
        split = hybrid.TrajectorySplit(
            training=(valid_frame.trajectory_key, invalid_frame.trajectory_key),
            validation=(),
            test=(),
        )

        statistics = hybrid.compute_base_training_statistics(
            [valid_sample, invalid_sample], split
        )
        self.assertEqual(statistics.target.count, 1)
        torch.testing.assert_close(
            statistics.target.mean,
            torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
        )
        with self.assertRaises(hybrid.IncompleteGridSupportError):
            hybrid.prepare_frame(invalid_sample, statistics)

        invalid_only_split = hybrid.TrajectorySplit(
            training=(invalid_frame.trajectory_key,), validation=(), test=()
        )
        with self.assertRaisesRegex(hybrid.ContractError, "valid grid support"):
            hybrid.compute_base_training_statistics(
                [invalid_sample], invalid_only_split
            )

    def test_statistics_use_training_trajectories_only(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="shondy-hybrid-statistics-"
        ) as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            base = hybrid.load_teacher_trajectory(path)[0]

        training_frame = dataclasses.replace(base, trajectory_id="training-trajectory")
        validation_local = base.local_feature_start.copy()
        validation_local[:, 3] = 1.0e8
        validation_local[:, 8:11] = 1.0e8
        validation_frame = dataclasses.replace(
            base,
            trajectory_id="validation-trajectory",
            velocity_start=np.full_like(base.velocity_start, 1.0e8),
            local_feature_start=validation_local,
            target_residual_acceleration=np.full_like(
                base.target_residual_acceleration, 1.0e8
            ),
            conditions=(1.0e8, 1.0e8),
        )
        training_sample = hybrid.HybridFrameSample(
            frame=training_frame,
            wall_channels=make_wall_channels(training_frame.geometry),
        )
        validation_sample = hybrid.HybridFrameSample(
            frame=validation_frame,
            wall_channels=make_wall_channels(validation_frame.geometry),
        )
        split = hybrid.TrajectorySplit(
            training=(training_frame.trajectory_key,),
            validation=(validation_frame.trajectory_key,),
            test=(),
        )

        expected = hybrid.compute_base_training_statistics([training_sample], split)
        actual = hybrid.compute_base_training_statistics(
            [training_sample, validation_sample], split
        )

        torch.testing.assert_close(actual.dynamic.mean, expected.dynamic.mean)
        torch.testing.assert_close(actual.dynamic.std, expected.dynamic.std)
        self.assertEqual(actual.dynamic.count, expected.dynamic.count)
        self.assertIsNotNone(actual.condition)
        self.assertIsNotNone(expected.condition)
        torch.testing.assert_close(actual.condition.mean, expected.condition.mean)
        torch.testing.assert_close(actual.condition.std, expected.condition.std)
        self.assertEqual(actual.condition.count, 1)
        torch.testing.assert_close(actual.target.mean, expected.target.mean)
        torch.testing.assert_close(actual.target.std, expected.target.std)
        self.assertEqual(actual.target.count, 1)

    def test_trajectory_split_has_no_cross_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-split-") as directory:
            path = Path(directory) / "teacher.h5"
            write_teacher_file(path)
            base = hybrid.load_teacher_trajectory(path)[0]
        frames = []
        for trajectory_index in range(10):
            for frame_index in range(2):
                frames.append(
                    dataclasses.replace(
                        base,
                        trajectory_id=f"trajectory-{trajectory_index}",
                        macro_step_index=frame_index,
                    )
                )
        split = hybrid.split_trajectories(frames, seed=9)

        assigned = {}
        for frame in frames:
            name = split.assignment(frame)
            assigned.setdefault(frame.trajectory_key, set()).add(name)
        self.assertTrue(all(len(values) == 1 for values in assigned.values()))
        self.assertEqual(len(assigned), 10)
        self.assertTrue(split.training)
        self.assertTrue(split.validation)
        self.assertTrue(split.test)

        with self.assertRaisesRegex(hybrid.ContractError, "Duplicate frame"):
            hybrid.split_trajectories([frames[0], frames[0]])


class ModelAndExportTests(unittest.TestCase):
    def test_unet_contract_padding_crop_and_linear_latent_head(self) -> None:
        model = hybrid.CompactGridEncoder(21)
        value = torch.randn(1, 21, 9, 10, 11)
        output = model(value)

        self.assertEqual(tuple(output.shape), (1, 16, 9, 10, 11))
        self.assertEqual(model.down_half.downsample.stride, (2, 2, 2))
        self.assertEqual(model.down_quarter.downsample.out_channels, 64)
        self.assertEqual(model.down_eighth.downsample.out_channels, 96)
        self.assertIsInstance(model.output, torch.nn.Conv3d)
        for module in model.modules():
            if isinstance(module, torch.nn.GroupNorm):
                self.assertEqual(module.num_groups, 8)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-1.0)
            output = model(value)
        self.assertTrue(torch.all(output == -1.0))
        scripted = torch.jit.script(model)
        self.assertEqual(tuple(scripted(value).shape), (1, 16, 9, 10, 11))

    def test_particle_mlp_and_standardized_mse(self) -> None:
        model = hybrid.ParticleMLP()
        self.assertEqual(tuple(model.layer1.weight.shape), (128, 34))
        self.assertEqual(tuple(model.layer2.weight.shape), (128, 128))
        self.assertEqual(tuple(model.layer3.weight.shape), (64, 128))
        self.assertEqual(tuple(model.output.weight.shape), (3, 64))
        prediction = model(torch.zeros(2, 34))
        target = torch.stack((prediction[0], torch.full((3,), float("nan"))))
        loss = hybrid.standardized_target_mse(
            prediction, target, torch.tensor([True, False])
        )
        self.assertEqual(loss.item(), 0.0)

        position_error = hybrid.normalized_position_error(
            torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[3.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
            torch.tensor([[0.2, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
            torch.tensor([0.5, 0.5]),
            0.1,
            torch.tensor([True, False]),
        )
        torch.testing.assert_close(position_error, torch.zeros(1))

    def test_latent_calibration_preserves_particle_predictions(self) -> None:
        torch.manual_seed(3)
        model = hybrid.HybridReferenceModel(condition_count=0)
        value = torch.randn(5, 34)
        before = model.particle_mlp(value).detach()
        statistics = make_statistics(
            [float(index) / 10.0 for index in range(16)],
            [1.0 + float(index) / 20.0 for index in range(16)],
        )
        hybrid.calibrate_latent_standardization(model, statistics)
        transformed = value.clone()
        raw_latent = value[:, 18:]
        transformed[:, 18:] = (
            raw_latent - statistics.mean.to(torch.float32)
        ) / statistics.std.to(torch.float32)
        after = model.particle_mlp(transformed).detach()
        torch.testing.assert_close(after, before, rtol=1.0e-5, atol=1.0e-6)

    def test_joint_model_backward_and_separate_artifact_metadata_export(self) -> None:
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory(prefix="shondy-hybrid-export-") as directory:
            directory_path = Path(directory)
            teacher_path = directory_path / "teacher.h5"
            write_teacher_file(teacher_path)
            index = hybrid.index_teacher_trajectory(teacher_path)[0]
            frame = hybrid.load_teacher_frame(index)
            sample = hybrid.HybridFrameSample(
                frame=frame, wall_channels=make_wall_channels(frame.geometry)
            )
            split = hybrid.split_trajectories([index], fractions=(1.0, 0.0, 0.0))
            base_statistics = hybrid.compute_base_training_statistics([sample], split)
            prepared = hybrid.prepare_frame(sample, base_statistics)
            model = hybrid.HybridReferenceModel(condition_count=2).to(
                dtype=torch.float64
            )
            prediction = model(
                prepared.grid_input,
                prepared.positions,
                prepared.local_features,
                prepared.geometry,
            )
            loss = hybrid.standardized_target_mse(
                prediction, prepared.standardized_target, prepared.valid
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.grid_encoder.output.weight.grad)
            self.assertIsNotNone(model.particle_mlp.layer1.weight.grad)

            training_result = hybrid.train_reference_model_streaming(
                model,
                [index],
                base_statistics,
                split,
                epochs=1,
                learning_rate=1.0e-4,
                device="cpu",
            )
            self.assertEqual(len(training_result.epoch_losses), 1)
            self.assertTrue(math.isfinite(training_result.epoch_losses[0]))
            statistics = dataclasses.replace(
                base_statistics, latent=training_result.latent_statistics
            )
            metadata = hybrid.export_reference_artifacts(
                directory_path / "artifacts",
                model,
                statistics,
                [index],
                split,
                validate_cuda=False,
            )
            grid_path = directory_path / "artifacts/grid-encoder.pt"
            particle_path = directory_path / "artifacts/particle-mlp.pt"
            grid_onnx_path = directory_path / "artifacts/grid-encoder.onnx"
            particle_onnx_path = directory_path / "artifacts/particle-mlp.onnx"
            particle_native_path = directory_path / "artifacts/particle-mlp-native.json"
            metadata_path = directory_path / "artifacts/model-metadata.json"
            disk_metadata = json.loads(metadata_path.read_text(encoding="ascii"))

            self.assertTrue(grid_path.is_file())
            self.assertTrue(particle_path.is_file())
            self.assertTrue(grid_onnx_path.is_file())
            self.assertTrue(particle_onnx_path.is_file())
            self.assertTrue(particle_native_path.is_file())
            self.assertEqual(metadata, disk_metadata)
            self.assertEqual(metadata["schemaVersion"], 2)
            self.assertEqual(metadata["contractRegistrySha256"], hybrid.REGISTRY_SHA256)
            self.assertEqual(
                metadata["certificationProfile"],
                hybrid.CONTRACT["certificationProfiles"]["extendedUnverified"],
            )
            self.assertEqual(
                metadata["channels"]["gridBase"],
                list(hybrid.BASE_GRID_CHANNEL_NAMES),
            )
            self.assertTrue(metadata["target"]["targetIncludesTeacherCollision"])
            self.assertTrue(metadata["runtime"]["onnxArtifactsProduced"])
            self.assertEqual(
                metadata["runtime"]["expectedOnnxRuntimeProvider"],
                "CUDAExecutionProvider",
            )
            for consistency_name in (
                "gridEncoderConsistency",
                "particleMlpConsistency",
            ):
                consistency = metadata["runtime"][consistency_name]
                self.assertEqual(consistency["status"], "notRun")
            for artifact_name, path in (
                ("gridEncoder", grid_onnx_path),
                ("particleMlp", particle_onnx_path),
            ):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(metadata["artifacts"][artifact_name]["sha256"], digest)
                onnx.checker.check_model(onnx.load(str(path)))
            native = json.loads(particle_native_path.read_text(encoding="ascii"))
            self.assertEqual(native["architecture"], [34, 128, 128, 64, 3])
            self.assertEqual(len(native["parameters"]), 29443)
            native_digest = hashlib.sha256(
                particle_native_path.read_bytes()
            ).hexdigest()
            self.assertEqual(
                metadata["artifacts"]["particleMlpNative"]["sha256"],
                native_digest,
            )
            for artifact_name, path in (
                ("gridEncoder", grid_path),
                ("particleMlp", particle_path),
            ):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(
                    metadata["artifacts"][artifact_name]["torchscriptCompatibility"][
                        "sha256"
                    ],
                    digest,
                )

            scripted_grid = torch.jit.load(str(grid_path))
            scripted_particle = torch.jit.load(str(particle_path))
            export_input = prepared.grid_input.to(torch.float32)
            combined = torch.zeros(2, 34)
            grid_output = scripted_grid(export_input)
            particle_output = scripted_particle(combined)
            expected_model = copy.deepcopy(model).to(dtype=torch.float32).eval()
            with torch.no_grad():
                expected_grid = expected_model.standardized_grid_latent(export_input)
                expected_particle = expected_model.particle_mlp(
                    combined
                ) * base_statistics.target.std.to(
                    torch.float32
                ) + base_statistics.target.mean.to(torch.float32)
            self.assertEqual(tuple(grid_output.shape), (1, 16, 8, 8, 8))
            self.assertEqual(tuple(particle_output.shape), (2, 3))
            torch.testing.assert_close(
                grid_output, expected_grid, rtol=1.0e-5, atol=1.0e-6
            )
            torch.testing.assert_close(
                particle_output, expected_particle, rtol=1.0e-5, atol=1.0e-6
            )

            grid_session = ort.InferenceSession(
                str(grid_onnx_path), providers=["CPUExecutionProvider"]
            )
            particle_session = ort.InferenceSession(
                str(particle_onnx_path), providers=["CPUExecutionProvider"]
            )
            grid_onnx_output = grid_session.run(
                None, {"gridInput": export_input.numpy()}
            )[0]
            particle_onnx_output = particle_session.run(
                None, {"particleInput": combined.numpy()}
            )[0]
            np.testing.assert_allclose(
                grid_onnx_output, expected_grid.numpy(), rtol=2.0e-3, atol=1.0e-5
            )
            np.testing.assert_allclose(
                particle_onnx_output,
                expected_particle.numpy(),
                rtol=2.0e-3,
                atol=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
