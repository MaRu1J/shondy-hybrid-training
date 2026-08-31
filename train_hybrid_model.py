"""Command-line training and export for the dense-grid hybrid surrogate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import torch

import hybrid_reference_model as hybrid


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the shonDy dense-grid hybrid model."
    )
    parser.add_argument(
        "teacher",
        nargs="+",
        type=Path,
        help="One or more published Schema 3 Teacher HDF5 trajectories.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--model-profile",
        choices=tuple(hybrid.MODEL_PROFILES),
        default=hybrid.DEFAULT_MODEL_PROFILE,
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preprocessing-device",
        help=(
            "Device for the one-time P2G statistics pass; defaults to --device. "
            "Use cpu to retain the legacy P2G reduction numerics."
        ),
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--prefetch-frames",
        type=int,
        default=2,
        help="Validated Teacher frames to load ahead while the device is training.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=1.0,
        help="Maximum interval between non-interactive epoch progress updates.",
    )
    parser.add_argument(
        "--dynamic-grid-cache-gib",
        type=float,
        default=8.0,
        help="Maximum host-memory cache for deterministic P2G grids; zero disables it.",
    )
    parser.add_argument(
        "--training-frame-cache-gib",
        type=float,
        default=16.0,
        help="Maximum host-memory cache for minimal validated frame tensors.",
    )
    parser.add_argument(
        "--frame-subset-count",
        type=int,
        help=(
            "Select an evenly spaced, endpoint-inclusive subset from exactly one "
            "trajectory."
        ),
    )
    parser.add_argument(
        "--checkpoint-epochs",
        nargs="*",
        type=int,
        default=(),
        help="Epochs at which to save CPU model-state checkpoints.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Restore a profile-checked model-state checkpoint before training.",
    )
    parser.add_argument(
        "--split-fractions",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--collision-post-process",
        choices=("full", "guard", "none"),
        default="full",
    )
    parser.add_argument(
        "--skip-cuda-export-validation",
        action="store_true",
        help="Skip PyTorch/ONNX Runtime CUDA consistency checks.",
    )
    return parser.parse_args()


def load_dataset(
    paths: list[Path] | tuple[Path, ...],
) -> tuple[hybrid.TeacherFrameIndex, ...]:
    indexes = tuple(
        index for path in paths for index in hybrid.index_teacher_trajectory(path)
    )
    if not indexes:
        raise hybrid.ContractError("Training requires at least one Teacher frame.")
    hybrid.require_common_frame_contract(indexes)
    return indexes


def select_frame_subset(
    indexes: tuple[hybrid.TeacherFrameIndex, ...], frame_count: int | None
) -> tuple[hybrid.TeacherFrameIndex, ...]:
    """Select deterministic endpoint-inclusive frames for single-trajectory overfit."""

    if frame_count is None:
        return indexes
    if frame_count <= 0:
        raise hybrid.ContractError("--frame-subset-count must be positive.")
    trajectory_keys = {index.trajectory_key for index in indexes}
    if len(trajectory_keys) != 1:
        raise hybrid.ContractError(
            "--frame-subset-count requires exactly one Teacher trajectory."
        )
    ordered = tuple(sorted(indexes, key=lambda index: index.macro_step_index))
    if frame_count > len(ordered):
        raise hybrid.ContractError(
            "--frame-subset-count cannot exceed the available frame count."
        )
    if frame_count == 1:
        return (ordered[0],)
    positions = tuple(
        index * (len(ordered) - 1) // (frame_count - 1) for index in range(frame_count)
    )
    return tuple(ordered[position] for position in positions)


def _write_checkpoint(
    path: Path,
    model: hybrid.HybridReferenceModel,
    *,
    epoch: int,
    training_loss: float,
) -> None:
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "epoch": epoch,
            "trainingStandardizedMse": training_loss,
            "modelProfile": model.model_profile,
            "architecture": hybrid.model_architecture(model),
            "modelState": state,
        },
        path,
    )


def _load_checkpoint(path: Path, model: hybrid.HybridReferenceModel) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise hybrid.ContractError("Checkpoint must contain an object.")
    if checkpoint.get("modelProfile") != model.model_profile:
        raise hybrid.ContractError(
            "Checkpoint model profile does not match CLI profile."
        )
    if checkpoint.get("architecture") != hybrid.model_architecture(model):
        raise hybrid.ContractError("Checkpoint architecture metadata is inconsistent.")
    state = checkpoint.get("modelState")
    if not isinstance(state, dict):
        raise hybrid.ContractError("Checkpoint is missing modelState.")
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        not isinstance(state[name], torch.Tensor)
        or state[name].shape != expected[name].shape
        for name in expected
    ):
        raise hybrid.ContractError("Checkpoint tensor shapes do not match its profile.")
    model.load_state_dict(state, strict=True)
    return checkpoint


def _split_counts(
    frames: tuple[hybrid.TeacherFrameIndex, ...], split: hybrid.TrajectorySplit
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in ("training", "validation", "test"):
        keys = getattr(split, name)
        result[name] = {
            "trajectories": len(keys),
            "frames": sum(split.assignment(frame) == name for frame in frames),
        }
    return result


class EpochProgressReporter:
    def __init__(
        self,
        epoch_count: int,
        interval_seconds: float,
        *,
        stream: TextIO = sys.stderr,
    ) -> None:
        if (
            epoch_count <= 0
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0.0
        ):
            raise hybrid.ContractError("Progress reporting values must be positive.")
        self.epoch_count = epoch_count
        self.interval_seconds = interval_seconds
        self.stream = stream
        self.interactive = stream.isatty()
        self.current_epoch = 0
        self.epoch_started = 0.0
        self.last_reported = 0.0

    def __call__(self, epoch: int, completed: int, total: int) -> None:
        if total <= 0 or completed < 0 or completed > total:
            raise hybrid.ContractError("Invalid epoch progress counters.")
        now = time.perf_counter()
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self.epoch_started = now
            self.last_reported = 0.0
        if (
            not self.interactive
            and completed not in (0, total)
            and now - self.last_reported < self.interval_seconds
        ):
            return
        elapsed = max(0.0, now - self.epoch_started)
        fraction = completed / total
        eta = elapsed * (total - completed) / completed if completed else None
        if self.interactive:
            width = 30
            filled = min(width, int(fraction * width))
            bar = "#" * filled + "." * (width - filled)
            eta_text = f"{eta:7.1f}s" if eta is not None else "   --.-s"
            self.stream.write(
                f"\rEpoch {epoch}/{self.epoch_count} [{bar}] "
                f"{completed}/{total} {100.0 * fraction:6.2f}% "
                f"elapsed {elapsed:7.1f}s ETA {eta_text}"
            )
            if completed == total:
                self.stream.write("\n")
            self.stream.flush()
        else:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "progressFrames": completed,
                        "totalFrames": total,
                        "progressFraction": fraction,
                        "elapsedSeconds": elapsed,
                        "etaSeconds": eta,
                    },
                    sort_keys=True,
                ),
                file=self.stream,
                flush=True,
            )
        self.last_reported = now


@torch.no_grad()
def evaluate_standardized_mse_diagnostics(
    model: hybrid.HybridReferenceModel,
    indexes: list[hybrid.TeacherFrameIndex],
    statistics: hybrid.TrainingStatistics,
    *,
    device: torch.device | str,
    dynamic_grid_cache: dict[tuple[str, str, int], torch.Tensor] | None = None,
    wall_grid_cache: dict[Path, torch.Tensor] | None = None,
    training_frame_cache: dict[tuple[str, str, int], hybrid.TrainingFrameData]
    | None = None,
    prefetch_frames: int = 0,
    validated_indexes: bool = False,
) -> dict[str, Any] | None:
    if not indexes:
        return None
    model.to(device=device)
    model.eval()
    frame_losses: list[float] = []
    component_squared_error = torch.zeros(3, dtype=torch.float64)
    total_squared_error = 0.0
    total_valid_particles = 0
    collision_squared_error = 0.0
    collision_value_count = 0
    non_collision_squared_error = 0.0
    non_collision_value_count = 0
    absolute_target_thresholds = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    threshold_accumulators = {
        threshold: {"valueCount": 0, "modelSquaredError": 0.0, "targetSquared": 0.0}
        for threshold in absolute_target_thresholds
    }
    total_target_squared = 0.0
    per_frame: list[dict[str, Any]] = []
    parameter = next(model.parameters())
    frames = hybrid.iter_prepared_training_frames(
        indexes,
        statistics,
        device=parameter.device,
        dtype=parameter.dtype,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=prefetch_frames,
        validated_indexes=validated_indexes,
    )
    for index, frame in zip(indexes, frames, strict=True):
        local = frame.local_features
        target = frame.standardized_target
        valid = frame.valid
        prediction = model(
            frame.grid_input,
            frame.positions,
            local,
            frame.geometry,
            validate_inputs=not validated_indexes,
        )
        selected = torch.logical_and(valid, torch.isfinite(target).all(dim=1))
        if not torch.any(selected):
            raise hybrid.ContractError(
                f"Frame {frame.frame_key} has no valid finite target for evaluation."
            )
        selected_error = prediction[selected] - target[selected]
        squared_error = selected_error.square()
        frame_loss = float(squared_error.mean().cpu())
        valid_particle_count = int(selected.sum().cpu())
        frame_losses.append(frame_loss)
        component_squared_error += squared_error.sum(dim=0).to(
            device="cpu", dtype=torch.float64
        )
        total_squared_error += float(squared_error.sum().cpu())
        total_valid_particles += valid_particle_count

        selected_local = local[selected]
        wall_approach_speed = torch.sum(
            selected_local[:, 11:14] * selected_local[:, 14:17], dim=1
        )
        collision_candidate = torch.logical_and(
            selected_local[:, 17] < 1.0, wall_approach_speed > 0.0
        )
        collision_error = squared_error[collision_candidate]
        non_collision_error = squared_error[~collision_candidate]
        collision_squared_error += float(collision_error.sum().cpu())
        collision_value_count += collision_error.numel()
        non_collision_squared_error += float(non_collision_error.sum().cpu())
        non_collision_value_count += non_collision_error.numel()

        absolute_target = target[selected].abs().flatten()
        target_squared = target[selected].square().flatten()
        total_target_squared += float(target_squared.sum().cpu())
        flattened_squared_error = squared_error.flatten()
        for threshold, accumulator in threshold_accumulators.items():
            above = absolute_target > threshold
            accumulator["valueCount"] += int(above.sum().cpu())
            accumulator["modelSquaredError"] += float(
                flattened_squared_error[above].sum().cpu()
            )
            accumulator["targetSquared"] += float(target_squared[above].sum().cpu())
        quantiles = torch.quantile(
            absolute_target,
            torch.tensor(
                [0.5, 0.9, 0.99, 0.999],
                device=absolute_target.device,
                dtype=absolute_target.dtype,
            ),
        ).cpu()
        per_frame.append(
            {
                "macroStepIndex": index.macro_step_index,
                "timeStart": index.time_start,
                "validParticles": valid_particle_count,
                "collisionCandidateParticles": int(collision_candidate.sum().cpu()),
                "standardizedMse": frame_loss,
                "componentStandardizedMse": (squared_error.mean(dim=0).cpu().tolist()),
                "absoluteStandardizedTargetQuantiles": {
                    "p50": float(quantiles[0]),
                    "p90": float(quantiles[1]),
                    "p99": float(quantiles[2]),
                    "p999": float(quantiles[3]),
                    "max": float(absolute_target.max().cpu()),
                },
            }
        )

    value_count = total_valid_particles * hybrid.TARGET_CHANNEL_COUNT
    return {
        "evaluatedFrames": len(frame_losses),
        "validParticles": total_valid_particles,
        "frameMeanStandardizedMse": sum(frame_losses) / len(frame_losses),
        "particleWeightedStandardizedMse": total_squared_error / value_count,
        "componentStandardizedMse": (
            component_squared_error / total_valid_particles
        ).tolist(),
        "collisionCandidateDefinition": (
            "normalizedWallDistance < 1 and "
            "dot(nearestWallNormal, wallRelativeVelocity) > 0"
        ),
        "collisionCandidateStandardizedMse": (
            collision_squared_error / collision_value_count
            if collision_value_count
            else None
        ),
        "collisionCandidateParticles": collision_value_count // 3,
        "nonCollisionStandardizedMse": (
            non_collision_squared_error / non_collision_value_count
            if non_collision_value_count
            else None
        ),
        "absoluteStandardizedTargetThresholds": {
            str(threshold): {
                "valueCount": accumulator["valueCount"],
                "valueFraction": accumulator["valueCount"] / value_count,
                "modelSquaredErrorFraction": (
                    accumulator["modelSquaredError"] / total_squared_error
                    if total_squared_error
                    else None
                ),
                "zeroBaselineSquaredErrorFraction": (
                    accumulator["targetSquared"] / total_target_squared
                    if total_target_squared
                    else None
                ),
            }
            for threshold, accumulator in threshold_accumulators.items()
        },
        "perFrame": per_frame,
    }


def evaluate_standardized_mse(
    model: hybrid.HybridReferenceModel,
    indexes: list[hybrid.TeacherFrameIndex],
    statistics: hybrid.TrainingStatistics,
    *,
    device: torch.device | str,
    dynamic_grid_cache: dict[tuple[str, str, int], torch.Tensor] | None = None,
    wall_grid_cache: dict[Path, torch.Tensor] | None = None,
    training_frame_cache: dict[tuple[str, str, int], hybrid.TrainingFrameData]
    | None = None,
    prefetch_frames: int = 0,
    validated_indexes: bool = False,
) -> float | None:
    diagnostics = evaluate_standardized_mse_diagnostics(
        model,
        indexes,
        statistics,
        device=device,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=prefetch_frames,
        validated_indexes=validated_indexes,
    )
    return diagnostics["frameMeanStandardizedMse"] if diagnostics else None


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def main() -> None:
    args = parse_arguments()
    if args.epochs <= 0:
        raise hybrid.ContractError("--epochs must be positive.")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
    ):
        raise hybrid.ContractError("Invalid optimizer parameters.")
    if args.prefetch_frames < 0:
        raise hybrid.ContractError("--prefetch-frames must be non-negative.")
    if (
        not math.isfinite(args.progress_interval_seconds)
        or args.progress_interval_seconds <= 0.0
    ):
        raise hybrid.ContractError(
            "--progress-interval-seconds must be finite and positive."
        )
    if (
        not math.isfinite(args.dynamic_grid_cache_gib)
        or args.dynamic_grid_cache_gib < 0.0
        or not math.isfinite(args.training_frame_cache_gib)
        or args.training_frame_cache_gib < 0.0
    ):
        raise hybrid.ContractError("Training cache sizes must be non-negative.")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise hybrid.ContractError(
            "--output must be a new directory or an existing empty directory."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    all_indexes = load_dataset(tuple(args.teacher))
    indexes = select_frame_subset(all_indexes, args.frame_subset_count)
    checkpoint_epochs = tuple(sorted(set(args.checkpoint_epochs)))
    if any(epoch <= 0 or epoch > args.epochs for epoch in checkpoint_epochs):
        raise hybrid.ContractError(
            "--checkpoint-epochs must be positive and no greater than --epochs."
        )
    fractions = tuple(float(value) for value in args.split_fractions)
    split = hybrid.split_trajectories(indexes, fractions=fractions, seed=args.seed)
    counts = _split_counts(indexes, split)
    if counts["training"]["frames"] == 0:
        raise hybrid.ContractError("Trajectory split produced no training frames.")
    for name, fraction in zip(("training", "validation", "test"), fractions):
        if fraction > 0.0 and counts[name]["trajectories"] == 0:
            raise hybrid.ContractError(
                f"Requested non-empty {name} split but too few trajectories were supplied."
            )

    print(
        json.dumps(
            {
                "availableFrames": len(all_indexes),
                "selectedFrameTicks": [index.macro_step_index for index in indexes],
                "split": counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dynamic_grid_cache: dict[tuple[str, str, int], torch.Tensor] = {}
    wall_grid_cache: dict[Path, torch.Tensor] = {}
    training_frame_cache: dict[tuple[str, str, int], hybrid.TrainingFrameData] = {}
    cache_max_bytes = int(args.dynamic_grid_cache_gib * (1024**3))
    frame_cache_max_bytes = int(args.training_frame_cache_gib * (1024**3))
    preprocessing_device = args.preprocessing_device or args.device
    base_statistics = hybrid.compute_base_training_statistics_streaming(
        indexes,
        split,
        device=preprocessing_device,
        dynamic_grid_cache=dynamic_grid_cache,
        dynamic_grid_cache_max_bytes=cache_max_bytes,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        training_frame_cache_max_bytes=frame_cache_max_bytes,
        prefetch_frames=args.prefetch_frames,
    )
    cached_bytes = sum(
        value.numel() * value.element_size() for value in dynamic_grid_cache.values()
    )
    cached_frame_bytes = sum(value.nbytes for value in training_frame_cache.values())
    cached_dynamic_frame_count = len(dynamic_grid_cache)
    cached_training_frame_count = len(training_frame_cache)
    cached_wall_trajectory_count = len(wall_grid_cache)
    print(
        json.dumps(
            {
                "cachedDynamicGridFrames": cached_dynamic_frame_count,
                "cachedDynamicGridGiB": cached_bytes / (1024**3),
                "cachedFixedWallTrajectories": cached_wall_trajectory_count,
                "cachedTrainingFrames": cached_training_frame_count,
                "cachedTrainingFrameGiB": cached_frame_bytes / (1024**3),
                "preprocessingDevice": preprocessing_device,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    indexes_by_split = {
        name: [
            index
            for index in indexes
            if index.valid_grid_support and split.assignment(index) == name
        ]
        for name in ("training", "validation", "test")
    }
    model = hybrid.HybridReferenceModel(
        condition_count=len(indexes[0].condition_names),
        model_profile=args.model_profile,
    ).to(dtype=torch.float32)
    resumed_checkpoint = (
        _load_checkpoint(args.resume_checkpoint, model)
        if args.resume_checkpoint is not None
        else None
    )
    training_device = torch.device(args.device)
    if training_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(training_device)
    progress_reporter = EpochProgressReporter(
        args.epochs, args.progress_interval_seconds
    )

    def report_epoch(epoch: int, loss: float, elapsed: float) -> None:
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "trainingStandardizedMse": loss,
                    "epochSeconds": elapsed,
                }
            ),
            flush=True,
        )
        if epoch in checkpoint_epochs:
            _write_checkpoint(
                args.output / f"checkpoint-epoch-{epoch:04d}.pt",
                model,
                epoch=epoch,
                training_loss=loss,
            )

    training_result = hybrid.train_reference_model_streaming(
        model,
        indexes_by_split["training"],
        base_statistics,
        split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        epoch_callback=report_epoch,
        progress_callback=progress_reporter,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=args.prefetch_frames,
        validated_indexes=True,
    )
    training_peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(training_device))
        if training_device.type == "cuda"
        else None
    )
    statistics = dataclasses.replace(
        base_statistics, latent=training_result.latent_statistics
    )
    training_diagnostics = evaluate_standardized_mse_diagnostics(
        model,
        indexes_by_split["training"],
        statistics,
        device=args.device,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=args.prefetch_frames,
        validated_indexes=True,
    )
    validation_mse = evaluate_standardized_mse(
        model,
        indexes_by_split["validation"],
        statistics,
        device=args.device,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=args.prefetch_frames,
        validated_indexes=True,
    )
    test_mse = evaluate_standardized_mse(
        model,
        indexes_by_split["test"],
        statistics,
        device=args.device,
        dynamic_grid_cache=dynamic_grid_cache,
        wall_grid_cache=wall_grid_cache,
        training_frame_cache=training_frame_cache,
        prefetch_frames=args.prefetch_frames,
        validated_indexes=True,
    )

    dynamic_grid_cache.clear()
    wall_grid_cache.clear()
    training_frame_cache.clear()

    hybrid.export_reference_artifacts(
        args.output,
        model,
        statistics,
        indexes,
        split,
        collision_post_process=args.collision_post_process,
        validate_cuda=not args.skip_cuda_export_validation,
    )
    model.to(device="cpu")
    torch.save(
        {
            "modelProfile": model.model_profile,
            "architecture": hybrid.model_architecture(model),
            "modelState": model.state_dict(),
        },
        args.output / "model-state.pt",
    )
    export_validation = json.loads(
        (args.output / "export-validation.json").read_text(encoding="ascii")
    )
    architecture = hybrid.model_architecture(model)
    metrics = {
        "teacherFiles": [str(path.resolve()) for path in args.teacher],
        "availableFrames": len(all_indexes),
        "selectedFrameTicks": [index.macro_step_index for index in indexes],
        "frameSubsetCount": args.frame_subset_count,
        "checkpointEpochs": list(checkpoint_epochs),
        "epochs": args.epochs,
        "modelProfile": model.model_profile,
        "architecture": architecture,
        "conditionWidth": len(indexes[0].condition_names),
        "aiDeltaTime": indexes[0].ai_delta_time,
        "contractVersion": hybrid.CONTRACT["contractPackage"]["version"],
        "teacherSchemaVersion": hybrid.TEACHER_SCHEMA_VERSION,
        "modelBundleSchemaVersion": hybrid.MODEL_BUNDLE_SCHEMA_VERSION,
        "wallRasterizationAlgorithm": hybrid.RASTERIZATION_ALGORITHM_VERSION,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "optimizer": "AdamW",
        "learningRateSchedule": "constant",
        "batchSizeFrames": 1,
        "seed": args.seed,
        "trainingDevice": args.device,
        "preprocessingDevice": preprocessing_device,
        "prefetchFrames": args.prefetch_frames,
        "progressIntervalSeconds": args.progress_interval_seconds,
        "workerConfiguration": {
            "hdf5PrefetchThreadCount": int(args.prefetch_frames > 0),
            "prefetchFrames": args.prefetch_frames,
        },
        "dataOrder": "sorted-frame-key-then-seeded-epoch-shuffle",
        "dynamicGridCacheGiB": cached_bytes / (1024**3),
        "dynamicGridCachedFrames": cached_dynamic_frame_count,
        "trainingFrameCacheGiB": cached_frame_bytes / (1024**3),
        "trainingFrameCachedFrames": cached_training_frame_count,
        "splitFractions": list(fractions),
        "split": counts,
        "epochTrainingStandardizedMse": list(training_result.epoch_losses),
        "epochSeconds": list(training_result.epoch_seconds),
        "averageEpochSeconds": sum(training_result.epoch_seconds)
        / len(training_result.epoch_seconds),
        "peakGpuMemoryBytes": training_peak_gpu_memory,
        "resumeCheckpoint": (
            str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None
            else None
        ),
        "resumedCheckpointEpoch": (
            resumed_checkpoint.get("epoch") if resumed_checkpoint is not None else None
        ),
        "finalTrainingStandardizedMse": (
            training_diagnostics["frameMeanStandardizedMse"]
            if training_diagnostics
            else None
        ),
        "trainingDiagnostics": training_diagnostics,
        "validationStandardizedMse": validation_mse,
        "testStandardizedMse": test_mse,
        "inferenceExportSucceeded": True,
        "exportValidation": export_validation,
    }
    _write_metrics(args.output / "training-metrics.json", metrics)
    print(json.dumps({"artifacts": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
