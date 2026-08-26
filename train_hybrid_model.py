"""Command-line training and export for the dense-grid hybrid surrogate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

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
        help="One or more published schema-v2 Teacher HDF5 trajectories.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
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
            "modelState": state,
        },
        path,
    )


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


@torch.no_grad()
def evaluate_standardized_mse_diagnostics(
    model: hybrid.HybridReferenceModel,
    indexes: list[hybrid.TeacherFrameIndex],
    statistics: hybrid.TrainingStatistics,
    *,
    device: torch.device | str,
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
    for index in indexes:
        frame = hybrid.prepare_indexed_frame(index, statistics)
        parameter = next(model.parameters())
        local = frame.local_features.to(device=parameter.device, dtype=parameter.dtype)
        target = frame.standardized_target.to(
            device=parameter.device, dtype=parameter.dtype
        )
        valid = frame.valid.to(device=parameter.device)
        prediction = model(
            frame.grid_input.to(device=parameter.device, dtype=parameter.dtype),
            frame.positions.to(device=parameter.device, dtype=parameter.dtype),
            local,
            frame.geometry,
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
) -> float | None:
    diagnostics = evaluate_standardized_mse_diagnostics(
        model, indexes, statistics, device=device
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
    base_statistics = hybrid.compute_base_training_statistics_streaming(indexes, split)
    indexes_by_split = {
        name: [
            index
            for index in indexes
            if index.valid_grid_support and split.assignment(index) == name
        ]
        for name in ("training", "validation", "test")
    }
    model = hybrid.HybridReferenceModel(
        condition_count=len(indexes[0].condition_names)
    ).to(dtype=torch.float32)
    args.output.mkdir(parents=True, exist_ok=True)

    def report_epoch(epoch: int, loss: float) -> None:
        print(
            json.dumps({"epoch": epoch, "trainingStandardizedMse": loss}),
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
    )
    statistics = dataclasses.replace(
        base_statistics, latent=training_result.latent_statistics
    )
    training_diagnostics = evaluate_standardized_mse_diagnostics(
        model,
        indexes_by_split["training"],
        statistics,
        device=args.device,
    )
    validation_mse = evaluate_standardized_mse(
        model,
        indexes_by_split["validation"],
        statistics,
        device=args.device,
    )
    test_mse = evaluate_standardized_mse(
        model, indexes_by_split["test"], statistics, device=args.device
    )

    hybrid.export_reference_artifacts(
        args.output,
        model,
        statistics,
        indexes,
        split,
        collision_post_process=args.collision_post_process,
        validate_cuda=not args.skip_cuda_export_validation,
    )
    torch.save(model.to(device="cpu").state_dict(), args.output / "model-state.pt")
    metrics = {
        "teacherFiles": [str(path.resolve()) for path in args.teacher],
        "availableFrames": len(all_indexes),
        "selectedFrameTicks": [index.macro_step_index for index in indexes],
        "frameSubsetCount": args.frame_subset_count,
        "checkpointEpochs": list(checkpoint_epochs),
        "epochs": args.epochs,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "seed": args.seed,
        "splitFractions": list(fractions),
        "split": counts,
        "epochTrainingStandardizedMse": list(training_result.epoch_losses),
        "finalTrainingStandardizedMse": (
            training_diagnostics["frameMeanStandardizedMse"]
            if training_diagnostics
            else None
        ),
        "trainingDiagnostics": training_diagnostics,
        "validationStandardizedMse": validation_mse,
        "testStandardizedMse": test_mse,
    }
    _write_metrics(args.output / "training-metrics.json", metrics)
    print(json.dumps({"artifacts": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
