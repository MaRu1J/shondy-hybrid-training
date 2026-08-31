"""Plot the per-epoch training loss stored in a model artifact bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


METRICS_FILENAME = "training-metrics.json"
LOSS_KEY = "epochTrainingStandardizedMse"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an SVG loss curve from the training metrics in a ShonDy "
            "model artifact bundle."
        )
    )
    parser.add_argument(
        "artifacts",
        type=Path,
        help=(
            "Artifact directory containing training-metrics.json, or the metrics "
            "JSON file itself."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output SVG path (default: <artifacts>/training-loss.svg).",
    )
    parser.add_argument(
        "--title",
        default="Training Loss Curve",
        help="Title shown above the plot.",
    )
    return parser.parse_args()


def resolve_paths(artifacts: Path, output: Path | None) -> tuple[Path, Path]:
    metrics_path = artifacts / METRICS_FILENAME if artifacts.is_dir() else artifacts
    if output is None:
        output_path = metrics_path.parent / "training-loss.svg"
    else:
        output_path = output
    if output_path.suffix.lower() != ".svg":
        raise ValueError("--output must use the .svg extension.")
    return metrics_path, output_path


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="ascii"))
    except FileNotFoundError as exc:
        raise ValueError(f"Metrics file does not exist: {metrics_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid metrics JSON: {metrics_path}: {exc}") from exc

    if not isinstance(metrics, dict):
        raise ValueError("The metrics JSON root must be an object.")
    return metrics


def extract_losses(metrics: dict[str, Any]) -> list[float]:
    raw_losses = metrics.get(LOSS_KEY)
    if not isinstance(raw_losses, list) or not raw_losses:
        raise ValueError(f"Metrics field {LOSS_KEY!r} must be a non-empty array.")

    losses: list[float] = []
    for epoch, raw_loss in enumerate(raw_losses, start=1):
        if isinstance(raw_loss, bool) or not isinstance(raw_loss, (int, float)):
            raise ValueError(f"Loss at epoch {epoch} is not a number: {raw_loss!r}")
        loss = float(raw_loss)
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError(
                f"Loss at epoch {epoch} must be finite and non-negative: {raw_loss!r}"
            )
        losses.append(loss)
    return losses


def extract_checkpoint_epochs(metrics: dict[str, Any], epoch_count: int) -> list[int]:
    raw_epochs = metrics.get("checkpointEpochs", [])
    if not isinstance(raw_epochs, list):
        return []
    return sorted(
        {
            epoch
            for epoch in raw_epochs
            if isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and 1 <= epoch <= epoch_count
        }
    )


def nice_tick_step(span: float, target_ticks: int = 6) -> float:
    rough_step = span / target_ticks
    magnitude = 10.0 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1.0:
        multiplier = 1.0
    elif normalized <= 2.0:
        multiplier = 2.0
    elif normalized <= 5.0:
        multiplier = 5.0
    else:
        multiplier = 10.0
    return multiplier * magnitude


def axis_bounds(values: list[float]) -> tuple[float, float, float]:
    lower = min(values)
    upper = max(values)
    span = upper - lower
    if span == 0.0:
        span = max(abs(upper) * 0.1, 1.0)
    padding = span * 0.08
    tick_step = nice_tick_step(span + 2.0 * padding)
    axis_min = math.floor((lower - padding) / tick_step) * tick_step
    axis_max = math.ceil((upper + padding) / tick_step) * tick_step
    if axis_min == axis_max:
        axis_max += tick_step
    return axis_min, axis_max, tick_step


def format_tick(value: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step)))
    return f"{value:.{decimals}f}"


def render_svg(
    losses: list[float],
    checkpoint_epochs: list[int],
    *,
    title: str,
    source_name: str,
) -> str:
    width = 1200
    height = 720
    left = 112
    right = 55
    top = 112
    bottom = 92
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max, y_step = axis_bounds(losses)
    epoch_count = len(losses)

    def x_position(epoch: int) -> float:
        if epoch_count == 1:
            return left + plot_width / 2.0
        return left + (epoch - 1) * plot_width / (epoch_count - 1)

    def y_position(loss: float) -> float:
        return top + (y_max - loss) * plot_height / (y_max - y_min)

    x_step = max(1, math.ceil(epoch_count / 10))
    x_ticks = list(range(1, epoch_count + 1, x_step))
    if x_ticks[-1] != epoch_count:
        x_ticks.append(epoch_count)

    y_tick_count = int(round((y_max - y_min) / y_step))
    y_ticks = [y_min + index * y_step for index in range(y_tick_count + 1)]
    points = " ".join(
        f"{x_position(epoch):.2f},{y_position(loss):.2f}"
        for epoch, loss in enumerate(losses, start=1)
    )
    minimum_loss = min(losses)
    minimum_epoch = losses.index(minimum_loss) + 1
    final_loss = losses[-1]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        f"<title id=\"title\">{escape(title)}</title>",
        (
            '<desc id="description">Per-epoch standardized mean squared error '
            f"for {epoch_count} training epochs.</desc>"
        ),
        "<style>",
        "text { font-family: Inter, Arial, sans-serif; fill: #172033; }",
        ".grid { stroke: #dfe4ec; stroke-width: 1; }",
        ".axis { stroke: #596273; stroke-width: 1.5; }",
        ".tick { font-size: 15px; fill: #596273; }",
        ".label { font-size: 18px; font-weight: 600; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="48" font-size="28" font-weight="700">'
            f"{escape(title)}</text>"
        ),
        (
            f'<text x="{left}" y="76" font-size="15" fill="#697386">'
            f"{escape(source_name)}</text>"
        ),
    ]

    for tick in y_ticks:
        y = y_position(tick)
        lines.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        lines.append(
            f'<text class="tick" x="{left - 16}" y="{y + 5:.2f}" '
            f'text-anchor="end">{format_tick(tick, y_step)}</text>'
        )

    for epoch in x_ticks:
        x = x_position(epoch)
        lines.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{top + plot_height}"/>'
        )
        lines.append(
            f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 30}" '
            f'text-anchor="middle">{epoch}</text>'
        )

    lines.extend(
        [
            (
                f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
            ),
            (
                f'<line class="axis" x1="{left}" y1="{top}" '
                f'x2="{left}" y2="{top + plot_height}"/>'
            ),
            (
                f'<polyline points="{points}" fill="none" stroke="#176b87" '
                'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            ),
        ]
    )

    for epoch in checkpoint_epochs:
        x = x_position(epoch)
        y = y_position(losses[epoch - 1])
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#f59e0b" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )

    minimum_x = x_position(minimum_epoch)
    minimum_y = y_position(minimum_loss)
    lines.extend(
        [
            (
                f'<circle cx="{minimum_x:.2f}" cy="{minimum_y:.2f}" r="6" '
                'fill="#c23b53" stroke="#ffffff" stroke-width="2"/>'
            ),
            (
                f'<text x="{left + plot_width}" y="76" font-size="15" '
                'text-anchor="end" fill="#596273">'
                f"Final: {final_loss:.6g}  |  Min: {minimum_loss:.6g} "
                f"(epoch {minimum_epoch})</text>"
            ),
            (
                f'<text class="label" x="{left + plot_width / 2}" y="{height - 28}" '
                'text-anchor="middle">Epoch</text>'
            ),
            (
                f'<text class="label" x="30" y="{top + plot_height / 2}" '
                f'text-anchor="middle" transform="rotate(-90 30 '
                f'{top + plot_height / 2})">Standardized MSE</text>'
            ),
        ]
    )

    if checkpoint_epochs:
        legend_x = left + 12
        legend_y = top + 25
        lines.extend(
            [
                (
                    f'<circle cx="{legend_x}" cy="{legend_y}" r="4" '
                    'fill="#f59e0b"/>'
                ),
                (
                    f'<text x="{legend_x + 12}" y="{legend_y + 5}" '
                    'font-size="14" fill="#596273">Checkpoint</text>'
                ),
            ]
        )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_arguments()
    try:
        metrics_path, output_path = resolve_paths(args.artifacts, args.output)
        metrics = load_metrics(metrics_path)
        losses = extract_losses(metrics)
        checkpoint_epochs = extract_checkpoint_epochs(metrics, len(losses))
        svg = render_svg(
            losses,
            checkpoint_epochs,
            title=args.title,
            source_name=metrics_path.parent.name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(svg, encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(output_path.resolve())


if __name__ == "__main__":
    main()
