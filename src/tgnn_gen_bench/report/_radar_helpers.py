"""Radar plot helper functions."""

# import packages

from collections.abc import Mapping, Sequence

import numpy as np

# import modules

from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.report._radar_config import CATEGORY_ORDER
from tgnn_gen_bench.report.style import (
    DARK_GREY,
    FONT_TICK,
    GRID_ALPHA,
    GRID_LINESTYLE,
    GRID_LW,
    LIGHT_GREY,
)

# helpers


def group_by_category(
    scores: Mapping[str, Sequence[float]],
    category_of: Mapping[str, MetricCategory],
) -> dict[MetricCategory, list[str]]:
    """Group metric names by category in registration order."""
    grouped: dict[MetricCategory, list[str]] = {}
    for name in scores:
        grouped.setdefault(category_of[name], []).append(name)
    return {category: grouped[category] for category in CATEGORY_ORDER if category in grouped}


def polygon_area(values: np.ndarray) -> float:
    """Compute the normalized area enclosed by a radar polygon."""
    if len(values) < 3:
        return float("nan")
    return float(np.sum(values * np.roll(values, -1)) / len(values))


def ring(values: np.ndarray) -> np.ndarray:
    """Close a polar polygon by repeating its first value."""
    return np.concatenate([values, values[:1]])


def radar_axes(ax, theta, labels, title, area, subtitle: str | None = None) -> None:
    """Apply the shared axis styling for radar plots."""
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=FONT_TICK)
    ax.set_rlabel_position(45)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.spines["polar"].set_color(LIGHT_GREY)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_facecolor("#FBFBF8")
    ax.set_title(title, pad=18, color=DARK_GREY)
    if subtitle is not None:
        ax.text(
            0.5,
            1.10,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_TICK,
            color="#666666",
        )
    ax.text(
        0.5,
        1.03,
        f"Area {area:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONT_TICK,
        color="#666666",
    )


def label(name: str) -> str:
    """Convert a metric name into a compact axis label."""
    words = name.split("_")
    half = (len(words) + 1) // 2
    first = " ".join(words[:half]).capitalize()
    return f"{first}\n{' '.join(words[half:])}" if words[half:] else first


__all__ = [
    "group_by_category",
    "label",
    "polygon_area",
    "radar_axes",
    "ring",
]
