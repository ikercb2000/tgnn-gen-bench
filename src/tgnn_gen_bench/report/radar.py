"""Radar figures over metric scores, by category and by metric.

The style guide advises a dot plot over a radar for comparing methods across
metrics, and its reasons apply here too: enclosed area grows with the square of
the scores, and the axis order is not determined by the data. These radars use
its escape clause — the axis order is fixed by the caller and recorded in the
sidecar, and the shape is one polygon rather than several overlaid. State the
axis order in the caption of any figure produced here.
"""

# import packages

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# import modules

from tgnn_gen_bench.report.style import (
    DPI,
    DARK_GREY,
    ERROR_BAND_ALPHA,
    FONT_TICK,
    FOURTH,
    FULL_W,
    GRID_ALPHA,
    GRID_LINESTYLE,
    GRID_LW,
    LIGHT_GREY,
    LINE_LW,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    PRIMARY,
    SECONDARY,
    TERTIARY,
    TWO3_W,
    save,
)

# categories

CATEGORY_ORDER = [
    "temporal_fidelity",
    "structural_fidelity",
    "dynamics",
    "downstream_tasks",
]

CATEGORY_LABEL = {
    "temporal_fidelity": "Temporal\nfidelity",
    "structural_fidelity": "Structural\nfidelity",
    "dynamics": "Dynamical\nprocesses",
    "downstream_tasks": "Downstream\ntask",
}

# Colour is role-positional; category is also carried by marker shape, so it is
# never encoded by colour alone.
CATEGORY_COLOUR = {
    "temporal_fidelity": PRIMARY,
    "structural_fidelity": SECONDARY,
    "dynamics": TERTIARY,
    "downstream_tasks": FOURTH,
}
CATEGORY_MARKER = {
    "temporal_fidelity": "o",
    "structural_fidelity": "s",
    "dynamics": "D",
    "downstream_tasks": "^",
}

# helpers


def group_by_category(
    scores: Mapping[str, Sequence[float]], category_of: Mapping[str, str]
) -> dict[str, list[str]]:
    """Metric names per category, categories present only.

    Metrics keep the order they were registered in; that declared order is what
    the caption refers to. Sorting here would silently substitute an
    alphabetical one.
    """
    grouped: dict[str, list[str]] = {}
    for name in scores:
        grouped.setdefault(category_of[name], []).append(name)
    return {c: grouped[c] for c in CATEGORY_ORDER if c in grouped}


def polygon_area(values: np.ndarray) -> float:
    """Enclosed area as a fraction of the unit polygon.

    Quadratic in the scores, so it exaggerates differences, and one collapsed
    axis drags down both triangles adjacent to it. Report it as a number rather
    than inviting the reader to compare areas by eye.
    """
    if len(values) < 3:
        return float("nan")
    return float(np.sum(values * np.roll(values, -1)) / len(values))


def _ring(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def _radar_axes(ax, theta, labels, title, area) -> None:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=FONT_TICK)
    ax.set_rlabel_position(45)
    # The radial labels cross a polygon edge at any angle; keep them readable.
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.spines["polar"].set_color(LIGHT_GREY)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title(f"{title}\narea {area:.2f}", pad=14)


# figures


def by_category(similarity: Mapping[str, float], out, tex: bool, dpi: int = DPI) -> dict:
    """One axis per category, in CATEGORY_ORDER."""
    cats = [c for c in CATEGORY_ORDER if c in similarity]
    mean = np.array([similarity[c] for c in cats])
    theta = np.linspace(0, 2 * np.pi, len(cats), endpoint=False)
    area = polygon_area(mean)

    # A radar cannot be square: the perimeter belongs to the axis labels.
    fig, ax = plt.subplots(figsize=(TWO3_W, TWO3_W * 0.92),
                           subplot_kw={"projection": "polar"},
                           constrained_layout=True)
    ax.fill(_ring(theta), _ring(mean), color=PRIMARY, alpha=ERROR_BAND_ALPHA,
            linewidth=0)
    ax.plot(_ring(theta), _ring(mean), color=PRIMARY, lw=LINE_LW, marker="o",
            markersize=MARKER_SIZE, markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH)
    _radar_axes(ax, theta, [CATEGORY_LABEL.get(c, c) for c in cats],
                r"By category ($1-\mathrm{KS}$)" if tex else "By category (1 - KS)",
                area)
    save(fig, out, dpi=dpi)
    plt.close(fig)
    return {"axis_order": cats, "mean_similarity": mean.tolist(), "area": area}


def by_metric(
    similarity: Mapping[str, float],
    spread: Mapping[str, float],
    grouped: Mapping[str, Sequence[str]],
    out,
    tex: bool,
    dpi: int = DPI,
    band: bool = True,
) -> dict:
    """One axis per metric, ordered by category and coloured by it."""
    names = [n for c in grouped for n in grouped[c]]
    cats = [c for c in grouped for _ in grouped[c]]
    mean = np.array([similarity[n] for n in names])
    std = np.array([spread[n] for n in names])
    theta = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    area = polygon_area(mean)

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.62),
                           subplot_kw={"projection": "polar"},
                           constrained_layout=True)
    # One neutral outline for the shape; category identity lives on the markers,
    # so no two fills overlap and no colour has to be read as a quantity.
    ax.fill(_ring(theta), _ring(mean), color=LIGHT_GREY, alpha=ERROR_BAND_ALPHA,
            linewidth=0)
    ax.plot(_ring(theta), _ring(mean), color=DARK_GREY, lw=LINE_LW, zorder=2)
    if band:
        ax.fill_between(_ring(theta), _ring(np.clip(mean - std, 0, 1)),
                        _ring(np.clip(mean + std, 0, 1)),
                        color=LIGHT_GREY, alpha=ERROR_BAND_ALPHA, linewidth=0)
    for c in grouped:
        pick = [k for k, cat in enumerate(cats) if cat == c]
        ax.plot(theta[pick], mean[pick], linestyle="none",
                marker=CATEGORY_MARKER.get(c, "o"), markersize=MARKER_SIZE,
                color=CATEGORY_COLOUR.get(c, PRIMARY),
                markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH,
                zorder=3)

    _radar_axes(ax, theta, [_label(n) for n in names],
                r"By metric ($1-\mathrm{KS}$)" if tex else "By metric (1 - KS)", area)

    handles = [
        Line2D([], [], linestyle="none", marker=CATEGORY_MARKER.get(c, "o"),
               color=CATEGORY_COLOUR.get(c, PRIMARY), markersize=MARKER_SIZE,
               markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH,
               label=CATEGORY_LABEL.get(c, c).replace("\n", " "))
        for c in grouped
    ]
    # The perimeter belongs to the axis labels, so the legend gets its own strip
    # clear of them, not a slot just outside the circle.
    ax.legend(handles=handles, frameon=False, loc="center left",
              bbox_to_anchor=(1.34, 0.5))
    save(fig, out, dpi=dpi)
    plt.close(fig)
    return {"axis_order": names, "categories": cats,
            "mean_similarity": mean.tolist(), "std": std.tolist(), "area": area}


def _label(name: str) -> str:
    """Metric name to axis label: snake_case over two lines."""
    words = name.split("_")
    half = (len(words) + 1) // 2
    first = " ".join(words[:half]).capitalize()
    return f"{first}\n{' '.join(words[half:])}" if words[half:] else first


__all__ = [
    "CATEGORY_COLOUR",
    "CATEGORY_LABEL",
    "CATEGORY_MARKER",
    "CATEGORY_ORDER",
    "by_category",
    "by_metric",
    "group_by_category",
    "polygon_area",
]
