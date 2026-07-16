"""Radar figure builders."""

# import packages

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# import modules

from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.report._radar_config import (
    CATEGORY_COLOUR,
    CATEGORY_LABEL,
    CATEGORY_MARKER,
    CATEGORY_ORDER,
)
from tgnn_gen_bench.report._radar_helpers import label, polygon_area, radar_axes, ring
from tgnn_gen_bench.report.style import (
    DARK_GREY,
    DPI,
    ERROR_BAND_ALPHA,
    FULL_W,
    LIGHT_GREY,
    LINE_LW,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    PRIMARY,
    TWO3_W,
    save,
)

# figures


def by_category(
    similarity: Mapping[MetricCategory, float],
    out,
    tex: bool,
    dpi: int = DPI,
    *,
    context_label: str | None = None,
) -> dict:
    """Draw a category-level radar plot from similarity values."""
    categories = [category for category in CATEGORY_ORDER if category in similarity]
    mean = np.array([similarity[category] for category in categories])
    theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    area = polygon_area(mean)

    fig, ax = plt.subplots(
        figsize=(TWO3_W, TWO3_W * 0.92),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    ax.fill(ring(theta), ring(mean), color=PRIMARY, alpha=ERROR_BAND_ALPHA, linewidth=0)
    ax.plot(
        ring(theta),
        ring(mean),
        color=PRIMARY,
        lw=LINE_LW,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=MARKER_EDGE_COLOR,
        markeredgewidth=MARKER_EDGE_WIDTH,
    )
    radar_axes(
        ax,
        theta,
        [CATEGORY_LABEL.get(category, category) for category in categories],
        r"Category Similarity ($1-\mathrm{KS}$)" if tex else "Category Similarity (1 - KS)",
        area,
        subtitle=context_label,
    )
    save(fig, out, dpi=dpi)
    plt.close(fig)
    return {
        "axis_order": categories,
        "mean_similarity": mean.tolist(),
        "area": area,
        "context_label": context_label,
    }


def by_metric(
    similarity: Mapping[str, float],
    spread: Mapping[str, float],
    grouped: Mapping[MetricCategory, Sequence[str]],
    out,
    tex: bool,
    dpi: int = DPI,
    band: bool = True,
    *,
    context_label: str | None = None,
) -> dict:
    """Draw a metric-level radar plot from similarity values."""
    names = [name for category in grouped for name in grouped[category]]
    categories = [category for category in grouped for _ in grouped[category]]
    mean = np.array([similarity[name] for name in names])
    std = np.array([spread[name] for name in names])
    theta = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    area = polygon_area(mean)

    fig, ax = plt.subplots(
        figsize=(FULL_W, FULL_W * 0.62),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    ax.fill(ring(theta), ring(mean), color=LIGHT_GREY, alpha=ERROR_BAND_ALPHA, linewidth=0)
    ax.plot(ring(theta), ring(mean), color=DARK_GREY, lw=LINE_LW, zorder=2)
    if band:
        ax.fill_between(
            ring(theta),
            ring(np.clip(mean - std, 0, 1)),
            ring(np.clip(mean + std, 0, 1)),
            color=LIGHT_GREY,
            alpha=ERROR_BAND_ALPHA,
            linewidth=0,
        )
    for category in grouped:
        pick = [index for index, item in enumerate(categories) if item == category]
        ax.plot(
            theta[pick],
            mean[pick],
            linestyle="none",
            marker=CATEGORY_MARKER.get(category, "o"),
            markersize=MARKER_SIZE,
            color=CATEGORY_COLOUR.get(category, PRIMARY),
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )

    radar_axes(
        ax,
        theta,
        [label(name) for name in names],
        r"Metric Similarity ($1-\mathrm{KS}$)" if tex else "Metric Similarity (1 - KS)",
        area,
        subtitle=context_label,
    )
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=CATEGORY_MARKER.get(category, "o"),
            color=CATEGORY_COLOUR.get(category, PRIMARY),
            markersize=MARKER_SIZE,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=CATEGORY_LABEL.get(category, category).replace("\n", " "),
        )
        for category in grouped
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.34, 0.5))
    save(fig, out, dpi=dpi)
    plt.close(fig)
    return {
        "axis_order": names,
        "categories": categories,
        "mean_similarity": mean.tolist(),
        "std": std.tolist(),
        "area": area,
        "context_label": context_label,
    }


__all__ = ["by_category", "by_metric"]
