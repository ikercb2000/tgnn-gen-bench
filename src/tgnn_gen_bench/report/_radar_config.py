"""Radar plot category constants."""

# import modules

from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.report.style import FOURTH, PRIMARY, SECONDARY, TERTIARY

# categories

CATEGORY_ORDER = [
    MetricCategory.TEMPORAL_FIDELITY,
    MetricCategory.GLOBAL_METRICS,
    MetricCategory.STRUCTURAL_METRICS,
    MetricCategory.DYNAMICS_METRICS,
    MetricCategory.DOWNSTREAM_TASKS,
    MetricCategory.DYNAMICS,
]

CATEGORY_LABEL = {
    MetricCategory.TEMPORAL_FIDELITY: "Temporal\nfidelity",
    MetricCategory.GLOBAL_METRICS: "Global\nmetrics",
    MetricCategory.STRUCTURAL_METRICS: "Structural\nmetrics",
    MetricCategory.DYNAMICS_METRICS: "Dynamical\nmetrics",
    MetricCategory.DYNAMICS: "Dynamical\nprocesses",
    MetricCategory.DOWNSTREAM_TASKS: "Downstream\ntask",
}

CATEGORY_COLOUR = {
    MetricCategory.TEMPORAL_FIDELITY: PRIMARY,
    MetricCategory.GLOBAL_METRICS: SECONDARY,
    MetricCategory.STRUCTURAL_METRICS: TERTIARY,
    MetricCategory.DYNAMICS_METRICS: FOURTH,
    MetricCategory.DYNAMICS: TERTIARY,
    MetricCategory.DOWNSTREAM_TASKS: PRIMARY,
}

CATEGORY_MARKER = {
    MetricCategory.TEMPORAL_FIDELITY: "o",
    MetricCategory.GLOBAL_METRICS: "s",
    MetricCategory.STRUCTURAL_METRICS: "D",
    MetricCategory.DYNAMICS_METRICS: "^",
    MetricCategory.DYNAMICS: "D",
    MetricCategory.DOWNSTREAM_TASKS: "P",
}

__all__ = [
    "CATEGORY_COLOUR",
    "CATEGORY_LABEL",
    "CATEGORY_MARKER",
    "CATEGORY_ORDER",
]
