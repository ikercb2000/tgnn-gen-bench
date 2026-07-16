# import packages

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# import modules

from tgnn_gen_bench.metrics.categories import MetricCategory

# Metric results base class

@dataclass
class MetricResult:
    """Store the outcome of one metric comparison."""

    metric_name: str
    category: MetricCategory
    score: float | None
    real_value: Any
    generated_value: Any
    distance_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Aggregate metric results across several generated graphs."""

    distance_name: str | None
    results_by_graph: dict[str, list[MetricResult]]
    scores_by_metric: dict[str, list[float]]
    category_of: dict[str, MetricCategory]

    def metric_similarity(self) -> dict[str, float]:
        """Convert distance scores into similarity scores per metric."""
        return {
            metric_name: 1.0 - float(np.mean(scores))
            for metric_name, scores in self.scores_by_metric.items()
        }

    def metric_spread(self) -> dict[str, float]:
        """Measure score variability per metric."""
        return {
            metric_name: float(np.std(scores))
            for metric_name, scores in self.scores_by_metric.items()
        }

    def category_similarity(
        self,
        grouped: dict[MetricCategory, list[str]],
    ) -> dict[MetricCategory, float]:
        """Average similarity scores within each metric category."""
        similarity = self.metric_similarity()
        return {
            category: float(np.mean([similarity[metric_name] for metric_name in metric_names]))
            for category, metric_names in grouped.items()
        }
