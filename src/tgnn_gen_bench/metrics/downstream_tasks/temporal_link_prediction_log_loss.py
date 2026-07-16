from __future__ import annotations

# import packages

from collections.abc import Sequence

# import modules

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.downstream_tasks._temporal_link_prediction_helpers import (
    temporal_link_prediction_results,
)

# category

CATEGORY = MetricCategory.DOWNSTREAM_TASKS

# downstream task metric


class TemporalLinkPredictionLogLoss(Metric[Graph, list[float]]):
    """Measure per-day temporal link prediction log-loss."""

    name = "temporal_link_prediction_log_loss"
    category = CATEGORY

    def __init__(
        self,
        *,
        directed: bool = True,
        train_ratio: float = 0.60,
        validation_ratio: float = 0.20,
        negative_ratio: int = 1,
        recent_window: int = 7,
        c_grid: Sequence[float] = (0.1, 1.0, 10.0),
        seed: int = 42,
    ) -> None:
        """Configure the temporal link prediction baseline."""
        self.directed = directed
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.negative_ratio = negative_ratio
        self.recent_window = recent_window
        self.c_grid = tuple(c_grid)
        self.seed = seed

    def compute(self, graph: Graph) -> list[float]:
        """Compute the test-day log-loss profile for one graph."""
        results = temporal_link_prediction_results(
            graph,
            directed=self.directed,
            train_ratio=self.train_ratio,
            validation_ratio=self.validation_ratio,
            negative_ratio=self.negative_ratio,
            recent_window=self.recent_window,
            c_grid=self.c_grid,
            seed=self.seed,
        )
        per_day = results["test"]["per_day"]
        return [float(per_day[day]["log_loss"]) for day in sorted(per_day)]


__all__ = ["TemporalLinkPredictionLogLoss"]
