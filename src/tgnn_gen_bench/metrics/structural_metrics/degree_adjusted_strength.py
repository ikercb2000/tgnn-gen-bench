from __future__ import annotations

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_graphs
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.structural_metrics._degree_adjusted_strength_helpers import (
    snapshot_degree_adjusted_strengths,
)

# category

CATEGORY = MetricCategory.STRUCTURAL_METRICS

# degree adjusted strength metric


class DegreeAdjustedStrength(Metric[Graph, list[float]]):
    """Measure strengths after adjusting for node degree."""

    name = "degree_adjusted_strength"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        weight_index: int = 0,
        eps: float = 1e-12,
    ) -> None:
        """Configure snapshotting and numerical stabilization."""
        self.snapshot = snapshot
        self.weight_index = weight_index
        self.eps = eps

    def compute(self, graph: Graph) -> list[float]:
        """Compute degree-adjusted strengths across all snapshots."""
        values: list[float] = []
        snapshots = build_snapshot_graphs(
            graph,
            snapshot=self.snapshot,
            weight_index=self.weight_index,
        )
        for snapshot_graph in snapshots:
            values.extend(
                snapshot_degree_adjusted_strengths(snapshot_graph, eps=self.eps).tolist()
            )
        return values


__all__ = ["DegreeAdjustedStrength"]
