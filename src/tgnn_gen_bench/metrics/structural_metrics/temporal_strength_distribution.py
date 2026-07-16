from __future__ import annotations

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_graphs
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory

# category

CATEGORY = MetricCategory.STRUCTURAL_METRICS

# temporal strength distribution metric


class TemporalStrengthDistribution(Metric[Graph, list[float]]):
    name = "temporal_strength_distribution"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        weight_index: int = 0,
    ) -> None:
        self.snapshot = snapshot
        self.weight_index = weight_index

    def compute(self, graph: Graph) -> list[float]:
        values: list[float] = []
        snapshots = build_snapshot_graphs(
            graph,
            snapshot=self.snapshot,
            weight_index=self.weight_index,
        )
        for snapshot_graph in snapshots:
            values.extend(float(strength) for _, strength in snapshot_graph.degree(weight="weight"))
        return values


__all__ = ["TemporalStrengthDistribution"]
