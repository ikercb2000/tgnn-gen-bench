from __future__ import annotations

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_graphs
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory

# category

CATEGORY = MetricCategory.STRUCTURAL_METRICS

# temporal degree distribution metric


class TemporalDegreeDistribution(Metric[Graph, list[int]]):
    name = "temporal_degree_distribution"
    category = CATEGORY

    def __init__(self, snapshot: str | TimeDeltaDG | None = None) -> None:
        self.snapshot = snapshot

    def compute(self, graph: Graph) -> list[int]:
        values: list[int] = []
        for snapshot_graph in build_snapshot_graphs(graph, snapshot=self.snapshot):
            values.extend(int(degree) for _, degree in snapshot_graph.degree())
        return values


__all__ = ["TemporalDegreeDistribution"]
