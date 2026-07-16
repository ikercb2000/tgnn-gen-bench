from __future__ import annotations

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_graphs
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.structural_metrics._community import snapshot_louvain_modularity

# category

CATEGORY = MetricCategory.STRUCTURAL_METRICS

# modularity trajectory metric


class ModularityTrajectory(Metric[Graph, list[float]]):
    name = "modularity_trajectory"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        weight_index: int = 0,
        resolution: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.snapshot = snapshot
        self.weight_index = weight_index
        self.resolution = resolution
        self.seed = seed

    def compute(self, graph: Graph) -> list[float]:
        values: list[float] = []
        snapshots = build_snapshot_graphs(
            graph,
            snapshot=self.snapshot,
            weight_index=self.weight_index,
        )
        for snapshot_graph in snapshots:
            modularity, _ = snapshot_louvain_modularity(
                snapshot_graph,
                resolution=self.resolution,
                seed=self.seed,
            )
            values.append(modularity)
        return values


__all__ = ["ModularityTrajectory"]
