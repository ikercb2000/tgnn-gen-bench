from __future__ import annotations

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_adjacency
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.dynamics_metrics._random_walk_helpers import (
    entropy_signature,
)

# category

CATEGORY = MetricCategory.DYNAMICS_METRICS

# random walk entropy metric


class RandomWalkEntropy(Metric[Graph, list[float]]):
    name = "random_walk_entropy"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        n_samples: int = 100,
        n_walks: int = 2000,
        horizon: int = 14,
        stay_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.snapshot = snapshot
        self.n_samples = n_samples
        self.n_walks = n_walks
        self.horizon = horizon
        self.stay_probability = stay_probability
        self.seed = seed

    def compute(self, graph: Graph) -> list[float]:
        if int(graph.num_nodes) <= 1:
            return [0.0] * min(self.horizon, max(int(graph.num_nodes), 1))

        snapshots = build_snapshot_adjacency(graph, snapshot=self.snapshot)
        effective_horizon = min(self.horizon, len(snapshots))
        if effective_horizon == 0:
            return []

        curves = entropy_signature(
            snapshots,
            num_nodes=int(graph.num_nodes),
            n_samples=self.n_samples,
            n_walks=self.n_walks,
            horizon=effective_horizon,
            stay_probability=self.stay_probability,
            seed=self.seed,
        )
        return curves.mean(axis=0).tolist()


__all__ = ["RandomWalkEntropy"]
