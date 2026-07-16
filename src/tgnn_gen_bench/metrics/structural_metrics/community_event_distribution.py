from __future__ import annotations

# import packages

import numpy as np

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_graphs
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.structural_metrics._community import (
    community_event_distribution,
    community_transition_statistics,
)

# category

CATEGORY = MetricCategory.STRUCTURAL_METRICS

# community event distribution metric


class CommunityEventDistribution(Metric[Graph, list[float]]):
    name = "community_event_distribution"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        weight_index: int = 0,
        overlap_threshold: float = 0.3,
        resolution: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.snapshot = snapshot
        self.weight_index = weight_index
        self.overlap_threshold = overlap_threshold
        self.resolution = resolution
        self.seed = seed

    def compute(self, graph: Graph) -> list[float]:
        snapshots = build_snapshot_graphs(
            graph,
            snapshot=self.snapshot,
            weight_index=self.weight_index,
        )
        if len(snapshots) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        counts = np.zeros(4, dtype=float)
        for graph_t, graph_t1 in zip(snapshots[:-1], snapshots[1:]):
            statistics = community_transition_statistics(
                graph_t,
                graph_t1,
                overlap_threshold=self.overlap_threshold,
                resolution=self.resolution,
                seed=self.seed,
            )
            counts += community_event_distribution(statistics)

        if counts.sum() == 0.0:
            return [0.0, 0.0, 0.0, 0.0]
        return (counts / counts.sum()).tolist()


__all__ = ["CommunityEventDistribution"]
