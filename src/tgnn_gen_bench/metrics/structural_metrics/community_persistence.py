from __future__ import annotations

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import build_snapshot_graphs
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.structural_metrics._community import community_transition_statistics

# category

CATEGORY = MetricCategory.STRUCTURAL_METRICS

# community persistence metric


class CommunityPersistence(Metric[Graph, list[float]]):
    """Measure how similar communities remain across snapshots."""

    name = "community_persistence"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        weight_index: int = 0,
        overlap_threshold: float = 0.3,
        resolution: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Configure snapshotting and community matching settings."""
        self.snapshot = snapshot
        self.weight_index = weight_index
        self.overlap_threshold = overlap_threshold
        self.resolution = resolution
        self.seed = seed

    def compute(self, graph: Graph) -> list[float]:
        """Compute mean community persistence per transition."""
        snapshots = build_snapshot_graphs(
            graph,
            snapshot=self.snapshot,
            weight_index=self.weight_index,
        )
        if len(snapshots) < 2:
            return []

        persistence: list[float] = []
        for graph_t, graph_t1 in zip(snapshots[:-1], snapshots[1:]):
            statistics = community_transition_statistics(
                graph_t,
                graph_t1,
                overlap_threshold=self.overlap_threshold,
                resolution=self.resolution,
                seed=self.seed,
            )
            persistence.append(float(statistics["mean_persistence"]))
        return persistence


__all__ = ["CommunityPersistence"]
