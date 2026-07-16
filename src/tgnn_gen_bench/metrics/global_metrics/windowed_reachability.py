from __future__ import annotations

# import packages

import torch

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics._graph_utils import snapshot_event_groups
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.global_metrics._temporal_paths import temporal_reachability_counts

# category

CATEGORY = MetricCategory.GLOBAL_METRICS

# windowed reachability metric


class WindowedReachability(Metric[Graph, list[float]]):
    """Measure reachability independently inside each snapshot window."""

    name = "windowed_reachability"
    category = CATEGORY

    def __init__(self, snapshot: str | TimeDeltaDG | None = None, strict: bool = True) -> None:
        """Configure the snapshot scale and chaining rule."""
        self.snapshot = snapshot
        self.strict = strict

    def compute(self, graph: Graph) -> list[float]:
        """Compute the reachable-pair ratio for each snapshot."""
        groups, total = snapshot_event_groups(graph, snapshot=self.snapshot)
        if total == 0:
            return []

        num_nodes = int(graph.num_nodes)
        total_pairs = num_nodes * (num_nodes - 1)
        if total_pairs == 0:
            return [0.0] * total

        ratios: list[float] = []
        for edge_ids in groups:
            if edge_ids.numel() == 0:
                ratios.append(0.0)
                continue
            reachable_pairs = float(
                temporal_reachability_counts(
                    graph,
                    edge_ids=edge_ids,
                    strict=self.strict,
                ).sum().item()
            )
            ratios.append(reachable_pairs / float(total_pairs))
        return ratios


__all__ = ["WindowedReachability"]
