from __future__ import annotations

# import packages

import torch

# import modules

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.global_metrics._temporal_paths import temporal_reachability_counts

# category

CATEGORY = MetricCategory.GLOBAL_METRICS

# temporal reachability metric


class TemporalReachability(Metric[Graph, list[float]]):
    name = "temporal_reachability"
    category = CATEGORY

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def compute(self, graph: Graph) -> list[float]:
        num_nodes = int(graph.num_nodes)
        if num_nodes <= 1:
            return [0.0] * num_nodes

        counts = temporal_reachability_counts(graph, strict=self.strict)
        return (counts / float(num_nodes - 1)).tolist()


__all__ = ["TemporalReachability"]
