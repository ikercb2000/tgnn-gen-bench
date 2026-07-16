from __future__ import annotations

# import modules

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.global_metrics._temporal_paths import temporal_closeness_values

# category

CATEGORY = MetricCategory.GLOBAL_METRICS

# temporal closeness centrality metric


class TemporalClosenessCentrality(Metric[Graph, list[float]]):
    name = "temporal_closeness_centrality"
    category = CATEGORY

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def compute(self, graph: Graph) -> list[float]:
        return temporal_closeness_values(graph, strict=self.strict).tolist()


__all__ = ["TemporalClosenessCentrality"]
