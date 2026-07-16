from __future__ import annotations

# import packages

import torch

# import modules

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory

# category

CATEGORY = MetricCategory.GLOBAL_METRICS

# temporal degree centrality metric


class TemporalDegreeCentrality(Metric[Graph, list[float]]):
    name = "temporal_degree_centrality"
    category = CATEGORY

    def compute(self, graph: Graph) -> list[float]:
        num_nodes = int(graph.num_nodes)
        if num_nodes == 0:
            return []

        src = graph.edge_src.long()
        dst = graph.edge_dst.long()
        total_endpoints = 2 * int(src.numel())
        if total_endpoints == 0:
            return [0.0] * num_nodes

        out_degree = torch.bincount(src, minlength=num_nodes)
        in_degree = torch.bincount(dst, minlength=num_nodes)
        degree = (in_degree + out_degree).double()
        return (degree / float(total_endpoints)).tolist()


__all__ = ["TemporalDegreeCentrality"]
