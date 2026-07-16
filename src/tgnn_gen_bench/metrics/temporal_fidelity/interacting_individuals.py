from __future__ import annotations

# import packages

import torch

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.temporal_fidelity._interaction_helpers import (
    encode,
    snapshots,
)

# category

CATEGORY = MetricCategory.TEMPORAL_FIDELITY

# interacting individuals metric


class InteractingIndividuals(Metric[Graph, list[int]]):
    """Number of distinct individuals interacting in each snapshot."""

    name = "interacting_individuals"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        directed: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.directed = directed

    def compute(self, graph: Graph) -> list[int]:
        index, total = snapshots(graph, self.snapshot)
        nodes = torch.cat([graph.edge_src.long(), graph.edge_dst.long()])
        when = torch.cat([index, index])
        active = torch.unique(encode(when, nodes, int(graph.num_nodes)))
        snapshot_of = active.div(int(graph.num_nodes), rounding_mode="floor")
        return torch.bincount(snapshot_of, minlength=total).tolist()


__all__ = ["InteractingIndividuals"]
