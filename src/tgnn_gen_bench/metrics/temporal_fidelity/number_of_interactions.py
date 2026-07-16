from __future__ import annotations

# import packages

import torch

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.temporal_fidelity._interaction_helpers import snapshots

# category

CATEGORY = MetricCategory.TEMPORAL_FIDELITY

# number of interactions metric


class NumberOfInteractions(Metric[Graph, list[int]]):
    """Number of interactions occurring in each snapshot."""

    name = "number_of_interactions"
    category = CATEGORY

    def __init__(self, snapshot: str | TimeDeltaDG | None = None) -> None:
        self.snapshot = snapshot

    def compute(self, graph: Graph) -> list[int]:
        index, total = snapshots(graph, self.snapshot)
        return torch.bincount(index, minlength=total).tolist()


__all__ = ["NumberOfInteractions"]
