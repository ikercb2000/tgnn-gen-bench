from __future__ import annotations

# import packages

import torch

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.metrics.temporal_fidelity._interaction_helpers import contact_runs

# category

CATEGORY = MetricCategory.TEMPORAL_FIDELITY

# new interactions metric


class NewInteractions(Metric[Graph, list[int]]):
    """Number of interactions starting in each snapshot."""

    name = "new_interactions"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        directed: bool = False,
    ) -> None:
        """Choose the snapshot scale and edge direction rule."""
        self.snapshot = snapshot
        self.directed = directed

    def compute(self, graph: Graph) -> list[int]:
        """Count contact runs that start in each snapshot."""
        when, _, _, total = contact_runs(graph, self.snapshot, self.directed)
        return torch.bincount(when, minlength=total).tolist()


__all__ = ["NewInteractions"]
