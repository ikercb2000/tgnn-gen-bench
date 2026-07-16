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

# duration of contacts metric


class DurationOfContacts(Metric[Graph, list[float]]):
    """Mean duration, in snapshots, of the contact between each pair of nodes."""

    name = "duration_of_contacts"
    category = CATEGORY

    def __init__(
        self,
        snapshot: str | TimeDeltaDG | None = None,
        directed: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.directed = directed

    def compute(self, graph: Graph) -> list[float]:
        _, lengths, pair, _ = contact_runs(graph, self.snapshot, self.directed)
        if lengths.numel() == 0:
            return []

        _, inverse = torch.unique(pair, return_inverse=True)
        totals = torch.zeros(
            int(inverse.max()) + 1,
            dtype=torch.float64,
            device=lengths.device,
        )
        counts = torch.zeros_like(totals)
        totals.scatter_add_(0, inverse, lengths.double())
        counts.scatter_add_(0, inverse, torch.ones_like(lengths, dtype=torch.float64))
        return (totals / counts).tolist()


__all__ = ["DurationOfContacts"]
