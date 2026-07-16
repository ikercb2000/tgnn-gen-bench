# import packages

import torch
from torch import Tensor

# import modules

from tgm import TimeDeltaDG
from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric

# category

CATEGORY = "temporal_fidelity"

# helpers


def _snapshots(graph: Graph, snapshot: str | TimeDeltaDG | None) -> tuple[Tensor, int]:
    """Assign every edge event to a snapshot on the graph's time grid.

    Returns the per-event snapshot index and the total number of snapshots
    spanning [start_time, end_time]. Snapshots with no events are counted, so
    downstream distributions carry their zeros.
    """
    if graph.time_delta.is_event_ordered:
        raise ValueError(
            "snapshot metrics need a time-ordered graph: the timestamps of an "
            "event-ordered DGraph are an ordering, not a time grid. Build the "
            "data with a time-ordered time_delta, e.g. TimeDeltaDG('s')."
        )

    step = 1
    if snapshot is not None:
        delta = TimeDeltaDG(snapshot) if isinstance(snapshot, str) else snapshot
        step = int(delta.convert(graph.time_delta))
        if step < 1:
            raise ValueError(
                f"snapshot {snapshot} is finer than the graph's time_delta "
                f"{graph.time_delta}; it cannot be resolved."
            )

    time = graph.edge_time
    index = (time - graph.start_time).div(step, rounding_mode="floor")
    total = int((graph.end_time - graph.start_time) // step) + 1
    return index.long(), total


def _encode(high: Tensor, low: Tensor, base: int) -> Tensor:
    """Pack two non-negative integer tensors into one, as high * base + low."""
    limit = torch.iinfo(torch.int64).max
    if int(high.max()) > (limit - int(low.max())) // max(base, 1):
        raise OverflowError(
            "the graph is too large to pack (node pair, snapshot) into int64"
        )
    return high * base + low


def _pair_codes(graph: Graph, directed: bool) -> Tensor:
    """Encode each edge event's node pair as one integer."""
    src, dst = graph.edge_src.long(), graph.edge_dst.long()
    if not directed:
        src, dst = torch.minimum(src, dst), torch.maximum(src, dst)
    return src * int(graph.num_nodes) + dst


def _contact_runs(
    graph: Graph, snapshot: str | TimeDeltaDG | None, directed: bool
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Collapse events into per-pair runs of consecutive occupied snapshots.

    A run is a maximal stretch of snapshots s, s+1, ... over which a pair stays
    in contact. Returns the run start snapshots, run lengths, the pair code of
    each run, and the total snapshot count.
    """
    index, total = _snapshots(graph, snapshot)
    codes = _pair_codes(graph, directed)

    # One entry per (pair, snapshot) the pair is active in, ordered by pair then
    # snapshot. Encoded into a single integer: torch.unique(dim=0) would sort
    # rows lexicographically on a slow path, this sorts a flat int64 tensor.
    active = torch.unique(_encode(codes, index, total))
    if active.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=index.device)
        return empty, empty, empty, total
    pair, when = active.div(total, rounding_mode="floor"), active % total

    # A run starts at the first row of a pair, or after a gap of more than one.
    starts_run = torch.ones_like(when, dtype=torch.bool)
    starts_run[1:] = (pair[1:] != pair[:-1]) | ((when[1:] - when[:-1]) > 1)

    run_id = torch.cumsum(starts_run.long(), dim=0) - 1
    lengths = torch.bincount(run_id)
    return when[starts_run], lengths, pair[starts_run], total


# metrics


class NumberOfInteractions(Metric):
    """Number of interactions occurring in each snapshot."""

    name = "number_of_interactions"
    category = CATEGORY

    def __init__(self, snapshot: str | TimeDeltaDG | None = None) -> None:
        self.snapshot = snapshot

    def compute(self, graph: Graph) -> list[int]:
        index, total = _snapshots(graph, self.snapshot)
        return torch.bincount(index, minlength=total).tolist()


class InteractingIndividuals(Metric):
    """Number of distinct individuals interacting in each snapshot."""

    name = "interacting_individuals"
    category = CATEGORY

    def __init__(
        self, snapshot: str | TimeDeltaDG | None = None, directed: bool = False
    ) -> None:
        self.snapshot = snapshot
        self.directed = directed

    def compute(self, graph: Graph) -> list[int]:
        index, total = _snapshots(graph, self.snapshot)
        nodes = torch.cat([graph.edge_src.long(), graph.edge_dst.long()])
        when = torch.cat([index, index])
        active = torch.unique(_encode(when, nodes, int(graph.num_nodes)))
        snapshot_of = active.div(int(graph.num_nodes), rounding_mode="floor")
        return torch.bincount(snapshot_of, minlength=total).tolist()


class NewInteractions(Metric):
    """Number of interactions starting in each snapshot.

    A pair starts an interaction in snapshot s when it is in contact at s and
    was not at s-1; every pair present in the first snapshot it appears in
    counts as new.
    """

    name = "new_interactions"
    category = CATEGORY

    def __init__(
        self, snapshot: str | TimeDeltaDG | None = None, directed: bool = False
    ) -> None:
        self.snapshot = snapshot
        self.directed = directed

    def compute(self, graph: Graph) -> list[int]:
        when, _, _, total = _contact_runs(graph, self.snapshot, self.directed)
        return torch.bincount(when, minlength=total).tolist()


class DurationOfContacts(Metric):
    """Mean duration, in snapshots, of the contact between each pair of nodes.

    Each pair contributes the mean length of its runs of consecutive snapshots
    in contact, so the distribution is over pairs rather than over snapshots.
    """

    name = "duration_of_contacts"
    category = CATEGORY

    def __init__(
        self, snapshot: str | TimeDeltaDG | None = None, directed: bool = False
    ) -> None:
        self.snapshot = snapshot
        self.directed = directed

    def compute(self, graph: Graph) -> list[float]:
        _, lengths, pair, _ = _contact_runs(graph, self.snapshot, self.directed)
        if lengths.numel() == 0:
            return []
        # Mean run length per pair, without materialising a dict.
        _, inverse = torch.unique(pair, return_inverse=True)
        totals = torch.zeros(int(inverse.max()) + 1, dtype=torch.float64,
                             device=lengths.device)
        counts = torch.zeros_like(totals)
        totals.scatter_add_(0, inverse, lengths.double())
        counts.scatter_add_(0, inverse, torch.ones_like(lengths, dtype=torch.float64))
        return (totals / counts).tolist()


__all__ = [
    "DurationOfContacts",
    "InteractingIndividuals",
    "NewInteractions",
    "NumberOfInteractions",
]
