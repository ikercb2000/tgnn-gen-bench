from __future__ import annotations

# import packages

import torch
from torch import Tensor

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph

# snapshot helpers


def snapshots(
    graph: Graph,
    snapshot: str | TimeDeltaDG | None,
) -> tuple[Tensor, int]:
    """Assign every edge event to a snapshot on the graph's time grid."""
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


def encode(high: Tensor, low: Tensor, base: int) -> Tensor:
    """Pack two non-negative integer tensors into one, as high * base + low."""
    limit = torch.iinfo(torch.int64).max
    if int(high.max()) > (limit - int(low.max())) // max(base, 1):
        raise OverflowError(
            "the graph is too large to pack (node pair, snapshot) into int64"
        )
    return high * base + low


def pair_codes(graph: Graph, directed: bool) -> Tensor:
    """Encode each edge event's node pair as one integer."""
    src, dst = graph.edge_src.long(), graph.edge_dst.long()
    if not directed:
        src, dst = torch.minimum(src, dst), torch.maximum(src, dst)
    return src * int(graph.num_nodes) + dst


def contact_runs(
    graph: Graph,
    snapshot: str | TimeDeltaDG | None,
    directed: bool,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Collapse events into per-pair runs of consecutive occupied snapshots."""
    index, total = snapshots(graph, snapshot)
    codes = pair_codes(graph, directed)

    active = torch.unique(encode(codes, index, total))
    if active.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=index.device)
        return empty, empty, empty, total

    pair = active.div(total, rounding_mode="floor")
    when = active % total

    starts_run = torch.ones_like(when, dtype=torch.bool)
    starts_run[1:] = (pair[1:] != pair[:-1]) | ((when[1:] - when[:-1]) > 1)

    run_id = torch.cumsum(starts_run.long(), dim=0) - 1
    lengths = torch.bincount(run_id)
    return when[starts_run], lengths, pair[starts_run], total


__all__ = ["contact_runs", "encode", "pair_codes", "snapshots"]
