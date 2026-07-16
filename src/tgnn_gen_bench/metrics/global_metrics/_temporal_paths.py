from __future__ import annotations

# import packages

import torch
from torch import Tensor

# import modules

from tgnn_gen_bench.graphs import Graph

# temporal path helpers


def temporal_reachability_counts(
    graph: Graph,
    *,
    edge_ids: Tensor | None = None,
    strict: bool = True,
) -> Tensor:
    """Count how many nodes each node can reach through time."""
    num_nodes = int(graph.num_nodes)
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.float64)

    src, dst, time = _selected_edges(graph, edge_ids)
    if time.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float64)

    order = torch.argsort(time, stable=True)
    src, dst, time = src[order], dst[order], time[order]

    reach = torch.eye(num_nodes, dtype=torch.bool)
    if strict:
        for group in _time_groups(time):
            snapshot = reach.clone()
            for event_id in group.tolist():
                source = int(src[event_id])
                target = int(dst[event_id])
                reach[:, target] |= snapshot[:, source]
    else:
        for source, target in zip(src.tolist(), dst.tolist()):
            reach[:, target] |= reach[:, source]

    return reach.sum(dim=1).double() - 1.0


def temporal_closeness_values(
    graph: Graph,
    *,
    edge_ids: Tensor | None = None,
    strict: bool = True,
) -> Tensor:
    """Compute temporal closeness values from time-respecting paths."""
    num_nodes = int(graph.num_nodes)
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.float64)

    src, dst, time = _selected_edges(graph, edge_ids)
    dist = torch.full((num_nodes, num_nodes), float("inf"), dtype=torch.float64)
    dist.fill_diagonal_(0.0)

    if time.numel() > 0:
        order = torch.argsort(time, stable=True)
        src, dst, time = src[order], dst[order], time[order]

        if strict:
            for group in _time_groups(time):
                snapshot = dist.clone()
                for event_id in group.tolist():
                    source = int(src[event_id])
                    target = int(dst[event_id])
                    dist[:, target] = torch.minimum(dist[:, target], snapshot[:, source] + 1.0)
        else:
            for source, target in zip(src.tolist(), dst.tolist()):
                dist[:, target] = torch.minimum(dist[:, target], dist[:, source] + 1.0)

    reachable = torch.isfinite(dist)
    reachable.fill_diagonal_(False)
    hop_sum = torch.where(reachable, dist, torch.zeros_like(dist)).sum(dim=1)
    reached = reachable.sum(dim=1).double()
    return torch.where(reached > 0.0, reached / hop_sum, torch.zeros(num_nodes, dtype=torch.float64))


def _selected_edges(graph: Graph, edge_ids: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """Select the requested edges or all graph edges."""
    if edge_ids is None:
        return graph.edge_src.long(), graph.edge_dst.long(), graph.edge_time.long()
    return (
        graph.edge_src[edge_ids].long(),
        graph.edge_dst[edge_ids].long(),
        graph.edge_time[edge_ids].long(),
    )


def _time_groups(time: Tensor) -> list[Tensor]:
    """Split sorted timestamps into equal-time groups."""
    is_new_group = torch.cat([time.new_ones(1, dtype=torch.bool), time[1:] != time[:-1]])
    boundaries = torch.nonzero(is_new_group).flatten()
    groups: list[Tensor] = []
    for group_index, start in enumerate(boundaries.tolist()):
        end = int(boundaries[group_index + 1]) if group_index + 1 < boundaries.numel() else int(time.numel())
        groups.append(torch.arange(start, end))
    return groups


__all__ = ["temporal_closeness_values", "temporal_reachability_counts"]
