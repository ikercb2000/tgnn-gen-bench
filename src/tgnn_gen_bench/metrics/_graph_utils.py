from __future__ import annotations

# import packages

from collections.abc import Sequence

import networkx as nx
import torch
from torch import Tensor

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.graphs import Graph

# snapshot helpers


def edge_weights(graph: Graph, weight_index: int = 0) -> Tensor:
    """Extract one edge-weight channel or default to ones."""
    if graph.edge_x is None:
        return torch.ones(graph.num_edge_events, dtype=torch.float64, device=graph.device)

    weights = graph.edge_x
    if weights.ndim == 1:
        if weight_index != 0:
            raise IndexError(
                f"weight_index must be 0 when edge_x is one-dimensional, got {weight_index}"
            )
        return weights.double()

    if weight_index < 0 or weight_index >= weights.shape[1]:
        raise IndexError(
            f"weight_index {weight_index} is out of bounds for edge_x with "
            f"{weights.shape[1]} feature columns"
        )
    return weights[:, weight_index].double()


def snapshot_event_groups(
    graph: Graph,
    snapshot: str | TimeDeltaDG | None = None,
) -> tuple[list[Tensor], int]:
    """Group edge events by snapshot and return the total count."""
    if graph.num_edge_events == 0:
        return [], 0

    time = graph.edge_time.long()
    if graph.time_delta.is_event_ordered:
        if snapshot is not None:
            raise ValueError(
                "event-ordered graphs can only be grouped by their observed timestamps"
            )
        _, index = torch.unique(time, sorted=True, return_inverse=True)
        total = int(index.max().item()) + 1
        return _groups_from_index(index.long(), total), total

    step = 1
    if snapshot is not None:
        delta = TimeDeltaDG(snapshot) if isinstance(snapshot, str) else snapshot
        step = int(delta.convert(graph.time_delta))
        if step < 1:
            raise ValueError(
                f"snapshot {snapshot} is finer than the graph's time_delta "
                f"{graph.time_delta}; it cannot be resolved."
            )

    index = (time - graph.start_time).div(step, rounding_mode="floor").long()
    total = int((graph.end_time - graph.start_time) // step) + 1
    return _groups_from_index(index, total), total


def build_snapshot_graphs(
    graph: Graph,
    snapshot: str | TimeDeltaDG | None = None,
    *,
    directed: bool = False,
    weight_index: int = 0,
) -> list[nx.Graph]:
    """Build one NetworkX graph per snapshot."""
    groups, total = snapshot_event_groups(graph, snapshot)
    graph_cls = nx.DiGraph if directed else nx.Graph
    snapshots = [graph_cls() for _ in range(total)]

    nodes = range(int(graph.num_nodes))
    for snapshot_graph in snapshots:
        snapshot_graph.add_nodes_from(nodes)

    if not groups:
        return snapshots

    src = graph.edge_src.long().cpu()
    dst = graph.edge_dst.long().cpu()
    weights = edge_weights(graph, weight_index=weight_index).cpu()

    for snapshot_index, event_ids in enumerate(groups):
        if event_ids.numel() == 0:
            continue

        snapshot_graph = snapshots[snapshot_index]
        for event_id in event_ids.tolist():
            source = int(src[event_id])
            target = int(dst[event_id])
            if not directed and source > target:
                source, target = target, source

            weight = float(weights[event_id])
            if snapshot_graph.has_edge(source, target):
                snapshot_graph[source][target]["weight"] += weight
            else:
                snapshot_graph.add_edge(source, target, weight=weight)

    return snapshots


def build_snapshot_adjacency(
    graph: Graph,
    snapshot: str | TimeDeltaDG | None = None,
) -> list[dict[int, Tensor]]:
    """Build adjacency dictionaries for each undirected snapshot."""
    snapshots = build_snapshot_graphs(graph, snapshot=snapshot, directed=False)
    adjacency: list[dict[int, Tensor]] = []
    for snapshot_graph in snapshots:
        adjacency.append(
            {
                int(node): torch.tensor(
                    sorted(snapshot_graph.neighbors(node)),
                    dtype=torch.long,
                )
                for node in snapshot_graph.nodes
                if snapshot_graph.degree(node) > 0
            }
        )
    return adjacency


def _groups_from_index(index: Tensor, total: int) -> list[Tensor]:
    """Recover stable event groups from snapshot indices."""
    order = torch.argsort(index, stable=True)
    sorted_index = index[order]
    counts = torch.bincount(sorted_index, minlength=total)

    groups: list[Tensor] = []
    start = 0
    for count in counts.tolist():
        end = start + count
        groups.append(order[start:end])
        start = end
    return groups


__all__ = [
    "build_snapshot_adjacency",
    "build_snapshot_graphs",
    "edge_weights",
    "snapshot_event_groups",
]
