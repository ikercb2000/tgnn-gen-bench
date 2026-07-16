from __future__ import annotations

# import packages

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor
from tgm import DGraph, TimeDeltaDG
from tgm.data import DGData

# 
Node = int
Graph = DGraph
GraphData = DGData
TemporalGraph = DGraph
TemporalGraphData = DGData


@dataclass(frozen=True, slots=True)
class TemporalEdge:
    source: Node
    target: Node
    timestamp: int
    attributes: Mapping[str, Any] | None = None


def from_raw(
    edge_time: Tensor,
    edge_index: Tensor,
    edge_x: Tensor | None = None,
    node_x_time: Tensor | None = None,
    node_x_nids: Tensor | None = None,
    node_x: Tensor | None = None,
    node_y_time: Tensor | None = None,
    node_y_nids: Tensor | None = None,
    node_y: Tensor | None = None,
    static_node_x: Tensor | None = None,
    time_delta: TimeDeltaDG | str = "r",
    edge_type: Tensor | None = None,
    node_type: Tensor | None = None,
    device: str | torch.device = "cpu",
) -> DGraph:
    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        edge_x=edge_x,
        node_x_time=node_x_time,
        node_x_nids=node_x_nids,
        node_x=node_x,
        node_y_time=node_y_time,
        node_y_nids=node_y_nids,
        node_y=node_y,
        static_node_x=static_node_x,
        time_delta=time_delta,
        edge_type=edge_type,
        node_type=node_type,
    )
    return DGraph(data, device=device)


def from_temporal_edges(
    edges: Sequence[TemporalEdge],
    *,
    edge_feature_keys: Sequence[str] | None = None,
    time_delta: TimeDeltaDG | str = "r",
    device: str | torch.device = "cpu",
) -> DGraph:
    if not edges:
        raise ValueError("at least one temporal edge is required to build a DGraph")

    feature_keys = list(edge_feature_keys or _infer_feature_keys(edges))

    edge_time = torch.tensor([edge.timestamp for edge in edges], dtype=torch.int64)
    edge_index = torch.tensor(
        [[edge.source, edge.target] for edge in edges],
        dtype=torch.int32,
    )
    edge_x = _build_edge_features(edges, feature_keys)
    return from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        edge_x=edge_x,
        time_delta=time_delta,
        device=device,
    )


def _infer_feature_keys(edges: Sequence[TemporalEdge]) -> list[str]:
    keys: set[str] = set()
    for edge in edges:
        if edge.attributes is not None:
            keys.update(edge.attributes.keys())
    return sorted(keys)


def _build_edge_features(
    edges: Sequence[TemporalEdge],
    feature_keys: Sequence[str],
) -> Tensor | None:
    if not feature_keys:
        return None

    rows: list[list[float]] = []
    for edge in edges:
        attrs = edge.attributes or {}
        row: list[float] = []
        for key in feature_keys:
            value = attrs.get(key, 0.0)
            if not isinstance(value, int | float):
                raise TypeError(
                    f"edge attribute '{key}' must be numeric to be converted into TGM edge features"
                )
            row.append(float(value))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


__all__ = [
    "DGraph",
    "DGData",
    "Graph",
    "GraphData",
    "Node",
    "TemporalEdge",
    "TemporalGraph",
    "TemporalGraphData",
    "TimeDeltaDG",
    "from_raw",
    "from_temporal_edges",
]
