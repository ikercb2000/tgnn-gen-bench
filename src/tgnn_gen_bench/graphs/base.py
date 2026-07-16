from __future__ import annotations

# import packages

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from tgm import DGraph, TimeDeltaDG
from tgm.data import DGData

# graph aliases

Node = int
Graph = DGraph
GraphData = DGData
TemporalGraph = DGraph
TemporalGraphData = DGData

# graph components

@dataclass(frozen=True, slots=True)
class TemporalEdge:
    source: Node
    target: Node
    timestamp: int
    attributes: Mapping[str, Any] | None = None

# import modules

from tgnn_gen_bench.graphs.builders import from_raw, from_temporal_edges

# public exports

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
