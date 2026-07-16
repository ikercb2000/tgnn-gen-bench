# import packages

from dataclasses import FrozenInstanceError

import pytest
import torch

# import modules

from tgnn_gen_bench.graphs import (
    DGData,
    DGraph,
    Graph,
    GraphData,
    Node,
    TemporalEdge,
    TemporalGraph,
    TemporalGraphData,
    from_raw,
    from_temporal_edges,
)
from tgnn_gen_bench.graphs import __all__ as graphs_public_exports
from tgnn_gen_bench.graphs import base as graphs_base
from tgnn_gen_bench.graphs import builders

# graph aliases tests

def test_graph_aliases_tgm_objects() -> None:
    assert Node is int
    assert Graph is DGraph
    assert TemporalGraph is DGraph
    assert GraphData is DGData
    assert TemporalGraphData is DGData

# graph component tests

def test_temporal_edge_is_frozen() -> None:
    edge = TemporalEdge(source=0, target=1, timestamp=2, attributes={"weight": 1.0})

    with pytest.raises(FrozenInstanceError):
        edge.source = 3

# public exports tests

def test_graphs_package_exports_expected_public_api() -> None:
    assert set(graphs_public_exports) == {
        "DGData",
        "DGraph",
        "Graph",
        "GraphData",
        "Node",
        "TemporalEdge",
        "TemporalGraph",
        "TemporalGraphData",
        "TimeDeltaDG",
        "from_raw",
        "from_temporal_edges",
    }

# compatibility imports tests

def test_graphs_base_keeps_builder_compatibility_imports() -> None:
    assert graphs_base.from_raw is from_raw
    assert graphs_base.from_temporal_edges is from_temporal_edges

# from_raw builder tests

def test_from_raw_builds_tgm_dgraph() -> None:
    graph = from_raw(
        edge_time=torch.tensor([1, 3], dtype=torch.int64),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
        edge_x=torch.tensor([[1.5], [2.0]], dtype=torch.float32),
    )

    assert isinstance(graph, DGraph)
    assert graph.num_edge_events == 2
    assert graph.edge_x_dim == 1

# from_temporal_edges builder tests

def test_from_temporal_edges_builds_tgm_dgraph() -> None:
    graph = from_temporal_edges(
        [
            TemporalEdge(source=0, target=1, timestamp=1, attributes={"weight": 1.5}),
            TemporalEdge(source=1, target=2, timestamp=3, attributes={"weight": 2.0}),
        ]
    )

    assert isinstance(graph, DGraph)
    assert graph.num_edge_events == 2
    assert graph.edge_x_dim == 1


def test_from_temporal_edges_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="at least one temporal edge"):
        from_temporal_edges([])

# private helpers tests

def test_infer_feature_keys_collects_sorted_union() -> None:
    edges = [
        TemporalEdge(source=0, target=1, timestamp=1, attributes={"weight": 1.0, "count": 2}),
        TemporalEdge(source=1, target=2, timestamp=2, attributes={"alpha": 3.0}),
    ]

    assert builders._infer_feature_keys(edges) == ["alpha", "count", "weight"]


def test_build_edge_features_fills_missing_values_with_zero() -> None:
    edges = [
        TemporalEdge(source=0, target=1, timestamp=1, attributes={"weight": 1.5}),
        TemporalEdge(source=1, target=2, timestamp=2, attributes={"count": 2}),
    ]

    edge_x = builders._build_edge_features(edges, ["count", "weight"])

    assert torch.equal(
        edge_x,
        torch.tensor([[0.0, 1.5], [2.0, 0.0]], dtype=torch.float32),
    )


def test_build_edge_features_returns_none_without_feature_keys() -> None:
    edges = [TemporalEdge(source=0, target=1, timestamp=1, attributes={"weight": 1.5})]

    assert builders._build_edge_features(edges, []) is None


def test_build_edge_features_rejects_non_numeric_attributes() -> None:
    edges = [
        TemporalEdge(source=0, target=1, timestamp=1, attributes={"weight": "heavy"}),
    ]

    with pytest.raises(TypeError, match="must be numeric"):
        builders._build_edge_features(edges, ["weight"])
