from tgnn_gen_bench.graphs import (
    DGData,
    DGraph,
    GraphData,
    TemporalEdge,
    TemporalGraph,
    from_temporal_edges,
)


def test_temporal_graph_aliases_tgm_dgraph() -> None:
    assert TemporalGraph is DGraph


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


def test_graph_data_aliases_tgm_dgdata() -> None:
    assert GraphData is DGData
