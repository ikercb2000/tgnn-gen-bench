# import modules

from tgnn_gen_bench.graphs import TemporalEdge, from_temporal_edges

# graph builders


def path_graph():
    return from_temporal_edges(
        [
            TemporalEdge(source=0, target=1, timestamp=1),
            TemporalEdge(source=1, target=2, timestamp=2),
        ],
        time_delta="s",
    )


def weighted_graph():
    return from_temporal_edges(
        [
            TemporalEdge(source=0, target=1, timestamp=1, attributes={"weight": 2.0}),
            TemporalEdge(source=0, target=1, timestamp=2, attributes={"weight": 1.0}),
            TemporalEdge(source=1, target=2, timestamp=2, attributes={"weight": 3.0}),
        ],
        edge_feature_keys=["weight"],
        time_delta="s",
    )


def interaction_graph():
    return from_temporal_edges(
        [
            TemporalEdge(source=0, target=1, timestamp=1),
            TemporalEdge(source=0, target=1, timestamp=2),
            TemporalEdge(source=1, target=2, timestamp=2),
            TemporalEdge(source=0, target=1, timestamp=4),
        ],
        time_delta="s",
    )
