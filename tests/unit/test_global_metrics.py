# import packages

import pytest

# import modules

from tgnn_gen_bench.metrics import (
    TemporalClosenessCentrality,
    TemporalDegreeCentrality,
    TemporalReachability,
    WindowedReachability,
)

from ._metric_graphs import path_graph

# temporal reachability tests


def test_temporal_reachability_returns_per_node_reachability_ratio() -> None:
    metric = TemporalReachability()

    assert metric.compute(path_graph()) == pytest.approx([1.0, 0.5, 0.0])


# windowed reachability tests


def test_windowed_reachability_returns_ratio_per_snapshot_window() -> None:
    metric = WindowedReachability(snapshot="s")

    assert metric.compute(path_graph()) == pytest.approx([1.0 / 6.0, 1.0 / 6.0])


# temporal degree centrality tests


def test_temporal_degree_centrality_returns_endpoint_share_per_node() -> None:
    metric = TemporalDegreeCentrality()

    assert metric.compute(path_graph()) == pytest.approx([0.25, 0.5, 0.25])


# temporal closeness centrality tests


def test_temporal_closeness_centrality_returns_time_respecting_closeness() -> None:
    metric = TemporalClosenessCentrality()

    assert metric.compute(path_graph()) == pytest.approx([2.0 / 3.0, 1.0, 0.0])
