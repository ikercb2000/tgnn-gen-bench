# import packages

import math

import pytest

# import modules

from tgnn_gen_bench.metrics import (
    TemporalLinkPredictionLogLoss,
    TemporalLinkPredictionRocAuc,
)

from ._metric_graphs import downstream_graph, path_graph

# downstream task tests


def test_temporal_link_prediction_log_loss_returns_one_value_per_test_day() -> None:
    metric = TemporalLinkPredictionLogLoss(
        train_ratio=0.5,
        validation_ratio=0.25,
        negative_ratio=1,
        recent_window=2,
        c_grid=(0.1, 1.0),
        seed=7,
    )

    values = metric.compute(downstream_graph())

    assert len(values) == 2
    assert all(math.isfinite(value) for value in values)
    assert all(value >= 0.0 for value in values)


def test_temporal_link_prediction_roc_auc_returns_one_value_per_test_day() -> None:
    metric = TemporalLinkPredictionRocAuc(
        train_ratio=0.5,
        validation_ratio=0.25,
        negative_ratio=1,
        recent_window=2,
        c_grid=(0.1, 1.0),
        seed=7,
    )

    values = metric.compute(downstream_graph())

    assert len(values) == 2
    assert all(math.isfinite(value) for value in values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_temporal_link_prediction_requires_enough_snapshots() -> None:
    metric = TemporalLinkPredictionLogLoss()

    with pytest.raises(ValueError, match="Not enough snapshots"):
        metric.compute(path_graph())
