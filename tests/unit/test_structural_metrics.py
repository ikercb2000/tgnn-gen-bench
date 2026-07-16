# import packages

import pytest

# import modules

from tgnn_gen_bench.metrics import (
    CommunityEventDistribution,
    CommunityPersistence,
    DegreeAdjustedStrength,
    ModularityTrajectory,
    TemporalDegreeDistribution,
    TemporalStrengthDistribution,
)

from ._metric_graphs import weighted_graph

# temporal degree distribution tests


def test_temporal_degree_distribution_collects_snapshot_degrees() -> None:
    metric = TemporalDegreeDistribution(snapshot="s")

    assert metric.compute(weighted_graph()) == [1, 1, 0, 1, 2, 1]


# temporal strength distribution tests


def test_temporal_strength_distribution_collects_snapshot_strengths() -> None:
    metric = TemporalStrengthDistribution(snapshot="s")

    assert metric.compute(weighted_graph()) == pytest.approx([2.0, 2.0, 0.0, 1.0, 4.0, 3.0])


# degree adjusted strength tests


def test_degree_adjusted_strength_normalizes_snapshot_strengths() -> None:
    metric = DegreeAdjustedStrength(snapshot="s")

    assert metric.compute(weighted_graph()) == pytest.approx(
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        abs=1e-9,
    )


# community persistence tests


def test_community_persistence_tracks_mean_overlap_between_snapshots() -> None:
    metric = CommunityPersistence(snapshot="s")

    assert metric.compute(weighted_graph()) == pytest.approx([2.0 / 3.0])


# community event distribution tests


def test_community_event_distribution_returns_normalized_event_counts() -> None:
    metric = CommunityEventDistribution(snapshot="s")

    assert metric.compute(weighted_graph()) == pytest.approx([0.0, 0.0, 0.0, 1.0])


# modularity trajectory tests


def test_modularity_trajectory_returns_one_value_per_snapshot() -> None:
    metric = ModularityTrajectory(snapshot="s")

    assert metric.compute(weighted_graph()) == pytest.approx([0.0, 0.0])

