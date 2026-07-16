# import packages

import pytest

# import modules

from tgnn_gen_bench.metrics import (
    DurationOfContacts,
    InteractingIndividuals,
    NewInteractions,
    NumberOfInteractions,
)

from ._metric_graphs import interaction_graph

# number of interactions tests


def test_number_of_interactions_counts_events_per_snapshot() -> None:
    metric = NumberOfInteractions(snapshot="s")

    assert metric.compute(interaction_graph()) == [1, 2, 0, 1]


# interacting individuals tests


def test_interacting_individuals_counts_distinct_nodes_per_snapshot() -> None:
    metric = InteractingIndividuals(snapshot="s")

    assert metric.compute(interaction_graph()) == [2, 3, 0, 2]


# new interactions tests


def test_new_interactions_counts_run_starts_per_snapshot() -> None:
    metric = NewInteractions(snapshot="s")

    assert metric.compute(interaction_graph()) == [1, 1, 0, 1]


# duration of contacts tests


def test_duration_of_contacts_returns_mean_run_length_per_pair() -> None:
    metric = DurationOfContacts(snapshot="s")

    assert sorted(metric.compute(interaction_graph())) == pytest.approx([1.0, 1.5])
