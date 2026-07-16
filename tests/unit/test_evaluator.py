# import packages

import pytest

# import modules

from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation import EvaluationSummary, Evaluator
from tgnn_gen_bench.metrics import (
    MetricCategory,
    NumberOfInteractions,
    TemporalDegreeCentrality,
)

from ._metric_graphs import interaction_graph, path_graph

# evaluator pair tests


def test_evaluator_returns_metric_results_for_one_graph_pair() -> None:
    evaluator = Evaluator(
        metrics=[NumberOfInteractions(snapshot="s"), TemporalDegreeCentrality()],
        distance=KSDistance(),
    )

    results = evaluator.evaluate(path_graph(), path_graph())

    assert [result.metric_name for result in results] == [
        "number_of_interactions",
        "temporal_degree_centrality",
    ]
    assert all(result.score == pytest.approx(0.0) for result in results)
    assert results[0].category is MetricCategory.TEMPORAL_FIDELITY
    assert results[1].category is MetricCategory.GLOBAL_METRICS


# evaluator batch tests


def test_evaluator_evaluate_many_builds_summary_over_generated_graphs() -> None:
    evaluator = Evaluator(
        metrics=[NumberOfInteractions(snapshot="s"), TemporalDegreeCentrality()],
        distance=KSDistance(),
    )

    summary = evaluator.evaluate_many(
        path_graph(),
        [
            ("same", path_graph()),
            ("different", interaction_graph()),
        ],
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.distance_name == "ks"
    assert list(summary.results_by_graph) == ["same", "different"]
    assert summary.category_of == {
        "number_of_interactions": MetricCategory.TEMPORAL_FIDELITY,
        "temporal_degree_centrality": MetricCategory.GLOBAL_METRICS,
    }
    assert summary.scores_by_metric["number_of_interactions"][0] == pytest.approx(0.0)
    assert summary.scores_by_metric["temporal_degree_centrality"][0] == pytest.approx(0.0)


def test_evaluation_summary_computes_similarity_and_spread() -> None:
    evaluator = Evaluator(
        metrics=[NumberOfInteractions(snapshot="s"), TemporalDegreeCentrality()],
        distance=KSDistance(),
    )
    summary = evaluator.evaluate_many(
        path_graph(),
        [
            ("same", path_graph()),
            ("different", interaction_graph()),
        ],
    )

    grouped = {
        MetricCategory.TEMPORAL_FIDELITY: ["number_of_interactions"],
        MetricCategory.GLOBAL_METRICS: ["temporal_degree_centrality"],
    }

    similarity = summary.metric_similarity()
    spread = summary.metric_spread()
    per_category = summary.category_similarity(grouped)

    assert similarity["number_of_interactions"] < 1.0
    assert similarity["temporal_degree_centrality"] < 1.0
    assert spread["number_of_interactions"] > 0.0
    assert spread["temporal_degree_centrality"] > 0.0
    assert per_category[MetricCategory.TEMPORAL_FIDELITY] == pytest.approx(
        similarity["number_of_interactions"]
    )
    assert per_category[MetricCategory.GLOBAL_METRICS] == pytest.approx(
        similarity["temporal_degree_centrality"]
    )
