# import packages

from pathlib import Path
from uuid import uuid4

# import modules

from tgnn_gen_bench.data import from_csv
from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation import EvaluationSummary, Evaluator
from tgnn_gen_bench.metrics import (
    MetricCategory,
    NumberOfInteractions,
    RandomWalkEntropy,
    TemporalDegreeCentrality,
    TemporalDegreeDistribution,
)
from tgnn_gen_bench.report import radar

# csv graph fixtures


def _write_graph_csv(path: Path, rows: list[tuple[int, int, int, float]]) -> None:
    """Write a weighted temporal edge list to CSV."""
    lines = ["t,i,j,weight"]
    lines.extend(f"{time},{source},{target},{weight}" for time, source, target, weight in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _integration_workdir() -> Path:
    """Create a repo-local directory for integration test artifacts."""
    workdir = Path(".tmp") / "integration_tests" / uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


# evaluator integration tests


def test_evaluator_integrates_graph_loading_metrics_distance_and_grouping() -> None:
    workdir = _integration_workdir()
    reference_path = workdir / "reference.csv"
    generated_same_path = workdir / "generated_same.csv"
    generated_shifted_path = workdir / "generated_shifted.csv"

    _write_graph_csv(
        reference_path,
        [
            (1, 0, 1, 1.0),
            (2, 1, 2, 2.0),
            (3, 0, 2, 1.5),
        ],
    )
    _write_graph_csv(
        generated_same_path,
        [
            (1, 0, 1, 1.0),
            (2, 1, 2, 2.0),
            (3, 0, 2, 1.5),
        ],
    )
    _write_graph_csv(
        generated_shifted_path,
        [
            (1, 0, 1, 5.0),
            (2, 0, 1, 5.0),
            (3, 1, 2, 5.0),
            (4, 1, 2, 5.0),
        ],
    )

    reference = from_csv(reference_path, time_unit="s")
    generated_same = from_csv(generated_same_path, time_unit="s")
    generated_shifted = from_csv(generated_shifted_path, time_unit="s")

    evaluator = Evaluator(
        metrics=[
            NumberOfInteractions(snapshot="s"),
            TemporalDegreeCentrality(),
            TemporalDegreeDistribution(snapshot="s"),
            RandomWalkEntropy(snapshot="s", n_samples=2, n_walks=16, horizon=3, seed=7),
        ],
        distance=KSDistance(),
    )

    summary = evaluator.evaluate_many(
        reference,
        [
            ("generated_same.csv", generated_same),
            ("generated_shifted.csv", generated_shifted),
        ],
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.distance_name == "ks"
    assert set(summary.results_by_graph) == {"generated_same.csv", "generated_shifted.csv"}
    assert set(summary.scores_by_metric) == {
        "number_of_interactions",
        "temporal_degree_centrality",
        "temporal_degree_distribution",
        "random_walk_entropy",
    }
    assert summary.category_of == {
        "number_of_interactions": MetricCategory.TEMPORAL_FIDELITY,
        "temporal_degree_centrality": MetricCategory.GLOBAL_METRICS,
        "temporal_degree_distribution": MetricCategory.STRUCTURAL_METRICS,
        "random_walk_entropy": MetricCategory.DYNAMICS_METRICS,
    }

    assert summary.scores_by_metric["number_of_interactions"][0] == 0.0
    assert summary.scores_by_metric["temporal_degree_centrality"][0] == 0.0
    assert summary.scores_by_metric["temporal_degree_distribution"][0] == 0.0

    grouped = radar.group_by_category(summary.scores_by_metric, summary.category_of)
    similarity = summary.metric_similarity()
    spread = summary.metric_spread()
    per_category = summary.category_similarity(grouped)

    assert list(grouped) == [
        MetricCategory.TEMPORAL_FIDELITY,
        MetricCategory.GLOBAL_METRICS,
        MetricCategory.STRUCTURAL_METRICS,
        MetricCategory.DYNAMICS_METRICS,
    ]
    assert similarity["temporal_degree_centrality"] < 1.0
    assert spread["temporal_degree_centrality"] > 0.0
    assert per_category[MetricCategory.TEMPORAL_FIDELITY] == similarity["number_of_interactions"]
    assert per_category[MetricCategory.GLOBAL_METRICS] == similarity["temporal_degree_centrality"]
    assert per_category[MetricCategory.STRUCTURAL_METRICS] == similarity[
        "temporal_degree_distribution"
    ]
