# import packages

from pathlib import Path
from uuid import uuid4

import matplotlib

# import modules

from tgnn_gen_bench.evaluation import Evaluator
from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.metrics import NumberOfInteractions, TemporalDegreeCentrality
from tgnn_gen_bench.report import radar_plotter

from ._metric_graphs import interaction_graph, path_graph

# backend

matplotlib.use("Agg")

# radar plotter tests


def _report_workdir() -> Path:
    """Create a repo-local directory for report test artifacts."""
    workdir = Path(".tmp") / "report_tests" / uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def test_radar_plotter_builds_plots_from_evaluation_summary() -> None:
    evaluator = Evaluator(
        metrics=[NumberOfInteractions(snapshot="s"), TemporalDegreeCentrality()],
        distance=KSDistance(),
    )
    summary = evaluator.evaluate_many(
        path_graph(),
        [
            ("same_graph", path_graph()),
            ("other_graph", interaction_graph()),
        ],
    )
    out = _report_workdir() / "summary"

    plot_data = radar_plotter.plot(
        summary,
        out,
        tex=False,
        graph_labels={"same_graph": "Same", "other_graph": "Other"},
    )

    assert plot_data["distance_name"] == "ks"
    assert plot_data["graph_names"] == ["Same", "Other"]
    assert (out.parent / "summary_categories.png").exists()
    assert (out.parent / "summary_metrics.png").exists()
    assert plot_data["by_category"]["graph_names"] == ["Same", "Other"]
    assert plot_data["by_metric"]["graph_names"] == ["Same", "Other"]
