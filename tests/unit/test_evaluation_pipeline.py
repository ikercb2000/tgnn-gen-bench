# import packages

from pathlib import Path
from uuid import uuid4

import matplotlib

# import modules

from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation import Evaluator
from tgnn_gen_bench.metrics import NumberOfInteractions, TemporalDegreeCentrality
from tgnn_gen_bench.pipeline import EvaluationRadarPipeline, PipelineResult

from ._metric_graphs import interaction_graph, path_graph

# backend

matplotlib.use("Agg")

# evaluation pipeline tests


def _pipeline_workdir() -> Path:
    """Create a repo-local directory for pipeline test artifacts."""
    workdir = Path(".tmp") / "pipeline_tests" / uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def test_evaluation_radar_pipeline_saves_plots_in_dedicated_folder() -> None:
    evaluator = Evaluator(
        metrics=[NumberOfInteractions(snapshot="s"), TemporalDegreeCentrality()],
        distance=KSDistance(),
    )
    pipeline = EvaluationRadarPipeline(evaluator)
    workdir = _pipeline_workdir()

    result = pipeline.run(
        path_graph(),
        interaction_graph(),
        output_dir=workdir,
        base_name="demo",
        reference_name="reference_graph",
        generated_name="generated_graph",
    )

    assert isinstance(result, PipelineResult)
    assert result.output_dir == workdir
    assert result.plot_dir == workdir / "plots"
    assert result.plot_base == workdir / "plots" / "demo"
    assert (workdir / "plots" / "demo_categories.png").exists()
    assert (workdir / "plots" / "demo_categories.pdf").exists()
    assert (workdir / "plots" / "demo_metrics.png").exists()
    assert (workdir / "plots" / "demo_metrics.pdf").exists()
    assert not (workdir / "demo_categories.png").exists()
    assert result.plot_data["graph_names"] == ["generated_graph"]
