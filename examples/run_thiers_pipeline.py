"""Run the evaluation radar pipeline on two graphs built from thiers_2012.csv."""

# import packages

import csv
from pathlib import Path

import matplotlib

# import modules

matplotlib.use("Agg")

from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation import Evaluator
from tgnn_gen_bench.graphs import TemporalEdge, from_temporal_edges
from tgnn_gen_bench.metrics import (
    InteractingIndividuals,
    NumberOfInteractions,
    RandomWalkEntropy,
    TemporalDegreeCentrality,
    TemporalDegreeDistribution,
    TemporalLinkPredictionLogLoss,
    TemporalLinkPredictionRocAuc,
    TemporalReachability,
)
from tgnn_gen_bench.pipeline import EvaluationRadarPipeline

# paths

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "thiers_2012.csv"
OUTPUT_DIR = ROOT / "examples" / "outputs" / "thiers_pipeline"

# dataset helpers


def load_temporal_edges(path: Path) -> list[TemporalEdge]:
    """Load temporal edges from the Thiers TSV file."""
    edges: list[TemporalEdge] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            timestamp = int(row[0])
            source = int(row[1])
            target = int(row[2])
            edges.append(TemporalEdge(source=source, target=target, timestamp=timestamp))
    if not edges:
        raise ValueError(f"No temporal edges could be read from {path}.")
    return edges


def split_edges(edges: list[TemporalEdge]) -> tuple[list[TemporalEdge], list[TemporalEdge]]:
    """Split the dataset into historical and generated-like halves."""
    ordered_edges = sorted(edges, key=lambda edge: (edge.timestamp, edge.source, edge.target))
    timestamps = sorted({edge.timestamp for edge in ordered_edges})
    if len(timestamps) < 2:
        raise ValueError("The dataset needs at least two distinct timestamps.")

    midpoint = len(timestamps) // 2
    historical_times = set(timestamps[:midpoint])
    generated_times = set(timestamps[midpoint:])

    historical_edges = [edge for edge in ordered_edges if edge.timestamp in historical_times]
    generated_edges = [edge for edge in ordered_edges if edge.timestamp in generated_times]
    if not historical_edges or not generated_edges:
        raise ValueError("The temporal split produced an empty graph.")
    return historical_edges, generated_edges


def build_graphs(path: Path):
    """Build two graphs from the Thiers dataset."""
    historical_edges, generated_edges = split_edges(load_temporal_edges(path))
    historical_graph = from_temporal_edges(historical_edges, time_delta="s")
    generated_graph = from_temporal_edges(generated_edges, time_delta="s")
    return historical_graph, generated_graph


# evaluator helpers


def build_evaluator() -> Evaluator:
    """Build the evaluator used by the example pipeline."""
    metrics = [
        NumberOfInteractions(snapshot="h"),
        InteractingIndividuals(snapshot="h", directed=False),
        TemporalDegreeCentrality(),
        TemporalReachability(),
        TemporalDegreeDistribution(snapshot="h"),
        RandomWalkEntropy(snapshot="h", n_samples=4, n_walks=128, horizon=8, seed=7),
        TemporalLinkPredictionLogLoss(seed=7),
        TemporalLinkPredictionRocAuc(seed=7),
    ]
    return Evaluator(metrics=metrics, distance=KSDistance())


# example


def main() -> None:
    """Run the end-to-end example pipeline."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    historical_graph, generated_graph = build_graphs(DATASET_PATH)
    pipeline = EvaluationRadarPipeline(build_evaluator())

    result = pipeline.run(
        historical_graph,
        generated_graph,
        output_dir=OUTPUT_DIR,
        base_name="thiers_comparison",
        reference_name="historical",
        generated_name="generated",
        comparison_label=f"{DATASET_PATH.stem}: historical vs generated",
        tex=False,
    )

    print(f"dataset: {DATASET_PATH}")
    print(f"plots: {result.plot_dir}")
    print(f"category plot: {result.plot_base}_categories.png")
    print(f"metric plot: {result.plot_base}_metrics.png")


if __name__ == "__main__":
    main()
