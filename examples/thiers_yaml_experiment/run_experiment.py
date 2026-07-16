"""Run pairwise Thiers comparisons from a YAML experiment file."""

# import packages

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

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
from tgnn_gen_bench.report import radar_plotter
from tgnn_gen_bench.report._radar_helpers import polygon_area, radar_axes, ring
from tgnn_gen_bench.report.radar import CATEGORY_LABEL, CATEGORY_ORDER
from tgnn_gen_bench.report.style import (
    DARK_GREY,
    DPI,
    FONT_TITLE,
    FULL_W,
    ERROR_BAND_ALPHA,
    LINE_LW,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    MARKER_SIZE,
    TWO3_W,
    apply_style,
    save,
)

# paths

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "experiment.yaml"

# dataset helpers


def load_config(path: Path) -> dict:
    """Load the YAML configuration for the experiment."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_temporal_edges(path: Path, delimiter: str) -> list[TemporalEdge]:
    """Load temporal edges from the configured dataset file."""
    edges: list[TemporalEdge] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) < 3:
                continue
            edges.append(
                TemporalEdge(
                    source=int(row[1]),
                    target=int(row[2]),
                    timestamp=int(row[0]),
                )
            )
    if not edges:
        raise ValueError(f"No temporal edges could be read from {path}.")
    return edges


def partition_edges(
    edges: list[TemporalEdge],
    *,
    count: int,
) -> dict[str, list[TemporalEdge]]:
    """Partition the dataset into contiguous timestamp windows."""
    if count < 2:
        raise ValueError("partition count must be at least 2.")

    ordered_edges = sorted(edges, key=lambda edge: (edge.timestamp, edge.source, edge.target))
    timestamps = sorted({edge.timestamp for edge in ordered_edges})
    if len(timestamps) < count:
        raise ValueError("partition count is larger than the number of distinct timestamps.")

    partitions: dict[str, list[TemporalEdge]] = {}
    for index in range(count):
        start = index * len(timestamps) // count
        end = (index + 1) * len(timestamps) // count
        window_times = set(timestamps[start:end])
        graph_id = f"graph_{index}"
        graph_edges = [edge for edge in ordered_edges if edge.timestamp in window_times]
        if not graph_edges:
            raise ValueError(f"Partition {graph_id} is empty.")
        partitions[graph_id] = graph_edges
    return partitions


def build_graphs(config: dict) -> dict[str, object]:
    """Build all configured graphs from the experiment dataset."""
    dataset_path = ROOT / config["dataset"]["path"]
    delimiter = config["dataset"].get("delimiter", "\t")
    time_unit = config["dataset"].get("time_unit", "s")
    edges = load_temporal_edges(dataset_path, delimiter=delimiter)
    partition_count = int(config["graphs"]["partition"]["count"])
    partitions = partition_edges(edges, count=partition_count)
    return {
        graph_id: from_temporal_edges(graph_edges, time_delta=time_unit)
        for graph_id, graph_edges in partitions.items()
    }


# evaluator helpers


def build_evaluator(config: dict) -> Evaluator:
    """Build the evaluator specified by the experiment configuration."""
    snapshot = config["metrics"]["snapshot"]
    directed = bool(config["metrics"].get("directed", False))
    random_walk = config["metrics"]["random_walk"]
    downstream_tasks = config["metrics"]["downstream_tasks"]

    metrics = [
        NumberOfInteractions(snapshot=snapshot),
        InteractingIndividuals(snapshot=snapshot, directed=directed),
        TemporalDegreeCentrality(),
        TemporalReachability(),
        TemporalDegreeDistribution(snapshot=snapshot),
        RandomWalkEntropy(
            snapshot=snapshot,
            n_samples=int(random_walk["n_samples"]),
            n_walks=int(random_walk["n_walks"]),
            horizon=int(random_walk["horizon"]),
            seed=int(random_walk["seed"]),
        ),
        TemporalLinkPredictionLogLoss(seed=int(downstream_tasks["seed"])),
        TemporalLinkPredictionRocAuc(seed=int(downstream_tasks["seed"])),
    ]
    return Evaluator(metrics=metrics, distance=KSDistance())


# output helpers


def output_dir(config: dict) -> Path:
    """Resolve the root output directory for the experiment."""
    return ROOT / config["output"]["root_dir"]


def comparison_name(prefix: str, historical_graph: str, generated_graph: str) -> str:
    """Build a stable comparison name for one pair of graphs."""
    return f"{prefix}_{historical_graph}_vs_{generated_graph}"


def display_model_name(graph_id: str) -> str:
    """Convert one internal graph id into a cleaner model label."""
    if graph_id.startswith("graph_"):
        suffix = graph_id.removeprefix("graph_")
        return f"model_{suffix}"
    return graph_id


def write_sidecar(path: Path, payload: dict) -> None:
    """Write JSON metadata describing one pipeline run."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_combined_category_plot(
    *,
    category_similarity_by_graph: dict[str, dict],
    historical_graph_id: str,
    dataset_label: str,
    out_dir: Path,
    tex: bool,
    dpi: int,
) -> Path:
    """Write a panel with one category radar per generated graph."""
    categories = [
        category
        for category in CATEGORY_ORDER
        if any(category in values for values in category_similarity_by_graph.values())
    ]
    labels = [CATEGORY_LABEL.get(category, str(category)) for category in categories]
    theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    palette = sns.color_palette("crest", n_colors=max(len(category_similarity_by_graph), 3))

    tex_enabled = apply_style(usetex=tex, dpi=dpi)
    fig, axes = plt.subplots(
        1,
        len(category_similarity_by_graph),
        figsize=(FULL_W * 1.65, TWO3_W * 1.05),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    fig.patch.set_facecolor("white")

    for index, (ax, (graph_id, values)) in enumerate(zip(axes, category_similarity_by_graph.items(), strict=False)):
        mean = np.array([float(values.get(category, np.nan)) for category in categories], dtype=float)
        area = polygon_area(mean)
        colour = palette[index % len(palette)]
        display_name = display_model_name(graph_id)
        ax.fill(ring(theta), ring(mean), color=colour, alpha=ERROR_BAND_ALPHA + 0.05, linewidth=0)
        ax.plot(
            ring(theta),
            ring(mean),
            color=colour,
            lw=LINE_LW + 0.4,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
        )
        radar_axes(
            ax,
            theta,
            labels,
            display_name,
            area,
        )

    fig.suptitle(
        f"{dataset_label}: category similarity across generated models",
        fontsize=FONT_TITLE + 2,
        color=DARK_GREY,
        y=1.03,
    )

    out_path = out_dir / "combined_category_radars"
    save(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# experiment


def main() -> None:
    """Run the configured pairwise Thiers experiment."""
    config = load_config(CONFIG_PATH)
    graphs = build_graphs(config)
    evaluator = build_evaluator(config)
    pipeline = EvaluationRadarPipeline(evaluator)

    historical_graph_id = config["graphs"]["historical_graph"]
    generated_graph_ids = list(config["graphs"]["generated_graphs"])
    prefix = config["output"]["base_name_prefix"]
    tex = bool(config["output"].get("tex", False))
    dpi = int(config["output"].get("dpi", 300))
    root_output_dir = output_dir(config)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    combined_category_similarity: dict[str, dict] = {}

    if historical_graph_id not in graphs:
        raise ValueError(f"Unknown historical_graph: {historical_graph_id}")

    historical_graph = graphs[historical_graph_id]
    manifest: dict[str, object] = {
        "config": str(CONFIG_PATH),
        "dataset": config["dataset"]["path"],
        "historical_graph": historical_graph_id,
        "generated_graphs": generated_graph_ids,
        "comparisons": [],
    }

    for generated_graph_id in generated_graph_ids:
        if generated_graph_id not in graphs:
            raise ValueError(f"Unknown generated graph id: {generated_graph_id}")

        name = comparison_name(prefix, historical_graph_id, generated_graph_id)
        comparison_output_dir = root_output_dir / name
        result = pipeline.run(
            historical_graph,
            graphs[generated_graph_id],
            output_dir=comparison_output_dir,
            base_name=name,
            reference_name=historical_graph_id,
            generated_name=generated_graph_id,
            comparison_label=f"{historical_graph_id} vs {generated_graph_id}",
            tex=tex,
            dpi=dpi,
        )

        sidecar_path = comparison_output_dir / f"{name}.summary.json"
        sidecar = {
            "historical_graph": historical_graph_id,
            "generated_graph": generated_graph_id,
            "plot_dir": str(result.plot_dir),
            "plot_base": str(result.plot_base),
            "distance_name": result.summary.distance_name,
            "scores_by_metric": result.summary.scores_by_metric,
            "category_of": {key: str(value) for key, value in result.summary.category_of.items()},
            "plot_data": result.plot_data,
        }
        write_sidecar(sidecar_path, sidecar)
        combined_category_similarity[generated_graph_id] = radar_plotter.category_similarity(
            result.summary
        )

        manifest["comparisons"].append(
            {
                "name": name,
                "output_dir": str(comparison_output_dir),
                "plot_dir": str(result.plot_dir),
                "category_plot": str(result.plot_base) + "_categories.png",
                "metric_plot": str(result.plot_base) + "_metrics.png",
                "summary": str(sidecar_path),
            }
        )

        print(f"comparison: {historical_graph_id} vs {generated_graph_id}")
        print(f"plots: {result.plot_dir}")

    combined_plot_path = write_combined_category_plot(
        category_similarity_by_graph=combined_category_similarity,
        historical_graph_id=historical_graph_id,
        dataset_label=Path(config["dataset"]["path"]).stem,
        out_dir=root_output_dir,
        tex=tex,
        dpi=dpi,
    )
    manifest["combined_category_plot"] = str(combined_plot_path) + ".png"
    write_sidecar(root_output_dir / "manifest.json", manifest)
    print(f"combined plot: {combined_plot_path}.png/.pdf")
    print(f"manifest: {root_output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
