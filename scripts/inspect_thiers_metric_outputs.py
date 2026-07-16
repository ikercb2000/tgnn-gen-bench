"""Print exact metric outputs for two temporal graphs built from thiers_2012.csv."""

# import packages

import argparse
import csv
import json
from pathlib import Path

# import modules

from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation import Evaluator
from tgnn_gen_bench.graphs import TemporalEdge, from_temporal_edges
from tgnn_gen_bench.metrics import (
    CommunityEventDistribution,
    CommunityPersistence,
    DegreeAdjustedStrength,
    DurationOfContacts,
    InteractingIndividuals,
    ModularityTrajectory,
    NewInteractions,
    NumberOfInteractions,
    RandomWalkEntropy,
    TemporalClosenessCentrality,
    TemporalDegreeCentrality,
    TemporalDegreeDistribution,
    TemporalLinkPredictionLogLoss,
    TemporalLinkPredictionRocAuc,
    TemporalReachability,
    TemporalStrengthDistribution,
    WindowedReachability,
)

# paths

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "thiers_2012.csv"

# metric profiles


def build_metrics(args: argparse.Namespace) -> list:
    """Build the metric list requested from the CLI."""
    snapshot = args.snapshot

    profiles = {
        "global": [
            TemporalClosenessCentrality(),
            TemporalDegreeCentrality(),
            TemporalReachability(),
            WindowedReachability(snapshot=snapshot),
        ],
        "temporal": [
            NumberOfInteractions(snapshot=snapshot),
            InteractingIndividuals(snapshot=snapshot, directed=args.directed),
            NewInteractions(snapshot=snapshot, directed=args.directed),
            DurationOfContacts(snapshot=snapshot, directed=args.directed),
        ],
        "structural": [
            TemporalDegreeDistribution(snapshot=snapshot),
            TemporalStrengthDistribution(snapshot=snapshot),
            DegreeAdjustedStrength(snapshot=snapshot),
            CommunityPersistence(snapshot=snapshot),
            CommunityEventDistribution(snapshot=snapshot),
            ModularityTrajectory(snapshot=snapshot),
        ],
        "dynamics": [
            RandomWalkEntropy(snapshot=snapshot, seed=args.seed),
        ],
        "downstream": [
            TemporalLinkPredictionLogLoss(seed=args.seed),
            TemporalLinkPredictionRocAuc(seed=args.seed),
        ],
    }

    if args.profile == "all":
        metrics = []
        for name in ("global", "temporal", "structural", "dynamics", "downstream"):
            metrics.extend(profiles[name])
        return metrics

    return profiles[args.profile]


# dataset helpers


def load_temporal_edges(path: Path) -> list[TemporalEdge]:
    """Load temporal edges from the Thiers TSV dataset."""
    edges: list[TemporalEdge] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
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


def partition_edges(edges: list[TemporalEdge], count: int) -> dict[str, list[TemporalEdge]]:
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
        partitions[f"graph_{index}"] = [
            edge for edge in ordered_edges if edge.timestamp in window_times
        ]
    return partitions


def load_graph_pair(args: argparse.Namespace):
    """Build the requested graph pair from thiers_2012.csv."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    partitions = partition_edges(load_temporal_edges(DATASET_PATH), count=args.partition_count)
    if args.reference_graph not in partitions:
        raise ValueError(f"Unknown reference graph id: {args.reference_graph}")
    if args.generated_graph not in partitions:
        raise ValueError(f"Unknown generated graph id: {args.generated_graph}")

    reference = from_temporal_edges(partitions[args.reference_graph], time_delta="s")
    generated = from_temporal_edges(partitions[args.generated_graph], time_delta="s")
    return reference, generated


# terminal helpers


def render_value(value, compact: bool) -> str:
    """Render one metric output for terminal display."""
    if compact:
        return json.dumps(value, separators=(",", ":"))
    return json.dumps(value, indent=2)


def print_result(result, compact: bool) -> None:
    """Print one metric comparison block."""
    print("=" * 88)
    print(f"metric    : {result.metric_name}")
    print(f"category  : {result.category}")
    print(f"distance  : {result.distance_name}")
    print(f"ks_score  : {result.score}")
    print()
    print("reference output")
    print("-" * 88)
    print(render_value(result.real_value, compact=compact))
    print()
    print("generated output")
    print("-" * 88)
    print(render_value(result.generated_value, compact=compact))
    print()


def main() -> None:
    """Run the Thiers metric inspection script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-graph",
        default="graph_0",
        help="Reference graph id from the thiers_2012 partition, e.g. graph_0.",
    )
    parser.add_argument(
        "--generated-graph",
        default="graph_1",
        help="Generated graph id from the thiers_2012 partition, e.g. graph_1.",
    )
    parser.add_argument(
        "--partition-count",
        type=int,
        default=4,
        help="Number of contiguous temporal graphs to build from thiers_2012.csv.",
    )
    parser.add_argument(
        "--profile",
        choices=["global", "temporal", "structural", "dynamics", "downstream", "all"],
        default="global",
        help="Which metric family to inspect.",
    )
    parser.add_argument(
        "--snapshot",
        default="h",
        help="Snapshot granularity for metrics that need it, e.g. 'm' or 'h'.",
    )
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print metric outputs on a single JSON line per value.",
    )
    args = parser.parse_args()

    reference, generated = load_graph_pair(args)
    evaluator = Evaluator(metrics=build_metrics(args), distance=KSDistance())
    results = evaluator.evaluate(reference, generated)

    print(f"dataset   : {DATASET_PATH}")
    print(f"reference : {args.reference_graph}")
    print(f"generated : {args.generated_graph}")
    print(f"profile   : {args.profile}")
    print(f"snapshot  : {args.snapshot}")
    print()

    for result in results:
        print_result(result, compact=args.compact)

    print("=" * 88)


if __name__ == "__main__":
    main()
