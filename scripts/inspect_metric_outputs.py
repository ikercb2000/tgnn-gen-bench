"""Print exact metric outputs for two graphs in the terminal."""

# import packages

import argparse
import json
from pathlib import Path

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.data import from_csv
from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation import Evaluator
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


def load_graph(path: Path, args: argparse.Namespace):
    """Load one graph from CSV."""
    return from_csv(
        path,
        src_col=args.src_col,
        dst_col=args.dst_col,
        time_col=args.time_col,
        time_unit=args.time_unit,
        device=args.device,
    )


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
    """Run the metric inspection script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["global", "temporal", "structural", "dynamics", "downstream", "all"],
        default="global",
        help="Which metric family to inspect.",
    )
    parser.add_argument("--src-col", default="i")
    parser.add_argument("--dst-col", default="j")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--time-unit", default="s")
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Snapshot granularity for metrics that need it, e.g. 'm' or 'h'.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print metric outputs on a single JSON line per value.",
    )
    args = parser.parse_args()

    if TimeDeltaDG(args.time_unit).is_event_ordered:
        parser.error("--time-unit must be time-ordered ('s', 'm', 'h', ...).")

    reference = load_graph(args.reference, args)
    generated = load_graph(args.generated, args)
    evaluator = Evaluator(metrics=build_metrics(args), distance=KSDistance())
    results = evaluator.evaluate(reference, generated)

    print(f"reference : {args.reference}")
    print(f"generated : {args.generated}")
    print(f"profile   : {args.profile}")
    if args.snapshot is not None:
        print(f"snapshot  : {args.snapshot}")
    print()

    for result in results:
        print_result(result, compact=args.compact)

    print("=" * 88)


if __name__ == "__main__":
    main()
