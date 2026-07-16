"""Evaluate generated temporal graphs against a reference.

Point it at one reference graph and a collection of generated graphs. Every
metric is computed per graph, each generated graph is compared to the reference
with a Kolmogorov-Smirnov distance, and two radars are drawn:

    <out>_categories.{png,pdf}  one axis per category, the mean over its metrics
    <out>_metrics.{png,pdf}     one axis per metric, grouped and coloured by category

    python scripts/evaluate.py \
        --reference data/hospital.csv \
        --generated generated/*.csv \
        --time-unit s --snapshot m \
        --out figures/fidelity

Scores need no standardisation: the KS distance is bounded in [0, 1] whatever
the units of the metric, which is what makes the axes commensurable. The radars
plot similarity = 1 - KS, so further from the centre is more similar.
"""

# import packages

import argparse
import json
from pathlib import Path

import numpy as np

# import modules

from tgm import TimeDeltaDG

from tgnn_gen_bench.data import from_csv
from tgnn_gen_bench.distances import KSDistance
from tgnn_gen_bench.evaluation.evaluator import Evaluator
from tgnn_gen_bench.metrics.temporal_fidelity import (
    DurationOfContacts,
    InteractingIndividuals,
    NewInteractions,
    NumberOfInteractions,
)
from tgnn_gen_bench.report import radar
from tgnn_gen_bench.report.style import DPI, DRAFT_DPI, apply_style


def build_metrics(args: argparse.Namespace) -> list:
    """The metrics to run, in the order their axes appear."""
    snapshot = args.snapshot
    return [
        NumberOfInteractions(snapshot=snapshot),
        InteractingIndividuals(snapshot=snapshot, directed=args.directed),
        NewInteractions(snapshot=snapshot, directed=args.directed),
        DurationOfContacts(snapshot=snapshot, directed=args.directed),
    ]


def load(path: Path, args: argparse.Namespace):
    return from_csv(path, src_col=args.src_col, dst_col=args.dst_col,
                    time_col=args.time_col, time_unit=args.time_unit,
                    device=args.device)


def evaluate(args: argparse.Namespace) -> tuple[dict[str, list[float]], dict[str, str]]:
    """KS distance per metric, one score per generated graph, plus its category."""
    reference = load(args.reference, args)
    evaluator = Evaluator(metrics=build_metrics(args), distance=KSDistance())

    scores: dict[str, list[float]] = {}
    category_of: dict[str, str] = {}
    for path in args.generated:
        print(f"  comparing {path.name} ...", flush=True)
        for result in evaluator.evaluate(reference, load(path, args)):
            scores.setdefault(result.metric_name, []).append(result.score)
            category_of[result.metric_name] = result.category
    return scores, category_of


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--generated", type=Path, nargs="+", required=True)
    p.add_argument("--src-col", default="i")
    p.add_argument("--dst-col", default="j")
    p.add_argument("--time-col", default="t")
    p.add_argument("--time-unit", default="s",
                   help="time unit of the timestamp column; must be time-ordered")
    p.add_argument("--snapshot", default=None,
                   help="snapshot granularity, e.g. 'm' or 'h'. Default: the "
                        "graph's own time_delta.")
    p.add_argument("--directed", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--draft", action="store_true", help="no usetex, lower dpi")
    p.add_argument("--out", type=Path, default=Path("figures/fidelity"))
    args = p.parse_args()

    if TimeDeltaDG(args.time_unit).is_event_ordered:
        p.error("--time-unit must be time-ordered ('s', 'm', 'h', ...); snapshot "
                "metrics are undefined on an event-ordered graph.")

    print(f"reference: {args.reference}")
    print(f"generated: {len(args.generated)} graph(s)")
    scores, category_of = evaluate(args)
    grouped = radar.group_by_category(scores, category_of)

    print()
    print(f"{'category':22s} {'metric':24s} {'KS mean':>9s} {'KS std':>8s}")
    for category, names in grouped.items():
        for n in names:
            print(f"{category:22s} {n:24s} "
                  f"{np.mean(scores[n]):9.3f} {np.std(scores[n]):8.3f}")

    if len(grouped) < 3:
        print()
        print(f"note: only {len(grouped)} categor{'y' if len(grouped) == 1 else 'ies'} "
              f"populated; a radar needs at least 3 axes to enclose an area.")

    similarity = {n: 1.0 - float(np.mean(v)) for n, v in scores.items()}
    spread = {n: float(np.std(v)) for n, v in scores.items()}
    per_category = {
        c: float(np.mean([similarity[n] for n in names]))
        for c, names in grouped.items()
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dpi = DRAFT_DPI if args.draft else DPI
    tex = apply_style(usetex=not args.draft, dpi=dpi)

    sidecar = {
        "script": Path(__file__).name,
        "reference": str(args.reference),
        "generated": [str(pth) for pth in args.generated],
        "n_generated": len(args.generated),
        "snapshot": args.snapshot or "native time_delta",
        "distance": "ks",
        "ks_per_graph": scores,
        "category_of": category_of,
        "by_category": radar.by_category(
            per_category, f"{args.out}_categories", tex, dpi=dpi),
        "by_metric": radar.by_metric(
            similarity, spread, grouped, f"{args.out}_metrics", tex, dpi=dpi,
            band=len(args.generated) > 1),
    }
    with open(f"{args.out}.plotdata.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    print()
    print(f"wrote {args.out}_categories.png/.pdf, {args.out}_metrics.png/.pdf, "
          f"{args.out}.plotdata.json")


if __name__ == "__main__":
    main()
