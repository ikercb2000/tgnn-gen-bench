from __future__ import annotations

# import packages

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# import modules

from tgnn_gen_bench.evaluation import EvaluationSummary, Evaluator
from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.report import radar_plotter
from tgnn_gen_bench.report.style import DPI, apply_style

# pipeline result


@dataclass
class PipelineResult:
    """Store the outputs produced by one evaluation pipeline run."""

    summary: EvaluationSummary
    plot_data: dict[str, Any]
    output_dir: Path
    plot_dir: Path
    plot_base: Path


# evaluation radar pipeline


class EvaluationRadarPipeline:
    """Run evaluator and radar plotter end to end for graph comparison."""

    def __init__(self, evaluator: Evaluator) -> None:
        """Store the evaluator used by the pipeline."""
        self.evaluator = evaluator

    def run(
        self,
        reference: Graph,
        generated: Graph,
        *,
        output_dir: str | Path,
        base_name: str = "comparison",
        reference_name: str = "reference",
        generated_name: str = "generated",
        comparison_label: str | None = None,
        tex: bool = False,
        dpi: int = DPI,
    ) -> PipelineResult:
        """Evaluate two graphs and save radar plots in a dedicated folder."""
        output_path = Path(output_dir)
        plot_dir = output_path / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        summary = self.evaluator.evaluate_many(reference, [(generated_name, generated)])
        tex_enabled = apply_style(usetex=tex, dpi=dpi)
        plot_base = plot_dir / base_name
        plot_data = radar_plotter.plot(
            summary,
            plot_base,
            tex=tex_enabled,
            dpi=dpi,
            graph_labels={generated_name: generated_name},
            comparison_label=comparison_label or f"{reference_name} vs {generated_name}",
        )

        return PipelineResult(
            summary=summary,
            plot_data=plot_data,
            output_dir=output_path,
            plot_dir=plot_dir,
            plot_base=plot_base,
        )


__all__ = ["EvaluationRadarPipeline", "PipelineResult"]
