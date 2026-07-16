"""Radar plotter object."""

# import packages

from collections.abc import Mapping

# import modules

from tgnn_gen_bench.evaluation.results import EvaluationSummary
from tgnn_gen_bench.metrics.categories import MetricCategory
from tgnn_gen_bench.report._radar_figures import by_category, by_metric
from tgnn_gen_bench.report._radar_helpers import group_by_category
from tgnn_gen_bench.report.style import DPI

# radar plotter


class RadarPlotter:
    """Create radar plots directly from evaluator summaries."""

    def grouped_metrics(self, summary: EvaluationSummary) -> dict[MetricCategory, list[str]]:
        """Group metrics by category from an evaluation summary."""
        return group_by_category(summary.scores_by_metric, summary.category_of)

    def metric_similarity(self, summary: EvaluationSummary) -> dict[str, float]:
        """Extract per-metric similarity values from an evaluation summary."""
        return summary.metric_similarity()

    def metric_spread(self, summary: EvaluationSummary) -> dict[str, float]:
        """Extract per-metric spread values from an evaluation summary."""
        return summary.metric_spread()

    def category_similarity(self, summary: EvaluationSummary) -> dict[MetricCategory, float]:
        """Extract per-category similarity values from an evaluation summary."""
        grouped = self.grouped_metrics(summary)
        return summary.category_similarity(grouped)

    def graph_names(
        self,
        summary: EvaluationSummary,
        graph_labels: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Resolve display names for the generated graphs in a summary."""
        names = list(summary.results_by_graph)
        if graph_labels is None:
            return names
        return [graph_labels.get(name, name) for name in names]

    def context_label(
        self,
        summary: EvaluationSummary,
        *,
        comparison_label: str | None = None,
        graph_labels: Mapping[str, str] | None = None,
    ) -> str | None:
        """Build an optional title label describing the compared graphs."""
        if comparison_label is not None:
            return comparison_label

        names = self.graph_names(summary, graph_labels=graph_labels)
        if not names:
            return None
        if len(names) <= 3:
            return "Graphs: " + ", ".join(names)
        return f"Graphs: {len(names)} generated graphs"

    def plot_by_category(
        self,
        summary: EvaluationSummary,
        out,
        tex: bool,
        dpi: int = DPI,
        *,
        comparison_label: str | None = None,
        graph_labels: Mapping[str, str] | None = None,
    ) -> dict:
        """Plot category-level similarities from an evaluation summary."""
        return by_category(
            self.category_similarity(summary),
            out,
            tex,
            dpi=dpi,
            context_label=self.context_label(
                summary,
                comparison_label=comparison_label,
                graph_labels=graph_labels,
            ),
        ) | {
            "distance_name": summary.distance_name,
            "graph_names": self.graph_names(summary, graph_labels=graph_labels),
        }

    def plot_by_metric(
        self,
        summary: EvaluationSummary,
        out,
        tex: bool,
        dpi: int = DPI,
        *,
        band: bool | None = None,
        comparison_label: str | None = None,
        graph_labels: Mapping[str, str] | None = None,
    ) -> dict:
        """Plot metric-level similarities from an evaluation summary."""
        grouped = self.grouped_metrics(summary)
        if band is None:
            band = len(summary.results_by_graph) > 1
        return by_metric(
            self.metric_similarity(summary),
            self.metric_spread(summary),
            grouped,
            out,
            tex,
            dpi=dpi,
            band=band,
            context_label=self.context_label(
                summary,
                comparison_label=comparison_label,
                graph_labels=graph_labels,
            ),
        ) | {
            "distance_name": summary.distance_name,
            "graph_names": self.graph_names(summary, graph_labels=graph_labels),
        }

    def plot(
        self,
        summary: EvaluationSummary,
        out,
        tex: bool,
        dpi: int = DPI,
        *,
        band: bool | None = None,
        comparison_label: str | None = None,
        graph_labels: Mapping[str, str] | None = None,
    ) -> dict:
        """Create both radar plots from one evaluation summary."""
        return {
            "grouped_metrics": self.grouped_metrics(summary),
            "graph_names": self.graph_names(summary, graph_labels=graph_labels),
            "distance_name": summary.distance_name,
            "by_category": self.plot_by_category(
                summary,
                f"{out}_categories",
                tex,
                dpi=dpi,
                comparison_label=comparison_label,
                graph_labels=graph_labels,
            ),
            "by_metric": self.plot_by_metric(
                summary,
                f"{out}_metrics",
                tex,
                dpi=dpi,
                band=band,
                comparison_label=comparison_label,
                graph_labels=graph_labels,
            ),
        }


radar_plotter = RadarPlotter()

__all__ = ["RadarPlotter", "radar_plotter"]
