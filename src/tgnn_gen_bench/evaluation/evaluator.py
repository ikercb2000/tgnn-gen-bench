# import packages

from collections.abc import Mapping, Sequence
from typing import Any

# import modules

from tgnn_gen_bench.evaluation.results import EvaluationSummary, MetricResult
from tgnn_gen_bench.graphs.base import Graph
from tgnn_gen_bench.metrics.base import Metric
from tgnn_gen_bench.metrics.categories import MetricCategory

# evaluation pipeline class


class Evaluator:
    """Run a set of metrics and compare their outputs with a distance."""

    def __init__(self, metrics: Sequence[Metric], distance: Any) -> None:
        """Store the metrics and distance used during evaluation."""
        self.metrics = list(metrics)
        self.distance = distance

    def evaluate(self, reference: Graph, generated: Graph) -> list[MetricResult]:
        """Evaluate one generated graph against the reference graph."""
        results: list[MetricResult] = []

        for metric in self.metrics:
            real_value = metric.compute(reference)
            generated_value = metric.compute(generated)

            score = self.distance.compute(real_value, generated_value)

            results.append(
                MetricResult(
                    metric_name=metric.name,
                    category=metric.category,
                    score=float(score),
                    real_value=real_value,
                    generated_value=generated_value,
                    distance_name=self.distance.name,
                )
            )

        return results

    def evaluate_many(
        self,
        reference: Graph,
        generated_graphs: Mapping[str, Graph] | Sequence[tuple[str, Graph]],
    ) -> EvaluationSummary:
        """Evaluate several generated graphs against one reference graph."""
        results_by_graph: dict[str, list[MetricResult]] = {}
        scores_by_metric: dict[str, list[float]] = {}
        category_of: dict[str, MetricCategory] = {}

        for graph_name, generated_graph in self._named_graphs(generated_graphs):
            graph_results = self.evaluate(reference, generated_graph)
            results_by_graph[graph_name] = graph_results

            for result in graph_results:
                if result.score is None:
                    continue
                scores_by_metric.setdefault(result.metric_name, []).append(float(result.score))
                category_of[result.metric_name] = result.category

        return EvaluationSummary(
            distance_name=self.distance.name,
            results_by_graph=results_by_graph,
            scores_by_metric=scores_by_metric,
            category_of=category_of,
        )

    def _named_graphs(
        self,
        generated_graphs: Mapping[str, Graph] | Sequence[tuple[str, Graph]],
    ) -> list[tuple[str, Graph]]:
        """Normalize generated graphs into a name-graph list."""
        if isinstance(generated_graphs, Mapping):
            return list(generated_graphs.items())
        return list(generated_graphs)
