# import packages

from typing import Sequence

# import modules

from tgnn_gen_bench.evaluation.results import MetricResult
from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.base import Metric

# evaluation pipeline class

class Evaluator:
    def __init__(self, metrics: Sequence[Metric], distance) -> None:
        self.metrics = list(metrics)
        self.distance = distance

    def evaluate(self, reference: Graph, generated: Graph) -> list[MetricResult]:
        results: list[MetricResult] = []

        for metric in self.metrics:
            real_value = metric.compute(reference)
            generated_value = metric.compute(generated)

            score = self.distance.compute(real_value,generated_value)

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
