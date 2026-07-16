# import packages

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# import modules

from tgnn_gen_bench.graphs import Graph
from tgnn_gen_bench.metrics.categories import MetricCategory

# generic variable type

GraphT = TypeVar("GraphT", bound=Graph)
OutputT = TypeVar("OutputT")

# base class metric

class Metric(ABC, Generic[GraphT, OutputT]):
    """Define the interface shared by every metric."""

    name: str
    category: MetricCategory

    @abstractmethod
    def compute(self, graph: GraphT) -> OutputT:
        """Compute the metric on one graph."""
        pass

    def __call__(self, graph: GraphT) -> OutputT:
        """Call `compute` through function syntax."""
        return self.compute(graph)
