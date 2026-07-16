# import packages

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# import modules

from tgnn_gen_bench.graphs import Graph

# generic variable type

GraphT = TypeVar("GraphT", bound=Graph)
OutputT = TypeVar("OutputT")

# base class metric

class Metric(ABC, Generic[GraphT, OutputT]):

    name: str
    category: str

    @abstractmethod
    def compute(self, graph: GraphT) -> OutputT:
        pass

    def __call__(self, graph: GraphT) -> OutputT:
        return self.compute(graph)
