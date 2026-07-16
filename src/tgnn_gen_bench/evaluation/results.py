# import packages

from dataclasses import dataclass, field
from typing import Any

# import modules

from tgnn_gen_bench.metrics.categories import MetricCategory

# Metric results base class

@dataclass
class MetricResult:
    metric_name: str
    category: MetricCategory
    score: float | None
    real_value: Any
    generated_value: Any
    distance_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
