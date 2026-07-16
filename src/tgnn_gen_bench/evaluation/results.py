# import packages

from dataclasses import dataclass, field
from typing import Any

# Metric results base class

@dataclass
class MetricResult:
    metric_name: str
    category: str
    score: float | None
    real_value: Any
    generated_value: Any
    distance_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)