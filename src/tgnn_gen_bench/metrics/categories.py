from __future__ import annotations

# import packages

from enum import StrEnum

# metric categories


class MetricCategory(StrEnum):
    """Enumerate the metric families used across the project."""

    TEMPORAL_FIDELITY = "temporal_fidelity"
    GLOBAL_METRICS = "global_metrics"
    STRUCTURAL_METRICS = "structural_metrics"
    DYNAMICS_METRICS = "dynamics_metrics"
    DOWNSTREAM_TASKS = "downstream_tasks"
    DYNAMICS = "dynamics"


__all__ = ["MetricCategory"]
