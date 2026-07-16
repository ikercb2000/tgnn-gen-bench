# import modules

from tgnn_gen_bench.metrics import MetricCategory
from tgnn_gen_bench.metrics import (
    DurationOfContacts,
    RandomWalkEntropy,
    TemporalLinkPredictionLogLoss,
    TemporalReachability,
    TemporalStrengthDistribution,
)

# metric category enum tests


def test_metric_category_is_string_enum() -> None:
    assert str(MetricCategory.TEMPORAL_FIDELITY) == "temporal_fidelity"
    assert MetricCategory.TEMPORAL_FIDELITY == "temporal_fidelity"


def test_metrics_expose_typed_categories() -> None:
    assert DurationOfContacts.category is MetricCategory.TEMPORAL_FIDELITY
    assert TemporalReachability.category is MetricCategory.GLOBAL_METRICS
    assert TemporalStrengthDistribution.category is MetricCategory.STRUCTURAL_METRICS
    assert RandomWalkEntropy.category is MetricCategory.DYNAMICS_METRICS
    assert TemporalLinkPredictionLogLoss.category is MetricCategory.DOWNSTREAM_TASKS
