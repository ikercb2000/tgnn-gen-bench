"""Public radar plotting API."""

# import modules

from tgnn_gen_bench.report._radar_config import (
    CATEGORY_COLOUR,
    CATEGORY_LABEL,
    CATEGORY_MARKER,
    CATEGORY_ORDER,
)
from tgnn_gen_bench.report._radar_figures import by_category, by_metric
from tgnn_gen_bench.report._radar_helpers import group_by_category, polygon_area
from tgnn_gen_bench.report._radar_plotter import RadarPlotter, radar_plotter

__all__ = [
    "CATEGORY_COLOUR",
    "CATEGORY_LABEL",
    "CATEGORY_MARKER",
    "CATEGORY_ORDER",
    "RadarPlotter",
    "by_category",
    "by_metric",
    "group_by_category",
    "polygon_area",
    "radar_plotter",
]
