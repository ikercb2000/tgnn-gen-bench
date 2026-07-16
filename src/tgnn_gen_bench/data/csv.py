# import packages

from pathlib import Path

import torch

# import modules

from tgm import TimeDeltaDG
from tgnn_gen_bench.graphs import DGData, Graph, DGraph

# loaders


def from_csv(
    path: str | Path,
    src_col: str = "i",
    dst_col: str = "j",
    time_col: str = "t",
    time_unit: str | TimeDeltaDG = "s",
    device: str | torch.device = "cpu",
) -> Graph:
    """Load a time-ordered temporal graph from a CSV edge list."""
    time_delta = TimeDeltaDG(time_unit) if isinstance(time_unit, str) else time_unit
    if time_delta.is_event_ordered:
        raise ValueError(
            f"time_unit must be time-ordered ('s', 'm', 'h', ...), got {time_delta}"
        )

    data = DGData.from_csv(
        edge_file_path=path,
        edge_src_col=src_col,
        edge_dst_col=dst_col,
        edge_time_col=time_col,
        time_delta=time_delta,
    )
    return DGraph(data, device=device)


__all__ = ["from_csv"]
