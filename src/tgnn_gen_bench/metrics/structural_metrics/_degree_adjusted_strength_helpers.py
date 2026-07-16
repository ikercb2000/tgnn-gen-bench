from __future__ import annotations

# import packages

import numpy as np

# degree adjusted strength helpers


def snapshot_degree_adjusted_strengths(graph, eps: float) -> np.ndarray:
    nodes = list(graph.nodes())
    if not nodes:
        return np.asarray([], dtype=float)

    edge_weights = np.asarray(
        [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    if edge_weights.size == 0:
        return np.zeros(len(nodes), dtype=float)
    if not np.all(np.isfinite(edge_weights)):
        raise ValueError("Edge weights must be finite.")

    mean_weight = float(edge_weights.mean())
    std_weight = float(edge_weights.std(ddof=0))
    if std_weight <= eps:
        return np.zeros(len(nodes), dtype=float)

    adjusted_strengths = []
    for node in nodes:
        degree = int(graph.degree(node))
        strength = float(graph.degree(node, weight="weight"))
        if degree == 0:
            adjusted_strengths.append(0.0)
            continue

        expected_strength = degree * mean_weight
        expected_std = np.sqrt(degree) * std_weight
        adjusted_strengths.append((strength - expected_strength) / (expected_std + eps))
    return np.asarray(adjusted_strengths, dtype=float)


__all__ = ["snapshot_degree_adjusted_strengths"]
