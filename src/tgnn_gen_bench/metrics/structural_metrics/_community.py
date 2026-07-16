from __future__ import annotations

# import packages

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# community helpers


def detect_communities(
    graph: nx.Graph,
    *,
    resolution: float = 1.0,
    seed: int = 42,
    weight: str = "weight",
) -> list[set[int]]:
    """Detect communities in one snapshot graph."""
    if graph.number_of_nodes() == 0:
        return []

    if graph.number_of_edges() == 0:
        return [{int(node)} for node in graph.nodes()]

    communities = nx.community.louvain_communities(
        graph,
        weight=weight,
        resolution=resolution,
        seed=seed,
    )
    return [set(map(int, community)) for community in communities]


def snapshot_louvain_modularity(
    graph: nx.Graph,
    *,
    resolution: float = 1.0,
    seed: int = 42,
    weight: str = "weight",
) -> tuple[float, list[set[int]]]:
    """Compute Louvain modularity and its detected communities."""
    communities = detect_communities(
        graph,
        resolution=resolution,
        seed=seed,
        weight=weight,
    )
    if graph.number_of_edges() == 0:
        return 0.0, communities

    modularity = nx.community.modularity(
        graph,
        communities,
        weight=weight,
        resolution=resolution,
    )
    return float(modularity), communities


def community_transition_statistics(
    graph_t: nx.Graph,
    graph_t1: nx.Graph,
    *,
    overlap_threshold: float = 0.3,
    resolution: float = 1.0,
    seed: int = 42,
    weight: str = "weight",
) -> dict[str, object]:
    """Summarize community births, deaths, merges, and splits."""
    communities_t = detect_communities(
        graph_t,
        resolution=resolution,
        seed=seed,
        weight=weight,
    )
    communities_t1 = detect_communities(
        graph_t1,
        resolution=resolution,
        seed=seed,
        weight=weight,
    )

    overlap = community_overlap_matrix(communities_t, communities_t1)
    matches = match_communities(overlap)
    matched_scores = [score for _, _, score in matches if score >= overlap_threshold]
    mean_persistence = float(np.mean(matched_scores)) if matched_scores else 0.0

    significant_overlap = overlap >= overlap_threshold
    if significant_overlap.size == 0:
        previous_match_counts = np.zeros(len(communities_t), dtype=int)
        next_match_counts = np.zeros(len(communities_t1), dtype=int)
    else:
        previous_match_counts = significant_overlap.sum(axis=1)
        next_match_counts = significant_overlap.sum(axis=0)

    return {
        "mean_persistence": mean_persistence,
        "births": int(np.sum(next_match_counts == 0)),
        "deaths": int(np.sum(previous_match_counts == 0)),
        "splits": int(np.sum(previous_match_counts > 1)),
        "merges": int(np.sum(next_match_counts > 1)),
    }


def community_event_distribution(statistics: dict[str, object]) -> np.ndarray:
    """Normalize community event counts into a distribution."""
    counts = np.asarray(
        [
            statistics["births"],
            statistics["deaths"],
            statistics["splits"],
            statistics["merges"],
        ],
        dtype=float,
    )
    if counts.sum() == 0:
        return np.zeros(4, dtype=float)
    return counts / counts.sum()


def community_overlap_matrix(
    communities_t: list[set[int]],
    communities_t1: list[set[int]],
) -> np.ndarray:
    """Compute pairwise Jaccard overlaps between communities."""
    matrix = np.zeros((len(communities_t), len(communities_t1)), dtype=float)
    for row, community_t in enumerate(communities_t):
        for column, community_t1 in enumerate(communities_t1):
            matrix[row, column] = jaccard_similarity(community_t, community_t1)
    return matrix


def match_communities(overlap: np.ndarray) -> list[tuple[int, int, float]]:
    """Match communities across snapshots by maximum overlap."""
    if overlap.size == 0:
        return []

    rows, columns = linear_sum_assignment(-overlap)
    return [
        (int(row), int(column), float(overlap[row, column]))
        for row, column in zip(rows, columns)
    ]


def jaccard_similarity(set_a: set[int], set_b: set[int]) -> float:
    """Compute the Jaccard similarity between two node sets."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


__all__ = [
    "community_event_distribution",
    "community_transition_statistics",
    "detect_communities",
    "snapshot_louvain_modularity",
]
