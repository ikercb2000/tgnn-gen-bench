from __future__ import annotations

# import packages

import numpy as np

# random walk helpers


def entropy_signature(
    snapshots: list[dict[int, np.ndarray | object]],
    *,
    num_nodes: int,
    n_samples: int,
    n_walks: int,
    horizon: int,
    stay_probability: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 <= stay_probability <= 1.0:
        raise ValueError("stay_probability must lie in [0, 1].")
    if n_walks < 1:
        raise ValueError("n_walks must be at least 1.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    valid_start_indices = [
        index
        for index in range(len(snapshots) - horizon + 1)
        if snapshots[index]
    ]
    if not valid_start_indices:
        return np.zeros((1, horizon), dtype=float)

    rng = np.random.default_rng(seed)
    curves = np.zeros((n_samples, horizon), dtype=float)

    for sample in range(n_samples):
        start_index = int(rng.choice(valid_start_indices))
        active_nodes = np.asarray(sorted(snapshots[start_index].keys()), dtype=np.int64)
        start_node = int(rng.choice(active_nodes))
        positions = simulate_temporal_random_walk(
            snapshots=snapshots,
            start_node=start_node,
            start_index=start_index,
            horizon=horizon,
            n_walks=n_walks,
            stay_probability=stay_probability,
            seed=int(rng.integers(0, 2**32 - 1)),
        )

        for step in range(horizon):
            _, counts = np.unique(positions[:, step], return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log(probabilities))
            curves[sample, step] = entropy / np.log(num_nodes)

    return curves


def simulate_temporal_random_walk(
    snapshots: list[dict[int, np.ndarray | object]],
    *,
    start_node: int,
    start_index: int,
    horizon: int,
    n_walks: int,
    stay_probability: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    current_nodes = np.full(n_walks, start_node, dtype=np.int64)
    positions = np.empty((n_walks, horizon), dtype=np.int64)

    for step, snapshot_index in enumerate(range(start_index, start_index + horizon)):
        adjacency = snapshots[snapshot_index]
        for walk_id, current_node in enumerate(current_nodes):
            neighbors = adjacency.get(int(current_node))
            if neighbors is None or len(neighbors) == 0:
                continue
            if rng.random() >= stay_probability:
                current_nodes[walk_id] = int(rng.choice(neighbors))
        positions[:, step] = current_nodes

    return positions


__all__ = ["entropy_signature", "simulate_temporal_random_walk"]
