from __future__ import annotations

# import packages

from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# import modules

from tgnn_gen_bench.graphs import Graph

# feature names

FEATURE_NAMES = [
    "seen_before",
    "log_pair_count",
    "log_pair_count_recent",
    "recency",
    "log_source_activity",
    "log_destination_popularity",
    "log_source_activity_recent",
    "log_destination_popularity_recent",
]

# temporal link prediction helpers


def relative_temporal_edges(
    graph: Graph,
    *,
    directed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract graph edges and map their timestamps to relative days."""
    edge_time = graph.edge_time.detach().cpu().long().numpy().astype(np.int64, copy=False)
    edge_index = np.column_stack(
        [
            graph.edge_src.detach().cpu().long().numpy(),
            graph.edge_dst.detach().cpu().long().numpy(),
        ]
    ).astype(np.int64, copy=False)

    if not directed:
        reversed_edges = edge_index[:, [1, 0]]
        events = np.column_stack([edge_time, edge_index])
        reversed_events = np.column_stack([edge_time, reversed_edges])
        merged_events = np.vstack([events, reversed_events])
        merged_events = np.unique(merged_events, axis=0)
        order = np.argsort(merged_events[:, 0], kind="stable")
        merged_events = merged_events[order]
        edge_time = merged_events[:, 0]
        edge_index = merged_events[:, 1:]

    timestamps, relative_days = np.unique(edge_time, return_inverse=True)
    return edge_index, relative_days.astype(np.int64), timestamps.astype(np.int64)


def sample_negative_pairs(
    positive_pairs: np.ndarray,
    destination_nodes: np.ndarray,
    negative_ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample destination negatives for each positive source node."""
    positive_set = {(int(source), int(destination)) for source, destination in positive_pairs}
    negative_pairs: list[tuple[int, int]] = []

    for source, _ in positive_pairs:
        source = int(source)
        chosen_destinations: set[int] = set()

        for _ in range(negative_ratio):
            for _attempt in range(1000):
                destination = int(rng.choice(destination_nodes))
                if (source, destination) in positive_set:
                    continue
                if destination in chosen_destinations:
                    continue

                negative_pairs.append((source, destination))
                chosen_destinations.add(destination)
                break
            else:
                raise RuntimeError("Unable to sample a negative destination.")

    return np.asarray(negative_pairs, dtype=np.int64)


def build_causal_examples(
    graph: Graph,
    number_of_days: int,
    *,
    directed: bool,
    negative_ratio: int = 1,
    recent_window: int = 7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct causal trainable examples from one temporal graph."""
    edges, edge_days, _ = relative_temporal_edges(graph, directed=directed)
    if edges.size == 0:
        raise ValueError("No examples could be generated.")

    destination_nodes = np.unique(edges[:, 1])
    rng = np.random.default_rng(seed)
    num_nodes = int(graph.num_nodes)

    pair_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    pair_last_seen: dict[tuple[int, int], int] = {}
    source_activity = np.zeros(num_nodes, dtype=np.int64)
    destination_popularity = np.zeros(num_nodes, dtype=np.int64)

    pair_recent_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    source_recent_activity = np.zeros(num_nodes, dtype=np.int64)
    destination_recent_popularity = np.zeros(num_nodes, dtype=np.int64)

    recent_days: deque[tuple[int, np.ndarray]] = deque()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_days: list[np.ndarray] = []

    for day in range(number_of_days):
        while recent_days and recent_days[0][0] < day - recent_window:
            _, old_pairs = recent_days.popleft()
            for source, destination in old_pairs:
                source = int(source)
                destination = int(destination)
                pair = (source, destination)

                source_recent_activity[source] -= 1
                destination_recent_popularity[destination] -= 1
                pair_recent_count[pair] -= 1
                if pair_recent_count[pair] == 0:
                    del pair_recent_count[pair]

        positive_pairs = np.unique(edges[edge_days == day], axis=0)
        if len(positive_pairs) == 0:
            continue

        negative_pairs = sample_negative_pairs(
            positive_pairs=positive_pairs,
            destination_nodes=destination_nodes,
            negative_ratio=negative_ratio,
            rng=rng,
        )
        candidate_pairs = np.vstack([positive_pairs, negative_pairs])
        labels = np.concatenate(
            [
                np.ones(len(positive_pairs), dtype=np.int8),
                np.zeros(len(negative_pairs), dtype=np.int8),
            ]
        )

        sources = candidate_pairs[:, 0]
        destinations = candidate_pairs[:, 1]
        pair_keys = [(int(source), int(destination)) for source, destination in candidate_pairs]

        cumulative_pair_count = np.fromiter(
            (pair_count.get(pair, 0) for pair in pair_keys),
            dtype=float,
            count=len(pair_keys),
        )
        recent_pair_count = np.fromiter(
            (pair_recent_count.get(pair, 0) for pair in pair_keys),
            dtype=float,
            count=len(pair_keys),
        )
        recency = np.fromiter(
            (
                0.0 if pair not in pair_last_seen else 1.0 / (day - pair_last_seen[pair])
                for pair in pair_keys
            ),
            dtype=float,
            count=len(pair_keys),
        )

        features = np.column_stack(
            [
                (cumulative_pair_count > 0).astype(float),
                np.log1p(cumulative_pair_count),
                np.log1p(recent_pair_count),
                recency,
                np.log1p(source_activity[sources]),
                np.log1p(destination_popularity[destinations]),
                np.log1p(source_recent_activity[sources]),
                np.log1p(destination_recent_popularity[destinations]),
            ]
        )

        all_features.append(features)
        all_labels.append(labels)
        all_days.append(np.full(len(labels), day, dtype=np.int64))

        for source, destination in positive_pairs:
            source = int(source)
            destination = int(destination)
            pair = (source, destination)

            pair_count[pair] += 1
            pair_recent_count[pair] += 1
            pair_last_seen[pair] = day

            source_activity[source] += 1
            destination_popularity[destination] += 1
            source_recent_activity[source] += 1
            destination_recent_popularity[destination] += 1

        recent_days.append((day, positive_pairs))

    if not all_features:
        raise ValueError("No examples could be generated.")

    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_days)


def evaluate_temporal_classifier(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    days: np.ndarray,
    test_days: np.ndarray,
) -> dict[str, object]:
    """Evaluate one fitted classifier on a set of temporal examples."""
    probabilities = model.predict_proba(features)[:, 1]
    per_day: dict[int, dict[str, float | int]] = {}

    for day in test_days:
        mask = days == day
        if not mask.any():
            raise ValueError(f"No examples found for test day {day}.")

        per_day[int(day)] = {
            "log_loss": float(log_loss(labels[mask], probabilities[mask], labels=[0, 1])),
            "roc_auc": float(roc_auc_score(labels[mask], probabilities[mask])),
            "n_examples": int(mask.sum()),
            "n_positive": int(labels[mask].sum()),
        }

    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "macro_log_loss": float(np.mean([result["log_loss"] for result in per_day.values()])),
        "macro_roc_auc": float(np.mean([result["roc_auc"] for result in per_day.values()])),
        "per_day": per_day,
    }


def temporal_link_prediction_results(
    graph: Graph,
    *,
    directed: bool = True,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
    negative_ratio: int = 1,
    recent_window: int = 7,
    c_grid: Sequence[float] = (0.1, 1.0, 10.0),
    seed: int = 42,
) -> dict[str, object]:
    """Train and evaluate a causal temporal link prediction baseline."""
    _, _, timestamps = relative_temporal_edges(graph, directed=directed)
    number_of_days = len(timestamps)

    if not 0 < train_ratio < 1:
        raise ValueError("Invalid train_ratio.")
    if not 0 < validation_ratio < 1:
        raise ValueError("Invalid validation_ratio.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be < 1.")
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be >= 1.")

    train_end = int(number_of_days * train_ratio)
    validation_end = train_end + int(number_of_days * validation_ratio)
    if train_end < 1 or validation_end <= train_end or validation_end >= number_of_days:
        raise ValueError("Not enough snapshots for train/validation/test.")

    features, labels, days = build_causal_examples(
        graph,
        number_of_days,
        directed=directed,
        negative_ratio=negative_ratio,
        recent_window=recent_window,
        seed=seed,
    )

    train_mask = days < train_end
    validation_mask = (days >= train_end) & (days < validation_end)
    test_mask = days >= validation_end

    best_c = None
    best_validation_loss = np.inf
    for c_value in c_grid:
        candidate_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c_value),
                solver="liblinear",
                max_iter=500,
                random_state=seed,
            ),
        )
        candidate_model.fit(features[train_mask], labels[train_mask])
        validation_probabilities = candidate_model.predict_proba(features[validation_mask])[:, 1]
        validation_loss = float(
            log_loss(labels[validation_mask], validation_probabilities, labels=[0, 1])
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_c = float(c_value)

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=best_c,
            solver="liblinear",
            max_iter=500,
            random_state=seed,
        ),
    )
    final_model.fit(features[train_mask | validation_mask], labels[train_mask | validation_mask])

    test_days = np.arange(validation_end, number_of_days, dtype=np.int64)
    test_results = evaluate_temporal_classifier(
        model=final_model,
        features=features[test_mask],
        labels=labels[test_mask],
        days=days[test_mask],
        test_days=test_days,
    )
    return {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "best_c": best_c,
        "validation_log_loss": best_validation_loss,
        "split": {
            "train_days": (0, train_end - 1),
            "validation_days": (train_end, validation_end - 1),
            "test_days": (validation_end, number_of_days - 1),
        },
        "test": test_results,
    }


__all__ = ["FEATURE_NAMES", "temporal_link_prediction_results"]
