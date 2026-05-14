from __future__ import annotations

import os

import numpy as np

TREE_CONFIDENCE_DEFAULT = 0.99
TREE_SAMPLE_SEED_DEFAULT = 123


def quantize_vectors(vectors: np.ndarray) -> np.ndarray:
    clipped = np.clip(vectors, -1.0, 1.0)
    return np.rint((clipped + 1.0) * 127.5).astype(np.uint8)


def quantize_vector(vector: np.ndarray) -> np.ndarray:
    clipped = np.clip(vector, -1.0, 1.0)
    return np.rint((clipped + 1.0) * 127.5).astype(np.uint8)


def train_confident_tree(
    vectors: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    if vectors.shape[0] == 0:
        return {
            "scores": np.array([0.5], dtype=np.float32),
            "features": np.array([-1], dtype=np.int16),
            "thresholds": np.array([0.0], dtype=np.float32),
            "left": np.array([-1], dtype=np.int32),
            "right": np.array([-1], dtype=np.int32),
        }

    tree_seed = int(os.getenv("RINHA_TREE_SEED", str(TREE_SAMPLE_SEED_DEFAULT)))
    rng = np.random.default_rng(tree_seed)
    sample_size = min(vectors.shape[0], int(os.getenv("RINHA_TREE_SAMPLE", "500000")))
    sample_ids = rng.choice(vectors.shape[0], size=sample_size, replace=False)
    sample_vectors = vectors[sample_ids]
    sample_labels = labels[sample_ids]
    max_depth = int(os.getenv("RINHA_TREE_DEPTH", "10"))
    min_leaf = int(os.getenv("RINHA_TREE_MIN_LEAF", "50"))
    quantiles = int(os.getenv("RINHA_TREE_QUANTILES", "199"))
    candidate_thresholds = [
        np.unique(
            np.quantile(
                sample_vectors[:, feature], np.linspace(0.01, 0.99, quantiles)
            ).astype(np.float32)
        )
        for feature in range(sample_vectors.shape[1])
    ]

    nodes: list[list[float | int]] = []

    def best_split(
        ids: np.ndarray,
    ) -> tuple[int | None, float, np.ndarray | None, np.ndarray | None]:
        node_labels = sample_labels[ids]
        total = int(node_labels.sum())
        count = ids.shape[0]
        base = total * (count - total) / count
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold = 0.0
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None

        for feature, thresholds in enumerate(candidate_thresholds):
            values = sample_vectors[ids, feature]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_labels = node_labels[order]
            cumulative = np.cumsum(sorted_labels)
            positions = np.searchsorted(sorted_values, thresholds, side="right")

            for position, threshold in zip(positions, thresholds):
                right_count = count - position
                if position < min_leaf or right_count < min_leaf:
                    continue

                left_positive = int(cumulative[position - 1])
                right_positive = total - left_positive
                left_impurity = left_positive * (position - left_positive) / position
                right_impurity = (
                    right_positive * (right_count - right_positive) / right_count
                )
                gain = base - left_impurity - right_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_left = ids[order[:position]]
                    best_right = ids[order[position:]]

        return best_feature, best_threshold, best_left, best_right

    def build(ids: np.ndarray, depth: int) -> int:
        node_id = len(nodes)
        score = float(sample_labels[ids].mean()) if ids.shape[0] else 0.5
        nodes.append([score, -1, 0.0, -1, -1])

        if (
            depth >= max_depth
            or ids.shape[0] < min_leaf * 2
            or score == 0.0
            or score == 1.0
        ):
            return node_id

        feature, threshold, left_ids, right_ids = best_split(ids)
        if feature is None or left_ids is None or right_ids is None:
            return node_id

        left_node = build(left_ids, depth + 1)
        right_node = build(right_ids, depth + 1)
        nodes[node_id] = [score, feature, threshold, left_node, right_node]
        return node_id

    build(np.arange(sample_vectors.shape[0], dtype=np.int32), 0)
    raw = np.asarray(nodes, dtype=np.float32)
    return {
        "scores": raw[:, 0].astype(np.float32),
        "features": raw[:, 1].astype(np.int16),
        "thresholds": raw[:, 2].astype(np.float32),
        "left": raw[:, 3].astype(np.int32),
        "right": raw[:, 4].astype(np.int32),
    }


def tree_score(
    query: np.ndarray,
    tree: dict[str, np.ndarray] | None,
    confidence: float = TREE_CONFIDENCE_DEFAULT,
) -> float | None:
    if tree is None:
        return None

    features = tree["features"]
    thresholds = tree["thresholds"]
    scores = tree["scores"]
    left = tree["left"]
    right = tree["right"]

    node = 0
    while features[node] >= 0:
        feature = int(features[node])
        node = (
            int(left[node]) if query[feature] <= thresholds[node] else int(right[node])
        )

    score = float(scores[node])
    if score <= 1.0 - confidence or score >= confidence:
        return score
    return None
