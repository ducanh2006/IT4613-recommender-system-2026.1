from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from backend.app.recommender.contracts import (
    Dataset,
    ItemFeatures,
    RecommendationRequest,
    RecommendationMode,
    Recommender,
)


@dataclass(frozen=True)
class EvaluationResult:
    per_user: pd.DataFrame
    samples: pd.DataFrame
    global_metrics: dict[str, float]


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    values = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    values = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(values)))


def precision_at_k(
    recommended: Sequence[Hashable],
    relevant: set[Hashable],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(
    recommended: Sequence[Hashable],
    relevant: set[Hashable],
    k: int,
) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(
    recommended: Sequence[Hashable],
    relevant: set[Hashable],
    k: int,
) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def mrr_at_k(
    recommended: Sequence[Hashable],
    relevant: set[Hashable],
    k: int,
) -> float:
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    recommended: Sequence[Hashable],
    relevant: set[Hashable],
    k: int,
) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    denominator = min(len(relevant), k)
    return precision_sum / denominator if denominator > 0 else 0.0


def ndcg_at_k(
    recommended: Sequence[Hashable],
    relevance: Mapping[Hashable, float],
    k: int,
) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        gain = float(relevance.get(item, 0.0))
        if gain > 0.0:
            dcg += (2.0**gain - 1.0) / math.log2(rank + 1.0)
    ideal_gains = sorted(
        (float(value) for value in relevance.values() if value > 0.0),
        reverse=True,
    )[:k]
    idcg = sum(
        (2.0**gain - 1.0) / math.log2(rank + 1.0)
        for rank, gain in enumerate(ideal_gains, start=1)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_rating(
    model: Recommender,
    test: Dataset,
    split_name: str = "test",
) -> EvaluationResult:
    rows = []
    prediction_rows = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    for user_id, group in test.ratings.groupby("user_id", sort=True):
        actual = []
        predicted = []
        for row in group.itertuples(index=False):
            start = time.perf_counter()
            prediction = float(model.predict(row.user_id, row.item_id))
            latency_ms = (time.perf_counter() - start) * 1000.0
            actual.append(float(row.rating))
            predicted.append(prediction)
            actual_values.append(float(row.rating))
            predicted_values.append(prediction)
            prediction_rows.append(
                {
                    "split": split_name,
                    "model": model.name,
                    "user_id": user_id,
                    "item_id": row.item_id,
                    "actual": float(row.rating),
                    "predicted": prediction,
                    "latency_ms": latency_ms,
                }
            )
        rows.append(
            {
                "split": split_name,
                "model": model.name,
                "user_id": user_id,
                "rmse": rmse(actual, predicted),
                "mae": mae(actual, predicted),
                "n_ratings": len(actual),
            }
        )
    per_user = pd.DataFrame(rows)
    samples = pd.DataFrame(prediction_rows)
    metrics = {
        "rmse": rmse(actual_values, predicted_values) if actual_values else float("nan"),
        "mae": mae(actual_values, predicted_values) if actual_values else float("nan"),
        "n_ratings": float(len(actual_values)),
        "n_users": float(len(per_user)),
    }
    return EvaluationResult(per_user=per_user, samples=samples, global_metrics=metrics)


def evaluate_ranking(
    model: Recommender,
    test: Dataset,
    train: Dataset,
    k: int = 10,
    relevance_threshold: float = 4.0,
    catalog_item_ids: Iterable[Hashable] | None = None,
    negative_samples: int | None = None,
    seed: int = 42,
    split_name: str = "test",
) -> EvaluationResult:
    if catalog_item_ids is None:
        catalog = set(train.items)
    else:
        catalog = set(catalog_item_ids)
    seen_by_user = {
        key: set(value) for key, value in train.user_items().items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    sample_rows = []
    covered_items: set[Hashable] = set()

    for user_id, group in test.ratings.groupby("user_id", sort=True):
        seen = seen_by_user.get(user_id, set())
        test_items = set(group["item_id"].tolist())
        relevant = {
            row.item_id
            for row in group.itertuples(index=False)
            if float(row.rating) >= relevance_threshold
        }
        candidate_items = catalog.difference(seen)
        if not relevant:
            continue
        if negative_samples is not None:
            negatives = list(candidate_items.difference(relevant))
            sample_size = min(negative_samples, len(negatives))
            sampled = (
                rng.choice(negatives, size=sample_size, replace=False).tolist()
                if sample_size > 0
                else []
            )
            candidates = sorted(relevant.union(sampled), key=str)
        else:
            candidates = sorted(candidate_items, key=str)
        request = RecommendationRequest(
            mode=RecommendationMode.PERSONALIZED,
            user_id=user_id,
            top_k=k,
            exclude_item_ids=frozenset(seen),
            candidate_item_ids=tuple(candidates),
        )
        start = time.perf_counter()
        result = model.recommend(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        recommended = [item.item_id for item in result.items]
        covered_items.update(recommended)
        grades = {
            row.item_id: max(float(row.rating) - 3.0, 0.0)
            for row in group.itertuples(index=False)
        }
        binary_relevant = set(relevant)
        metrics = {
            "precision_at_k": precision_at_k(recommended, binary_relevant, k),
            "recall_at_k": recall_at_k(recommended, binary_relevant, k),
            "ndcg_at_k": ndcg_at_k(recommended, grades, k),
            "map_at_k": average_precision_at_k(recommended, binary_relevant, k),
            "hit_rate_at_k": hit_rate_at_k(recommended, binary_relevant, k),
            "mrr_at_k": mrr_at_k(recommended, binary_relevant, k),
        }
        rows.append(
            {
                "split": split_name,
                "model": model.name,
                "user_id": user_id,
                **metrics,
                "n_relevant": len(binary_relevant),
                "n_candidates": len(candidates),
                "n_recommended": len(recommended),
                "fallback_used": bool(result.fallback_used),
                "latency_ms": latency_ms,
            }
        )
        for rank, item in enumerate(result.items, start=1):
            sample_rows.append(
                {
                    "split": split_name,
                    "model": model.name,
                    "user_id": user_id,
                    "rank": rank,
                    "item_id": item.item_id,
                    "score": item.score,
                    "relevant": item.item_id in binary_relevant,
                    "test_rating": (
                        float(group.loc[group["item_id"] == item.item_id, "rating"].iloc[0])
                        if item.item_id in test_items
                        else None
                    ),
                    "fallback_used": bool(result.fallback_used),
                }
            )

    per_user = pd.DataFrame(rows)
    samples = pd.DataFrame(sample_rows)
    if per_user.empty:
        metrics = {
            "precision_at_k": float("nan"),
            "recall_at_k": float("nan"),
            "ndcg_at_k": float("nan"),
            "map_at_k": float("nan"),
            "hit_rate_at_k": float("nan"),
            "mrr_at_k": float("nan"),
            "n_users": 0.0,
            "n_recommended": 0.0,
            "empty_rate": float("nan"),
            "fallback_rate": float("nan"),
            "coverage": 0.0,
        }
    else:
        metrics = {
            "precision_at_k": float(per_user["precision_at_k"].mean()),
            "recall_at_k": float(per_user["recall_at_k"].mean()),
            "ndcg_at_k": float(per_user["ndcg_at_k"].mean()),
            "map_at_k": float(per_user["map_at_k"].mean()),
            "hit_rate_at_k": float(per_user["hit_rate_at_k"].mean()),
            "mrr_at_k": float(per_user["mrr_at_k"].mean()),
            "n_users": float(len(per_user)),
            "n_recommended": float(per_user["n_recommended"].sum()),
            "empty_rate": float((per_user["n_recommended"] == 0).mean()),
            "fallback_rate": float(per_user["fallback_used"].mean()),
            "coverage": len(covered_items) / len(catalog) if catalog else 0.0,
        }
    return EvaluationResult(per_user=per_user, samples=samples, global_metrics=metrics)


def intra_list_diversity(
    recommended_lists: Iterable[Sequence[Hashable]],
    item_features: ItemFeatures | None,
) -> float:
    if item_features is None or item_features.n_features == 0:
        return 0.0
    lookup = {item_id: idx for idx, item_id in enumerate(item_features.item_ids)}
    values = []
    for items in recommended_lists:
        vectors = [
            item_features.matrix[lookup[item]]
            for item in items
            if item in lookup
        ]
        if len(vectors) < 2:
            values.append(0.0)
            continue
        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
        similarity = matrix @ matrix.T
        upper = similarity[np.triu_indices(len(matrix), k=1)]
        values.append(float(1.0 - upper.mean()) if len(upper) else 0.0)
    return float(np.mean(values)) if values else 0.0


def novelty(
    recommended_lists: Iterable[Sequence[Hashable]],
    train_item_counts: Mapping[Hashable, int],
    n_users: int,
) -> float:
    if n_users <= 0:
        return 0.0
    scores = []
    for items in recommended_lists:
        for item in items:
            count = max(int(train_item_counts.get(item, 0)), 1)
            scores.append(-math.log2(count / n_users))
    return float(np.mean(scores)) if scores else 0.0
