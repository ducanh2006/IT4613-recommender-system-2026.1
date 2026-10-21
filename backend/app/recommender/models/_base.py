from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np

from ..contracts import Dataset, Recommendation, RecommendationRequest, RecommendationResult


@dataclass(frozen=True)
class RatingMatrix:
    values: np.ndarray
    user_ids: tuple[Hashable, ...]
    item_ids: tuple[Hashable, ...]

    @property
    def user_mask(self) -> np.ndarray:
        return ~np.isnan(self.values)


@dataclass(frozen=True)
class PopularityStats:
    counts: Mapping[Hashable, int]
    means: Mapping[Hashable, float]
    global_mean: float


def build_rating_matrix(dataset: Dataset) -> RatingMatrix:
    id_map = dataset.id_map
    values = np.full((id_map.n_users, id_map.n_items), np.nan, dtype=np.float64)
    user_idx = dataset.ratings["user_id"].map(id_map.user_to_idx).to_numpy(dtype=int)
    item_idx = dataset.ratings["item_id"].map(id_map.item_to_idx).to_numpy(dtype=int)
    values[user_idx, item_idx] = dataset.ratings["rating"].to_numpy(dtype=float)
    return RatingMatrix(
        values=values,
        user_ids=id_map.idx_to_user,
        item_ids=id_map.idx_to_item,
    )


def popularity_stats(dataset: Dataset) -> PopularityStats:
    grouped = dataset.ratings.groupby("item_id")["rating"]
    counts = grouped.size()
    means = grouped.mean()
    return PopularityStats(
        counts={key: int(value) for key, value in counts.items()},
        means={key: float(value) for key, value in means.items()},
        global_mean=float(dataset.ratings["rating"].mean()),
    )


def user_seen_items(dataset: Dataset) -> dict[Hashable, frozenset[Hashable]]:
    grouped = dataset.ratings.groupby("user_id", sort=False)["item_id"].apply(frozenset)
    return {key: value for key, value in grouped.items()}


def resolve_candidate_items(
    request: RecommendationRequest,
    known_items: Iterable[Hashable],
    seen_items: Mapping[Hashable, frozenset[Hashable]] | None = None,
) -> tuple[Hashable, ...]:
    known = set(known_items)
    if request.candidate_item_ids is None:
        candidates = known
    else:
        candidates = known.intersection(request.candidate_item_ids)
    excluded = set(request.exclude_item_ids)
    if request.user_id is not None and seen_items is not None:
        excluded.update(seen_items.get(request.user_id, frozenset()))
    if request.mode.value == "similar" and request.item_id is not None:
        excluded.add(request.item_id)
    return tuple(item for item in candidates if item not in excluded)


def rank_scores(
    scores: Mapping[Hashable, float],
    request: RecommendationRequest,
    top_k: int | None = None,
    tie_breakers: Mapping[Hashable, float] | None = None,
) -> list[tuple[Hashable, float]]:
    limit = request.top_k if top_k is None else top_k
    filtered = [
        (item_id, float(score))
        for item_id, score in scores.items()
        if np.isfinite(score) and item_id not in request.exclude_item_ids
    ]
    if tie_breakers:
        filtered.sort(
            key=lambda pair: (
                -pair[1],
                -float(tie_breakers.get(pair[0], 0.0)),
                str(pair[0]),
            )
        )
    else:
        filtered.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return filtered[:limit]


def build_result(
    model_name: str,
    model_version: str,
    request: RecommendationRequest,
    ranked: Iterable[tuple[Hashable, float]],
    reason: str | None = None,
    fallback_used: bool = False,
    latency_ms: float | None = None,
) -> RecommendationResult:
    items = tuple(
        Recommendation(
            item_id=item_id,
            score=score,
            rank=rank,
            reason=reason,
        )
        for rank, (item_id, score) in enumerate(ranked, start=1)
    )
    return RecommendationResult(
        request=request,
        items=items,
        model_name=model_name,
        model_version=model_version,
        fallback_used=fallback_used,
        latency_ms=latency_ms,
    )


def item_popularity_scores(
    stats: PopularityStats,
    fallback_m: float = 0.0,
) -> dict[Hashable, float]:
    scores: dict[Hashable, float] = {}
    for item_id, count in stats.counts.items():
        mean = stats.means.get(item_id, stats.global_mean)
        if fallback_m <= 0:
            scores[item_id] = float(count)
        else:
            scores[item_id] = (count / (count + fallback_m)) * mean + (
                fallback_m / (count + fallback_m)
            ) * stats.global_mean
    return scores


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high <= low:
        return np.where(finite, 1.0, 0.0)
    result = (values - low) / (high - low)
    result[~finite] = 0.0
    return result
