from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np

from ..contracts import (
    Dataset,
    Recommendation,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
    RecommenderCapabilities,
    UnsupportedOperation,
)

POPULARITY_FALLBACK_M = 50.0


@dataclass(frozen=True)
class RatingMatrix:
    values: np.ndarray
    user_ids: tuple[Hashable, ...]
    item_ids: tuple[Hashable, ...]


@dataclass(frozen=True)
class PopularityStats:
    counts: Mapping[Hashable, int]
    means: Mapping[Hashable, float]
    global_mean: float


def make_capabilities(
    supports_rating_prediction: bool = False,
    supported_modes: tuple[RecommendationMode, ...] = (
        RecommendationMode.PERSONALIZED,
        RecommendationMode.TRENDING,
    ),
    *,
    requires_training: bool = True,
    supports_cold_start: bool = True,
    supports_incremental_update: bool = False,
) -> RecommenderCapabilities:
    return RecommenderCapabilities(
        requires_training=requires_training,
        supports_cold_start=supports_cold_start,
        supports_rating_prediction=supports_rating_prediction,
        supports_incremental_update=supports_incremental_update,
        supported_modes=frozenset(supported_modes),
    )


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
    if request.mode is RecommendationMode.SIMILAR and request.item_id is not None:
        excluded.add(request.item_id)
    return tuple(item for item in candidates if item not in excluded)


def rank_scores(
    scores: Mapping[Hashable, float],
    request: RecommendationRequest,
    tie_breakers: Mapping[Hashable, float] | None = None,
) -> list[tuple[Hashable, float]]:
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
    return filtered[: request.top_k]


def build_result(
    model_name: str,
    model_version: str,
    request: RecommendationRequest,
    ranked: Iterable[tuple[Hashable, float]],
    reason: str | None = None,
    fallback_used: bool = False,
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
    )


def item_popularity_scores(
    stats: PopularityStats,
    fallback_m: float = POPULARITY_FALLBACK_M,
) -> dict[Hashable, float]:
    """IMDB style weighted rating: item mean shrunk towards the global mean."""
    scores: dict[Hashable, float] = {}
    for item_id, count in stats.counts.items():
        mean = stats.means.get(item_id, stats.global_mean)
        scores[item_id] = (count / (count + fallback_m)) * mean + (
            fallback_m / (count + fallback_m)
        ) * stats.global_mean
    return scores


def candidate_scores(
    scores: np.ndarray,
    candidates: Iterable[Hashable],
    item_index: Mapping[Hashable, int],
) -> dict[Hashable, float]:
    return {item_id: float(scores[item_index[item_id]]) for item_id in candidates}


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


class BaseRecommender:
    """Shared plumbing for the recommender implementations."""

    name = ""
    version = ""
    capabilities: RecommenderCapabilities

    def __init__(self) -> None:
        self.known_items: tuple[Hashable, ...] = ()
        self.fallback_items: tuple[Hashable, ...] = ()
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}
        self.popularity_scores: dict[Hashable, float] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "BaseRecommender":
        self.seen_items = data.user_items()
        return self

    def _fit_popularity(self, data: Dataset) -> None:
        stats = popularity_stats(data)
        self.popularity_scores = item_popularity_scores(stats)
        self.fallback_items = tuple(stats.counts)

    def _check_mode(self, request: RecommendationRequest) -> None:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(
                f"{self.name} does not support {request.mode.value}"
            )

    def _popularity_fallback(
        self,
        request: RecommendationRequest,
        reason: str = "popular_fallback",
    ) -> RecommendationResult:
        candidates = resolve_candidate_items(
            request, self.fallback_items, self.seen_items
        )
        scores = {
            item_id: self.popularity_scores.get(item_id, 0.0)
            for item_id in candidates
        }
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(
            self.name,
            self.version,
            request,
            ranked,
            reason=reason,
            fallback_used=True,
        )

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        raise UnsupportedOperation(f"{self.name} does not support rating prediction")
