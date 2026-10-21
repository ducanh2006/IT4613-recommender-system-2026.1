from __future__ import annotations

from typing import Hashable

from ..contracts import (
    Dataset,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
    RecommenderCapabilities,
    UnsupportedOperation,
)
from ._base import (
    build_result,
    popularity_stats,
    rank_scores,
    resolve_candidate_items,
    user_seen_items,
)


class MostPopularRecommender:
    name = "most_popular"
    version = "1.0.0"
    capabilities = RecommenderCapabilities(
        requires_training=True,
        supports_cold_start=True,
        supports_rating_prediction=False,
        supports_incremental_update=False,
        supported_modes=frozenset(
            {RecommendationMode.PERSONALIZED, RecommendationMode.TRENDING}
        ),
    )

    def __init__(self) -> None:
        self.counts: dict[Hashable, int] = {}
        self.means: dict[Hashable, float] = {}
        self.known_items: tuple[Hashable, ...] = ()
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "MostPopularRecommender":
        stats = popularity_stats(data)
        self.counts = dict(stats.counts)
        self.means = dict(stats.means)
        self.known_items = tuple(stats.counts)
        self.seen_items = user_seen_items(data)
        return self

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(f"most_popular does not support {request.mode.value}")
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = {
            item_id: float(self.counts.get(item_id, 0))
            + float(self.means.get(item_id, 0.0)) / 1_000_000.0
            for item_id in candidates
        }
        ranked = rank_scores(scores, request, tie_breakers=self.counts)
        return build_result(self.name, self.version, request, ranked, reason="popularity")

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        raise UnsupportedOperation("most_popular does not support rating prediction")


class PopularityRecommender:
    name = "popularity"
    version = "1.0.0"
    capabilities = RecommenderCapabilities(
        requires_training=True,
        supports_cold_start=True,
        supports_rating_prediction=False,
        supports_incremental_update=False,
        supported_modes=frozenset(
            {RecommendationMode.PERSONALIZED, RecommendationMode.TRENDING}
        ),
    )

    def __init__(self, m: float = 50.0) -> None:
        self.m = float(m)
        self.counts: dict[Hashable, int] = {}
        self.means: dict[Hashable, float] = {}
        self.global_mean = 0.0
        self.known_items: tuple[Hashable, ...] = ()
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "PopularityRecommender":
        stats = popularity_stats(data)
        self.counts = dict(stats.counts)
        self.means = dict(stats.means)
        self.global_mean = stats.global_mean
        self.known_items = tuple(stats.counts)
        self.seen_items = user_seen_items(data)
        return self

    def _weighted_rating(self, item_id: Hashable) -> float:
        count = float(self.counts.get(item_id, 0))
        mean = float(self.means.get(item_id, self.global_mean))
        if count + self.m <= 0:
            return self.global_mean
        return (count / (count + self.m)) * mean + (
            self.m / (count + self.m)
        ) * self.global_mean

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(f"popularity does not support {request.mode.value}")
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = {item_id: self._weighted_rating(item_id) for item_id in candidates}
        ranked = rank_scores(scores, request, tie_breakers=self.counts)
        return build_result(
            self.name,
            self.version,
            request,
            ranked,
            reason="weighted_popularity",
        )

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        raise UnsupportedOperation("popularity does not support rating prediction")
