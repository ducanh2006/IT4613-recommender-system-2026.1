from __future__ import annotations

from typing import Hashable

from ..contracts import (
    Dataset,
    RecommendationRequest,
    RecommendationResult,
)
from ._base import (
    BaseRecommender,
    build_result,
    item_popularity_scores,
    make_capabilities,
    popularity_stats,
    rank_scores,
    resolve_candidate_items,
)


class MostPopularRecommender(BaseRecommender):
    name = "most_popular"
    version = "1.0.0"
    capabilities = make_capabilities()

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[Hashable, int] = {}
        self.means: dict[Hashable, float] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "MostPopularRecommender":
        super().fit(data, validation)
        stats = popularity_stats(data)
        self.counts = dict(stats.counts)
        self.means = dict(stats.means)
        self.known_items = tuple(stats.counts)
        return self

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = {
            item_id: float(self.counts.get(item_id, 0))
            + float(self.means.get(item_id, 0.0)) / 1_000_000.0
            for item_id in candidates
        }
        ranked = rank_scores(scores, request, tie_breakers=self.counts)
        return build_result(self.name, self.version, request, ranked, reason="popularity")


class PopularityRecommender(BaseRecommender):
    name = "popularity"
    version = "1.0.0"
    capabilities = make_capabilities()

    def __init__(self, m: float = 50.0) -> None:
        super().__init__()
        self.m = float(m)
        self.counts: dict[Hashable, int] = {}
        self.global_mean = 0.0
        self.scores: dict[Hashable, float] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "PopularityRecommender":
        super().fit(data, validation)
        stats = popularity_stats(data)
        self.counts = dict(stats.counts)
        self.global_mean = stats.global_mean
        self.scores = item_popularity_scores(stats, fallback_m=self.m)
        self.known_items = tuple(stats.counts)
        return self

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = {
            item_id: self.scores.get(item_id, self.global_mean)
            for item_id in candidates
        }
        ranked = rank_scores(scores, request, tie_breakers=self.counts)
        return build_result(
            self.name,
            self.version,
            request,
            ranked,
            reason="weighted_popularity",
        )
