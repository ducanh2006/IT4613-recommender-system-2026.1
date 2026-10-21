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


class GlobalMeanRecommender:
    name = "global_mean"
    version = "1.0.0"
    capabilities = RecommenderCapabilities(
        requires_training=True,
        supports_cold_start=True,
        supports_rating_prediction=True,
        supports_incremental_update=False,
        supported_modes=frozenset(
            {RecommendationMode.PERSONALIZED, RecommendationMode.TRENDING}
        ),
    )

    def __init__(self) -> None:
        self.global_mean = 0.0
        self.known_items: tuple[Hashable, ...] = ()
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}
        self.item_means: dict[Hashable, float] = {}
        self.counts: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "GlobalMeanRecommender":
        stats = popularity_stats(data)
        self.global_mean = stats.global_mean
        self.item_means = stats.means
        self.counts = dict(stats.counts)
        self.known_items = tuple(stats.counts)
        self.seen_items = user_seen_items(data)
        return self

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(f"global_mean does not support {request.mode.value}")
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = {
            item_id: self.item_means.get(item_id, self.global_mean)
            for item_id in candidates
        }
        ranked = rank_scores(scores, request, tie_breakers=self.counts)
        return build_result(self.name, self.version, request, ranked, reason="global_mean")

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        return float(self.item_means.get(item_id, self.global_mean))
