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
    make_capabilities,
    popularity_stats,
    rank_scores,
    resolve_candidate_items,
)


class GlobalMeanRecommender(BaseRecommender):
    name = "global_mean"
    version = "1.0.0"
    capabilities = make_capabilities(supports_rating_prediction=True)

    def __init__(self) -> None:
        super().__init__()
        self.global_mean = 0.0
        self.item_means: dict[Hashable, float] = {}
        self.counts: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "GlobalMeanRecommender":
        super().fit(data, validation)
        stats = popularity_stats(data)
        self.global_mean = stats.global_mean
        self.item_means = stats.means
        self.counts = dict(stats.counts)
        self.known_items = tuple(stats.counts)
        return self

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = {
            item_id: self.item_means.get(item_id, self.global_mean)
            for item_id in candidates
        }
        ranked = rank_scores(scores, request, tie_breakers=self.counts)
        return build_result(self.name, self.version, request, ranked, reason="global_mean")

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        return float(self.item_means.get(item_id, self.global_mean))
