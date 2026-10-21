from __future__ import annotations

from typing import Hashable

import numpy as np

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
    item_popularity_scores,
    min_max_normalize,
    popularity_stats,
    rank_scores,
    resolve_candidate_items,
    user_seen_items,
)
from .content_based import ContentBasedRecommender
from .funk_svd import FunkSVDRecommender
from .item_knn import ItemKNNRecommender


class HybridRecommender:
    name = "hybrid"
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

    def __init__(
        self,
        content_weight: float = 0.2,
        item_knn_weight: float = 0.4,
        svd_weight: float = 0.4,
        seed: int = 42,
    ) -> None:
        self.content_weight = float(content_weight)
        self.item_knn_weight = float(item_knn_weight)
        self.svd_weight = float(svd_weight)
        self.seed = int(seed)
        self.content = ContentBasedRecommender()
        self.item_knn = ItemKNNRecommender()
        self.svd = FunkSVDRecommender(seed=self.seed)
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}
        self.popularity_scores: dict[Hashable, float] = {}
        self.known_items: tuple[Hashable, ...] = ()
        self.item_index: dict[Hashable, int] = {}
        self.user_index: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "HybridRecommender":
        self.content.fit(data)
        self.item_knn.fit(data)
        self.svd.fit(data, validation)
        self.seen_items = user_seen_items(data)
        self.known_items = tuple(data.id_map.item_ids)
        self.item_index = dict(data.id_map.item_to_idx)
        self.user_index = dict(data.id_map.user_to_idx)
        stats = popularity_stats(data)
        self.popularity_scores = item_popularity_scores(
            stats,
            fallback_m=50.0,
        )
        return self

    def _component_scores(
        self,
        user_id: Hashable,
        candidates: tuple[Hashable, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        content_scores = np.zeros(len(candidates), dtype=float)
        item_scores = np.zeros(len(candidates), dtype=float)
        svd_scores = np.zeros(len(candidates), dtype=float)

        content_user = user_id in self.content.seen_items
        if content_user:
            user_idx = self.content._user_index(user_id)
            profile = self.content.user_profiles[user_idx]
            if np.any(profile):
                for idx, item_id in enumerate(candidates):
                    if item_id in self.content.item_index:
                        content_scores[idx] = float(
                            profile @ self.content.feature_matrix[
                                self.content.item_index[item_id]
                            ]
                        )

        item_user_idx = self.item_knn.user_index.get(user_id)
        if item_user_idx is not None and self.item_knn.mask[item_user_idx].any():
            item_ranking = self.item_knn._ranking_scores(item_user_idx)
            for idx, item_id in enumerate(candidates):
                item_idx = self.item_knn.item_index.get(item_id)
                if item_idx is not None:
                    item_scores[idx] = float(item_ranking[item_idx])

        svd_user_idx = self.svd.user_index.get(user_id)
        if svd_user_idx is not None:
            svd_array = self.svd._score_user(svd_user_idx)
            for idx, item_id in enumerate(candidates):
                item_idx = self.svd.item_index.get(item_id)
                if item_idx is not None:
                    svd_scores[idx] = float(svd_array[item_idx])
        return content_scores, item_scores, svd_scores

    def _popularity_fallback(
        self,
        request: RecommendationRequest,
        reason: str = "popular_fallback",
    ) -> RecommendationResult:
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
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

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(f"hybrid does not support {request.mode.value}")
        if request.mode is RecommendationMode.TRENDING or request.user_id not in self.seen_items:
            return self._popularity_fallback(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        if not candidates:
            return build_result(self.name, self.version, request, [])
        content_scores, item_scores, svd_scores = self._component_scores(
            request.user_id, candidates
        )
        weights = [self.item_knn_weight, self.svd_weight, self.content_weight]
        arrays = [item_scores, svd_scores, content_scores]
        valid_weights = []
        normalized = []
        for weight, values in zip(weights, arrays):
            if weight <= 0.0:
                continue
            if np.all(values == 0.0):
                continue
            normalized.append(min_max_normalize(values))
            valid_weights.append(weight)
        if not normalized:
            return self._popularity_fallback(request)
        total_weight = float(sum(valid_weights))
        combined = np.zeros(len(candidates), dtype=float)
        for weight, values in zip(valid_weights, normalized):
            combined += (weight / total_weight) * values
        scores = {
            item_id: float(combined[idx])
            for idx, item_id in enumerate(candidates)
        }
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(self.name, self.version, request, ranked, reason="hybrid")

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        svd_prediction = self.svd.predict(user_id, item_id)
        item_prediction = self.item_knn.predict(user_id, item_id)
        weight_sum = self.svd_weight + self.item_knn_weight
        if weight_sum <= 0.0:
            return float(svd_prediction)
        return float(
            (
                self.svd_weight * svd_prediction
                + self.item_knn_weight * item_prediction
            )
            / weight_sum
        )
