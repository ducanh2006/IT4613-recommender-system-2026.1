from __future__ import annotations

from typing import Hashable

import numpy as np

from ..contracts import (
    Dataset,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
)
from ._base import (
    BaseRecommender,
    build_result,
    make_capabilities,
    min_max_normalize,
    rank_scores,
    resolve_candidate_items,
)
from .content_based import ContentBasedRecommender
from .funk_svd import FunkSVDRecommender
from .item_knn import ItemKNNRecommender


class HybridRecommender(BaseRecommender):
    name = "hybrid"
    version = "1.0.0"
    capabilities = make_capabilities(supports_rating_prediction=True)

    def __init__(
        self,
        content_weight: float = 0.2,
        item_knn_weight: float = 0.4,
        svd_weight: float = 0.4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.content_weight = float(content_weight)
        self.item_knn_weight = float(item_knn_weight)
        self.svd_weight = float(svd_weight)
        self.seed = int(seed)
        self.content = ContentBasedRecommender()
        self.item_knn = ItemKNNRecommender()
        self.svd = FunkSVDRecommender(seed=self.seed)

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "HybridRecommender":
        super().fit(data, validation)
        self._fit_popularity(data)
        self.content.fit(data)
        self.item_knn.fit(data)
        self.svd.fit(data, validation)
        self.known_items = tuple(data.id_map.item_ids)
        return self

    def _component_scores(
        self,
        user_id: Hashable,
        candidates: tuple[Hashable, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        content_scores = np.zeros(len(candidates), dtype=float)
        item_scores = np.zeros(len(candidates), dtype=float)
        svd_scores = np.zeros(len(candidates), dtype=float)

        content_user_idx = self.content.user_index.get(user_id)
        if content_user_idx is not None:
            profile = self.content.user_profiles[content_user_idx]
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

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
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
