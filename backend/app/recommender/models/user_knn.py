from __future__ import annotations

from typing import Hashable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..contracts import (
    Dataset,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
)
from ._base import (
    BaseRecommender,
    build_rating_matrix,
    build_result,
    candidate_scores,
    make_capabilities,
    rank_scores,
    resolve_candidate_items,
)


class UserKNNRecommender(BaseRecommender):
    name = "user_knn"
    version = "1.0.0"
    capabilities = make_capabilities(supports_rating_prediction=True)

    def __init__(self, k_neighbors: int = 40) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.centered = np.zeros((0, 0), dtype=np.float32)
        self.mask = np.zeros((0, 0), dtype=bool)
        self.user_means = np.zeros(0, dtype=np.float32)
        self.global_mean = 0.0
        self.neighbors: list[tuple[np.ndarray, np.ndarray]] = []
        self.item_ids: tuple[Hashable, ...] = ()
        self.user_ids: tuple[Hashable, ...] = ()
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "UserKNNRecommender":
        super().fit(data, validation)
        self._fit_popularity(data)
        rating_matrix = build_rating_matrix(data)
        values = rating_matrix.values.astype(np.float32)
        mask = ~np.isnan(values)
        user_counts = mask.sum(axis=1)
        self.global_mean = float(np.nanmean(values))
        self.user_means = np.divide(
            np.nansum(values, axis=1),
            user_counts,
            out=np.full(values.shape[0], self.global_mean, dtype=np.float32),
            where=user_counts > 0,
        ).astype(np.float32)
        centered = np.where(mask, values - self.user_means[:, None], 0.0)
        similarity = cosine_similarity(centered)
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < 0.0] = 0.0
        self.centered = centered.astype(np.float32)
        self.mask = mask
        self.user_ids = rating_matrix.user_ids
        self.item_ids = rating_matrix.item_ids
        self.user_index = {value: idx for idx, value in enumerate(self.user_ids)}
        self.item_index = {value: idx for idx, value in enumerate(self.item_ids)}
        self.known_items = tuple(self.item_ids)
        self.neighbors = []
        k = min(self.k_neighbors, max(0, len(self.user_ids) - 1))
        for user_idx in range(len(self.user_ids)):
            weights = similarity[user_idx]
            if k == 0 or not np.any(weights > 0.0):
                self.neighbors.append((np.empty(0, dtype=int), np.empty(0, dtype=np.float32)))
                continue
            threshold = np.partition(weights, -k)[-k]
            candidate_idx = np.flatnonzero(weights >= threshold)
            candidate_idx = candidate_idx[weights[candidate_idx] > 0.0]
            order = np.argsort(-weights[candidate_idx])
            candidate_idx = candidate_idx[order][:k]
            self.neighbors.append(
                (candidate_idx.astype(int), weights[candidate_idx].astype(np.float32))
            )
        return self

    def _ranking_scores(self, user_idx: int) -> np.ndarray:
        scores = np.zeros(len(self.item_ids), dtype=np.float64)
        neighbor_idx, weights = self.neighbors[user_idx]
        if len(neighbor_idx) == 0:
            return scores
        for neighbour_idx, weight in zip(neighbor_idx.tolist(), weights.tolist()):
            rated = np.flatnonzero(self.mask[neighbour_idx])
            if len(rated) == 0:
                continue
            ratings = self.centered[neighbour_idx, rated] + self.user_means[neighbour_idx]
            scores[rated] += float(weight) * ratings
        return scores

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
        if request.mode is RecommendationMode.TRENDING:
            return self._popularity_fallback(request, "trending_fallback")
        if request.user_id not in self.user_index:
            return self._popularity_fallback(request)
        user_idx = self.user_index[request.user_id]
        if len(self.neighbors[user_idx][0]) == 0:
            return self._popularity_fallback(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = candidate_scores(
            self._ranking_scores(user_idx), candidates, self.item_index
        )
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(self.name, self.version, request, ranked, reason="user_knn")

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        user_idx = self.user_index.get(user_id)
        item_idx = self.item_index.get(item_id)
        if user_idx is None or item_idx is None:
            return float(self.global_mean)
        neighbor_idx, weights = self.neighbors[user_idx]
        if len(neighbor_idx) == 0:
            return float(self.user_means[user_idx])
        rated_mask = self.mask[neighbor_idx, item_idx].astype(np.float32)
        if not rated_mask.any():
            return float(self.user_means[user_idx])
        numerator = float(
            np.dot(weights * rated_mask, self.centered[neighbor_idx, item_idx])
        )
        denominator = float(np.dot(weights, rated_mask))
        if denominator <= 0.0:
            return float(self.user_means[user_idx])
        return float(self.user_means[user_idx] + numerator / denominator)
