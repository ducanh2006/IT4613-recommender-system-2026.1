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


class ItemKNNRecommender(BaseRecommender):
    name = "item_knn"
    version = "1.0.0"
    capabilities = make_capabilities(supports_rating_prediction=True)

    def __init__(self, k_neighbors: int = 40) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.centered = np.zeros((0, 0), dtype=np.float32)
        self.mask = np.zeros((0, 0), dtype=bool)
        self.item_means = np.zeros(0, dtype=np.float32)
        self.global_mean = 0.0
        self.neighbors: list[tuple[np.ndarray, np.ndarray]] = []
        self.inverse_neighbors: list[tuple[np.ndarray, np.ndarray]] = []
        self.item_ids: tuple[Hashable, ...] = ()
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "ItemKNNRecommender":
        super().fit(data, validation)
        self._fit_popularity(data)
        rating_matrix = build_rating_matrix(data)
        values = rating_matrix.values.astype(np.float32)
        mask = ~np.isnan(values)
        item_counts = mask.sum(axis=0)
        self.global_mean = float(np.nanmean(values))
        self.item_means = np.divide(
            np.nansum(values, axis=0),
            item_counts,
            out=np.full(values.shape[1], self.global_mean, dtype=np.float32),
            where=item_counts > 0,
        ).astype(np.float32)
        centered = np.where(mask, values - self.item_means[None, :], 0.0)
        similarity = cosine_similarity(centered.T)
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < 0.0] = 0.0
        self.centered = centered.astype(np.float32)
        self.mask = mask
        self.item_ids = rating_matrix.item_ids
        self.user_index = {
            value: idx for idx, value in enumerate(rating_matrix.user_ids)
        }
        self.item_index = {value: idx for idx, value in enumerate(self.item_ids)}
        self.known_items = tuple(self.item_ids)
        self.neighbors = []
        inverse: list[list[tuple[int, float]]] = [
            [] for _ in range(len(self.item_ids))
        ]
        k = min(self.k_neighbors, max(0, len(self.item_ids) - 1))
        for item_idx in range(len(self.item_ids)):
            weights = similarity[item_idx]
            if k == 0 or not np.any(weights > 0.0):
                self.neighbors.append((np.empty(0, dtype=int), np.empty(0, dtype=np.float32)))
                continue
            threshold = np.partition(weights, -k)[-k]
            candidate_idx = np.flatnonzero(weights >= threshold)
            candidate_idx = candidate_idx[weights[candidate_idx] > 0.0]
            order = np.argsort(-weights[candidate_idx])
            candidate_idx = candidate_idx[order][:k]
            selected_weights = weights[candidate_idx].astype(np.float32)
            self.neighbors.append((candidate_idx.astype(int), selected_weights))
            for neighbour_idx, weight in zip(candidate_idx.tolist(), selected_weights.tolist()):
                inverse[neighbour_idx].append((item_idx, weight))
        self.inverse_neighbors = [
            (
                np.array([item for item, _ in entries], dtype=int),
                np.array([weight for _, weight in entries], dtype=np.float32),
            )
            for entries in inverse
        ]
        return self

    def _ranking_scores(self, user_idx: int) -> np.ndarray:
        scores = np.zeros(len(self.item_ids), dtype=np.float64)
        rated = np.flatnonzero(self.mask[user_idx])
        if len(rated) == 0:
            return scores
        ratings = self.centered[user_idx, rated] + self.item_means[rated]
        for item_idx, rating in zip(rated.tolist(), ratings.tolist()):
            neighbours, weights = self.inverse_neighbors[item_idx]
            if len(neighbours) == 0:
                continue
            scores[neighbours] += weights * float(rating)
        return scores

    def _score_item(self, user_idx: int, item_idx: int) -> float:
        neighbor_idx, weights = self.neighbors[item_idx]
        if len(neighbor_idx) == 0:
            return float(self.item_means[item_idx])
        rated_mask = self.mask[user_idx, neighbor_idx]
        if not rated_mask.any():
            return float(self.item_means[item_idx])
        active_weights = weights[rated_mask]
        numerator = float(
            np.dot(active_weights, self.centered[user_idx, neighbor_idx[rated_mask]])
        )
        denominator = float(np.abs(active_weights).sum())
        if denominator <= 0.0:
            return float(self.item_means[item_idx])
        return float(self.item_means[item_idx] + numerator / denominator)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
        if request.mode is RecommendationMode.TRENDING:
            return self._popularity_fallback(request, "trending_fallback")
        if request.user_id not in self.user_index:
            return self._popularity_fallback(request)
        user_idx = self.user_index[request.user_id]
        if not self.mask[user_idx].any():
            return self._popularity_fallback(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = candidate_scores(
            self._ranking_scores(user_idx), candidates, self.item_index
        )
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(self.name, self.version, request, ranked, reason="item_knn")

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        user_idx = self.user_index.get(user_id)
        item_idx = self.item_index.get(item_id)
        if user_idx is None or item_idx is None:
            return float(self.global_mean)
        if not self.mask[user_idx].any():
            return float(self.global_mean)
        return self._score_item(user_idx, item_idx)
