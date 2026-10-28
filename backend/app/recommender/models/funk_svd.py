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
    build_rating_matrix,
    build_result,
    candidate_scores,
    make_capabilities,
    rank_scores,
    resolve_candidate_items,
)


class FunkSVDRecommender(BaseRecommender):
    name = "funk_svd"
    version = "1.0.0"
    capabilities = make_capabilities(supports_rating_prediction=True)

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 30,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        patience: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_factors = int(n_factors)
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.patience = int(patience)
        self.seed = int(seed)
        self.global_mean = 0.0
        self.user_bias = np.zeros(0, dtype=np.float64)
        self.item_bias = np.zeros(0, dtype=np.float64)
        self.user_factors = np.zeros((0, 0), dtype=np.float64)
        self.item_factors = np.zeros((0, 0), dtype=np.float64)
        self.user_ids: tuple[Hashable, ...] = ()
        self.item_ids: tuple[Hashable, ...] = ()
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "FunkSVDRecommender":
        super().fit(data, validation)
        self._fit_popularity(data)
        rating_matrix = build_rating_matrix(data)
        values = rating_matrix.values
        user_idx, item_idx = np.where(~np.isnan(values))
        ratings = values[user_idx, item_idx].astype(np.float64)
        self.global_mean = float(ratings.mean())
        self.user_ids = rating_matrix.user_ids
        self.item_ids = rating_matrix.item_ids
        self.user_index = {value: idx for idx, value in enumerate(self.user_ids)}
        self.item_index = {value: idx for idx, value in enumerate(self.item_ids)}
        self.known_items = tuple(self.item_ids)

        rng = np.random.default_rng(self.seed)
        self.user_bias = np.zeros(len(self.user_ids), dtype=np.float64)
        self.item_bias = np.zeros(len(self.item_ids), dtype=np.float64)
        self.user_factors = rng.normal(0.0, 0.05, (len(self.user_ids), self.n_factors))
        self.item_factors = rng.normal(0.0, 0.05, (len(self.item_ids), self.n_factors))

        best_rmse = float("inf")
        best_state: tuple[np.ndarray, ...] | None = None
        stale_epochs = 0
        indices = np.arange(len(ratings))

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(indices)
            for idx in indices:
                user = int(user_idx[idx])
                item = int(item_idx[idx])
                rating = float(ratings[idx])
                user_vector = self.user_factors[user]
                item_vector = self.item_factors[item]
                prediction = (
                    self.global_mean
                    + self.user_bias[user]
                    + self.item_bias[item]
                    + float(np.dot(user_vector, item_vector))
                )
                error = rating - prediction
                self.user_bias[user] += self.learning_rate * (
                    error - self.regularization * self.user_bias[user]
                )
                self.item_bias[item] += self.learning_rate * (
                    error - self.regularization * self.item_bias[item]
                )
                old_user_vector = user_vector.copy()
                self.user_factors[user] += self.learning_rate * (
                    error * item_vector - self.regularization * user_vector
                )
                self.item_factors[item] += self.learning_rate * (
                    error * old_user_vector - self.regularization * item_vector
                )

            if validation is not None and not validation.ratings.empty:
                rmse = self._validation_rmse(validation)
                if rmse + 1e-8 < best_rmse:
                    best_rmse = rmse
                    best_state = (
                        self.user_bias.copy(),
                        self.item_bias.copy(),
                        self.user_factors.copy(),
                        self.item_factors.copy(),
                    )
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.patience:
                        break

        if best_state is not None:
            (
                self.user_bias,
                self.item_bias,
                self.user_factors,
                self.item_factors,
            ) = best_state
        return self

    def _validation_rmse(self, validation: Dataset) -> float:
        errors = []
        for row in validation.ratings.itertuples(index=False):
            prediction = self.predict(row.user_id, row.item_id)
            errors.append((row.rating - prediction) ** 2)
        return float(np.sqrt(np.mean(errors)))

    def _predict_internal(self, user_idx: int, item_idx: int) -> float:
        prediction = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        )
        return float(prediction)

    def _score_user(self, user_idx: int) -> np.ndarray:
        scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ self.user_factors[user_idx]
        )
        return scores.astype(np.float32)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
        if request.mode is RecommendationMode.TRENDING:
            return self._popularity_fallback(request, "trending_fallback")
        if request.user_id not in self.user_index:
            return self._popularity_fallback(request)
        user_idx = self.user_index[request.user_id]
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = candidate_scores(
            self._score_user(user_idx), candidates, self.item_index
        )
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(
            self.name, self.version, request, ranked, reason="matrix_factorization"
        )

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        user_idx = self.user_index.get(user_id)
        item_idx = self.item_index.get(item_id)
        if user_idx is None and item_idx is None:
            return float(self.global_mean)
        if user_idx is None:
            return float(self.global_mean + self.item_bias[item_idx])
        if item_idx is None:
            return float(self.global_mean + self.user_bias[user_idx])
        return self._predict_internal(user_idx, item_idx)
