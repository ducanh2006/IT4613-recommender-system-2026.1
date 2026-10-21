from __future__ import annotations

from typing import Hashable

import numpy as np

from ..contracts import (
    Dataset,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
    RecommenderCapabilities,
    UnknownItemError,
    UnsupportedOperation,
)
from ._base import (
    build_result,
    item_popularity_scores,
    popularity_stats,
    rank_scores,
    resolve_candidate_items,
    user_seen_items,
)


class ContentBasedRecommender:
    name = "content_based"
    version = "1.0.0"
    capabilities = RecommenderCapabilities(
        requires_training=True,
        supports_cold_start=True,
        supports_rating_prediction=False,
        supports_incremental_update=False,
        supported_modes=frozenset(
            {RecommendationMode.PERSONALIZED, RecommendationMode.SIMILAR}
        ),
    )

    def __init__(self) -> None:
        self.feature_matrix = np.zeros((0, 0), dtype=np.float32)
        self.user_profiles = np.zeros((0, 0), dtype=np.float32)
        self.item_ids: tuple[Hashable, ...] = ()
        self.known_items: tuple[Hashable, ...] = ()
        self.fallback_items: tuple[Hashable, ...] = ()
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}
        self.popularity_scores: dict[Hashable, float] = {}
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "ContentBasedRecommender":
        id_map = data.id_map
        features = data.item_features
        self.seen_items = user_seen_items(data)
        self.user_index = dict(id_map.user_to_idx)
        self.item_index = dict(id_map.item_to_idx)
        stats = popularity_stats(data)
        self.fallback_items = tuple(stats.counts)
        self.popularity_scores = item_popularity_scores(
            stats,
            fallback_m=50.0,
        )
        if features is None:
            self.feature_matrix = np.zeros((id_map.n_items, 0), dtype=np.float32)
            self.user_profiles = np.zeros((id_map.n_users, 0), dtype=np.float32)
            self.item_ids = id_map.item_ids
            self.known_items = self.fallback_items
            return self

        feature_lookup = {
            item_id: idx for idx, item_id in enumerate(features.item_ids)
        }
        matrix = np.zeros(
            (id_map.n_items, features.n_features),
            dtype=np.float32,
        )
        for item_id, item_idx in id_map.item_to_idx.items():
            feature_idx = feature_lookup.get(item_id)
            if feature_idx is not None:
                matrix[item_idx] = features.matrix[feature_idx]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms

        profiles = np.zeros((id_map.n_users, features.n_features), dtype=np.float32)
        user_idx = data.ratings["user_id"].map(id_map.user_to_idx).to_numpy(dtype=int)
        item_idx = data.ratings["item_id"].map(id_map.item_to_idx).to_numpy(dtype=int)
        weights = np.maximum(
            data.ratings["rating"].to_numpy(dtype=np.float32) - 3.0,
            0.1,
        )
        np.add.at(profiles, user_idx, matrix[item_idx] * weights[:, None])
        profile_norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        profile_norms[profile_norms == 0.0] = 1.0
        profiles = profiles / profile_norms

        self.feature_matrix = matrix
        self.user_profiles = profiles
        self.item_ids = id_map.item_ids
        self.known_items = tuple(
            item_id for item_id in id_map.item_ids if item_id in feature_lookup
        )
        return self

    def _popularity_fallback(
        self,
        request: RecommendationRequest,
        reason: str,
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

    def _scores_from_vector(
        self,
        vector: np.ndarray,
        request: RecommendationRequest,
    ) -> dict[Hashable, float]:
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores: dict[Hashable, float] = {}
        for item_id in candidates:
            idx = self.item_index[item_id]
            score = float(vector @ self.feature_matrix[idx])
            scores[item_id] = score
        return scores

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(
                f"content_based does not support {request.mode.value}"
            )
        if request.mode is RecommendationMode.SIMILAR:
            if self.feature_matrix.shape[1] == 0:
                raise UnsupportedOperation("content_based has no item features")
            if request.item_id not in self.known_items:
                raise UnknownItemError(f"Unknown item: {request.item_id!r}")
            target = self.feature_matrix[self.item_index[request.item_id]]
            scores = self._scores_from_vector(target, request)
            ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
            return build_result(self.name, self.version, request, ranked, reason="similarity")

        if request.user_id not in self.seen_items:
            return self._popularity_fallback(request, "popular_fallback")
        user_idx = self._user_index(request.user_id)
        profile = self.user_profiles[user_idx]
        if not np.any(profile):
            return self._popularity_fallback(request, "popular_fallback")
        scores = self._scores_from_vector(profile, request)
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(
            self.name,
            self.version,
            request,
            ranked,
            reason="content_profile",
        )

    def _user_index(self, user_id: Hashable) -> int:
        return self.user_index[user_id]

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        raise UnsupportedOperation("content_based does not support rating prediction")
