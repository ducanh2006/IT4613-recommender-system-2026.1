from __future__ import annotations

from typing import Hashable

import numpy as np

from ..contracts import (
    Dataset,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
    UnknownItemError,
    UnsupportedOperation,
)
from ._base import (
    BaseRecommender,
    build_result,
    make_capabilities,
    rank_scores,
    resolve_candidate_items,
)


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"
    version = "1.0.0"
    capabilities = make_capabilities(
        supported_modes=(RecommendationMode.PERSONALIZED, RecommendationMode.SIMILAR)
    )

    def __init__(self) -> None:
        super().__init__()
        self.feature_matrix = np.zeros((0, 0), dtype=np.float32)
        self.user_profiles = np.zeros((0, 0), dtype=np.float32)
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "ContentBasedRecommender":
        super().fit(data, validation)
        self._fit_popularity(data)
        id_map = data.id_map
        features = data.item_features
        self.user_index = dict(id_map.user_to_idx)
        self.item_index = dict(id_map.item_to_idx)
        if features is None:
            self.feature_matrix = np.zeros((id_map.n_items, 0), dtype=np.float32)
            self.user_profiles = np.zeros((id_map.n_users, 0), dtype=np.float32)
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
        self.known_items = tuple(
            item_id for item_id in id_map.item_ids if item_id in feature_lookup
        )
        return self

    def _scores_from_vector(
        self,
        vector: np.ndarray,
        request: RecommendationRequest,
    ) -> dict[Hashable, float]:
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores: dict[Hashable, float] = {}
        for item_id in candidates:
            idx = self.item_index[item_id]
            scores[item_id] = float(vector @ self.feature_matrix[idx])
        return scores

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        self._check_mode(request)
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
            return self._popularity_fallback(request)
        user_idx = self.user_index[request.user_id]
        profile = self.user_profiles[user_idx]
        if not np.any(profile):
            return self._popularity_fallback(request)
        scores = self._scores_from_vector(profile, request)
        ranked = rank_scores(scores, request, tie_breakers=self.popularity_scores)
        return build_result(
            self.name,
            self.version,
            request,
            ranked,
            reason="content_profile",
        )
