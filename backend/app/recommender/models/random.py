from __future__ import annotations

import zlib
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
    rank_scores,
    resolve_candidate_items,
    user_seen_items,
)


class RandomRecommender:
    name = "random"
    version = "1.0.0"
    capabilities = RecommenderCapabilities(
        requires_training=False,
        supports_cold_start=True,
        supports_rating_prediction=False,
        supports_incremental_update=False,
        supported_modes=frozenset(
            {RecommendationMode.PERSONALIZED, RecommendationMode.TRENDING}
        ),
    )

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.known_items: tuple[Hashable, ...] = ()
        self.seen_items: dict[Hashable, frozenset[Hashable]] = {}

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "RandomRecommender":
        self.known_items = data.id_map.item_ids
        self.seen_items = user_seen_items(data)
        return self

    def _scores(
        self,
        request: RecommendationRequest,
        candidates: tuple[Hashable, ...],
    ) -> dict[Hashable, float]:
        if request.user_id is None:
            salt = "trending"
        else:
            salt = repr(request.user_id)
        seed = self.seed + zlib.crc32(salt.encode("utf-8"))
        rng = np.random.default_rng(seed)
        values = rng.random(len(candidates))
        return {item_id: float(value) for item_id, value in zip(candidates, values)}

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        if request.mode not in self.capabilities.supported_modes:
            raise UnsupportedOperation(f"random does not support {request.mode.value}")
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = self._scores(request, candidates)
        ranked = rank_scores(scores, request)
        return build_result(
            self.name,
            self.version,
            request,
            ranked,
            reason="random",
            fallback_used=False,
        )

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        raise UnsupportedOperation("random does not support rating prediction")
