from __future__ import annotations

import zlib
from typing import Hashable

import numpy as np

from ..contracts import (
    Dataset,
    RecommendationRequest,
    RecommendationResult,
)
from ._base import (
    BaseRecommender,
    build_result,
    make_capabilities,
    rank_scores,
    resolve_candidate_items,
)


class RandomRecommender(BaseRecommender):
    name = "random"
    version = "1.0.0"
    capabilities = make_capabilities(requires_training=False)

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = int(seed)

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "RandomRecommender":
        super().fit(data, validation)
        self.known_items = data.id_map.item_ids
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
        self._check_mode(request)
        candidates = resolve_candidate_items(request, self.known_items, self.seen_items)
        scores = self._scores(request, candidates)
        ranked = rank_scores(scores, request)
        return build_result(self.name, self.version, request, ranked, reason="random")
