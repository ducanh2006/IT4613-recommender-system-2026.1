from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterator


class RecommendationMode(str, Enum):
    PERSONALIZED = "personalized"
    SIMILAR = "similar"
    TRENDING = "trending"


@dataclass(frozen=True)
class RecommendationRequest:
    mode: RecommendationMode
    user_id: Hashable | None = None
    item_id: Hashable | None = None
    top_k: int = 10
    exclude_item_ids: frozenset[Hashable] = frozenset()
    candidate_item_ids: tuple[Hashable, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RecommendationMode):
            object.__setattr__(self, "mode", RecommendationMode(self.mode))
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an int")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.mode is RecommendationMode.PERSONALIZED and self.user_id is None:
            raise ValueError("PERSONALIZED requires user_id")
        if self.mode is RecommendationMode.SIMILAR and self.item_id is None:
            raise ValueError("SIMILAR requires item_id")
        exclude = frozenset(self.exclude_item_ids)
        object.__setattr__(self, "exclude_item_ids", exclude)
        if self.candidate_item_ids is not None:
            candidates = tuple(self.candidate_item_ids)
            if len(set(candidates)) != len(candidates):
                raise ValueError("candidate_item_ids contains duplicates")
            object.__setattr__(self, "candidate_item_ids", candidates)


@dataclass(frozen=True)
class Recommendation:
    item_id: Hashable
    score: float
    rank: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("Recommendation.rank must start at 1")
        if not math.isfinite(float(self.score)):
            raise ValueError("Recommendation.score must be finite")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class RecommendationResult:
    request: RecommendationRequest
    items: tuple[Recommendation, ...]
    model_name: str
    model_version: str
    fallback_used: bool = False

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if len(items) > self.request.top_k:
            raise ValueError("RecommendationResult has more than top_k items")
        item_ids = [item.item_id for item in items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("RecommendationResult contains duplicate items")
        for expected_rank, item in enumerate(items, start=1):
            if item.rank != expected_rank:
                raise ValueError("Recommendation ranks must be contiguous from 1")
            if expected_rank > 1 and items[expected_rank - 2].score < item.score:
                raise ValueError("RecommendationResult must be sorted by score descending")
            if item.item_id in self.request.exclude_item_ids:
                raise ValueError("RecommendationResult contains an excluded item")
            if (
                self.request.candidate_item_ids is not None
                and item.item_id not in self.request.candidate_item_ids
            ):
                raise ValueError("RecommendationResult contains a non-candidate item")
        object.__setattr__(self, "items", items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Recommendation]:
        return iter(self.items)

    @property
    def item_ids(self) -> tuple[Hashable, ...]:
        return tuple(item.item_id for item in self.items)
