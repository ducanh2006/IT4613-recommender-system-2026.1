from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, runtime_checkable

from .data import Dataset
from .recommendation import (
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
)


class RecommenderError(Exception):
    pass


class UnsupportedOperation(RecommenderError):
    pass


class ColdStartError(RecommenderError):
    pass


class UnknownItemError(RecommenderError):
    pass


class InvalidRecommendationRequest(RecommenderError):
    pass


@dataclass(frozen=True)
class RecommenderCapabilities:
    requires_training: bool
    supports_cold_start: bool
    supports_rating_prediction: bool
    supports_incremental_update: bool
    supported_modes: frozenset[RecommendationMode]


@runtime_checkable
class Recommender(Protocol):
    name: str
    version: str
    capabilities: RecommenderCapabilities

    def fit(
        self,
        data: Dataset,
        validation: Dataset | None = None,
    ) -> "Recommender":
        ...

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        ...

    def predict(self, user_id: Hashable, item_id: Hashable) -> float:
        ...
