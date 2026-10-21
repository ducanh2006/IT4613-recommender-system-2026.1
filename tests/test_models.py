from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.recommender.contracts import (
    RecommendationMode,
    RecommendationRequest,
    UnsupportedOperation,
)
from backend.app.recommender.models import (
    ContentBasedRecommender,
    FunkSVDRecommender,
    GlobalMeanRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    PopularityRecommender,
    RandomRecommender,
    UserKNNRecommender,
)
from backend.app.recommender.models._base import rank_scores

MODEL_CLASSES = [
    RandomRecommender,
    GlobalMeanRecommender,
    MostPopularRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
    FunkSVDRecommender,
    HybridRecommender,
]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_models_obey_recommendation_contract(tiny_dataset, model_class):
    model = model_class()
    model.fit(tiny_dataset)
    seen = tiny_dataset.user_items()[1]
    request = RecommendationRequest(
        mode=RecommendationMode.PERSONALIZED,
        user_id=1,
        top_k=3,
        exclude_item_ids=frozenset(seen),
        candidate_item_ids=tiny_dataset.id_map.item_ids,
    )
    result = model.recommend(request)
    assert len(result) <= 3
    assert len(result.item_ids) == len(set(result.item_ids))
    assert not set(result.item_ids).intersection(seen)
    assert [item.rank for item in result] == list(range(1, len(result) + 1))
    assert all(math.isfinite(item.score) for item in result)
    assert result.model_name == model.name


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_models_handle_cold_start(tiny_dataset, model_class):
    model = model_class()
    model.fit(tiny_dataset)
    request = RecommendationRequest(
        mode=RecommendationMode.PERSONALIZED,
        user_id=999,
        top_k=2,
        candidate_item_ids=tiny_dataset.id_map.item_ids,
    )
    result = model.recommend(request)
    assert len(result) <= 2
    assert isinstance(result.fallback_used, bool)


def test_content_based_similar_and_unsupported_rating(tiny_dataset):
    model = ContentBasedRecommender()
    model.fit(tiny_dataset)
    request = RecommendationRequest(
        mode=RecommendationMode.SIMILAR,
        item_id=tiny_dataset.id_map.item_ids[0],
        top_k=3,
    )
    result = model.recommend(request)
    assert result.item_ids
    assert request.item_id not in result.item_ids
    with pytest.raises(UnsupportedOperation):
        model.predict(1, tiny_dataset.id_map.item_ids[0])


def test_rating_prediction_models_return_finite_values(tiny_dataset):
    for model_class in (
        GlobalMeanRecommender,
        UserKNNRecommender,
        ItemKNNRecommender,
        FunkSVDRecommender,
        HybridRecommender,
    ):
        model = model_class()
        model.fit(tiny_dataset)
        prediction = model.predict(1, tiny_dataset.id_map.item_ids[0])
        assert math.isfinite(prediction)


def test_random_and_svd_are_deterministic(tiny_dataset):
    request = RecommendationRequest(
        mode=RecommendationMode.PERSONALIZED,
        user_id=1,
        top_k=3,
        candidate_item_ids=tiny_dataset.id_map.item_ids,
    )
    for model_class in (RandomRecommender, FunkSVDRecommender):
        first = model_class(seed=7)
        second = model_class(seed=7)
        first.fit(tiny_dataset)
        second.fit(tiny_dataset)
        assert first.recommend(request).item_ids == second.recommend(request).item_ids


def test_rank_scores_breaks_ties_with_the_tie_breaker():
    request = RecommendationRequest(mode=RecommendationMode.TRENDING, top_k=3)
    scores = {10: 1.0, 20: 1.0, 30: 1.0}
    plain = rank_scores(scores, request)
    assert [item for item, _ in plain] == [10, 20, 30]
    ranked = rank_scores(
        scores, request, tie_breakers={10: 1.0, 20: 5.0, 30: 3.0}
    )
    assert [item for item, _ in ranked] == [20, 30, 10]


def test_random_never_reports_a_fallback(tiny_dataset):
    model = RandomRecommender(seed=3)
    model.fit(tiny_dataset)
    request = RecommendationRequest(
        mode=RecommendationMode.PERSONALIZED,
        user_id=999,
        top_k=2,
        candidate_item_ids=tiny_dataset.id_map.item_ids,
    )
    result = model.recommend(request)
    assert result.fallback_used is False


def test_knn_ranking_scores_are_personalized(tiny_dataset):
    for model_class in (UserKNNRecommender, ItemKNNRecommender):
        model = model_class()
        model.fit(tiny_dataset)
        first = model._ranking_scores(model.user_index[1])
        second = model._ranking_scores(model.user_index[2])
        assert first.shape == (tiny_dataset.n_items,)
        assert (first >= 0.0).all()
        assert not np.array_equal(first, second)
        assert first.sum() > 0.0
