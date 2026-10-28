from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.recommender.contracts import (
    Dataset,
    IdMap,
    ItemFeatures,
    Recommendation,
    RecommendationMode,
    RecommendationRequest,
    RecommendationResult,
)


def test_id_map_roundtrip():
    id_map = IdMap.from_ids([3, 1, 2], [20, 10])
    assert id_map.n_users == 3
    assert id_map.n_items == 2
    assert id_map.user_to_internal(1) == 1
    assert id_map.user_to_external(1) == 1
    assert id_map.item_to_internal(20) == 0
    assert id_map.item_to_external(0) == 20
    with pytest.raises(KeyError):
        id_map.user_to_internal(999)


def test_item_features_shape():
    features = ItemFeatures(
        item_ids=(1, 2),
        feature_names=("a", "b"),
        matrix=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert features.n_items == 2
    assert features.n_features == 2
    assert features.vector(2).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        ItemFeatures((1, 2), ("a",), np.zeros((2, 2)))


def test_dataset_rejects_duplicate_pairs(tiny_dataset):
    ratings = pd.concat([tiny_dataset.ratings, tiny_dataset.ratings.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        Dataset(
            name="bad",
            ratings=ratings,
            id_map=tiny_dataset.id_map,
            item_features=tiny_dataset.item_features,
        )


def test_dataset_rejects_unknown_ids(tiny_dataset):
    ratings = tiny_dataset.ratings.copy()
    ratings.loc[0, "user_id"] = 999
    with pytest.raises(ValueError, match="unknown users"):
        Dataset(name="bad", ratings=ratings, id_map=tiny_dataset.id_map)


def test_request_validation():
    with pytest.raises(ValueError, match="PERSONALIZED"):
        RecommendationRequest(mode=RecommendationMode.PERSONALIZED)
    with pytest.raises(ValueError, match="SIMILAR"):
        RecommendationRequest(mode=RecommendationMode.SIMILAR)
    with pytest.raises(ValueError, match="duplicates"):
        RecommendationRequest(
            mode=RecommendationMode.TRENDING,
            candidate_item_ids=(1, 1),
        )
    request = RecommendationRequest(
        mode=RecommendationMode.PERSONALIZED,
        user_id=1,
        top_k=3,
        exclude_item_ids={2},
    )
    assert request.exclude_item_ids == frozenset({2})


def test_recommendation_result_validation():
    request = RecommendationRequest(
        mode=RecommendationMode.PERSONALIZED,
        user_id=1,
        top_k=2,
    )
    items = (
        Recommendation(item_id=10, score=3.0, rank=1),
        Recommendation(item_id=11, score=2.0, rank=2),
    )
    result = RecommendationResult(
        request=request,
        items=items,
        model_name="test",
        model_version="1",
    )
    assert result.item_ids == (10, 11)
    with pytest.raises(ValueError, match="sorted"):
        RecommendationResult(
            request=request,
            items=(
                Recommendation(item_id=10, score=1.0, rank=1),
                Recommendation(item_id=11, score=2.0, rank=2),
            ),
            model_name="test",
            model_version="1",
        )
    with pytest.raises(ValueError, match="excluded"):
        RecommendationResult(
            request=RecommendationRequest(
                mode=RecommendationMode.TRENDING,
                exclude_item_ids={10},
            ),
            items=(Recommendation(item_id=10, score=1.0, rank=1),),
            model_name="test",
            model_version="1",
        )


def test_dataset_split_overlap_is_rejected(tiny_dataset):
    from backend.app.recommender.contracts import DatasetSplit

    with pytest.raises(ValueError, match="overlapping"):
        DatasetSplit(
            train=tiny_dataset,
            validation=None,
            test=tiny_dataset,
            protocol="temporal",
            seed=1,
        )
