from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.recommender.contracts import Dataset
from backend.app.recommender.models import PopularityRecommender
from evaluation.metrics import evaluate_ranking, evaluate_rating
from evaluation.splits import build_split
from tests.conftest import make_dataset


def test_evaluate_ranking_and_rating(tiny_dataset):
    split = build_split(
        make_dataset(20, 20, 15),
        "leave_one_out",
        min_user_ratings=5,
        min_item_ratings=5,
    )
    model = PopularityRecommender()
    model.fit(split.train)
    ranking = evaluate_ranking(model, split.test, split.train, k=3)
    assert not ranking.per_user.empty
    assert (ranking.per_user["n_recommended"] <= 3).all()
    assert all(metric in ranking.per_user for metric in ("ndcg_at_k", "precision_at_k"))
    assert np.isfinite(ranking.global_metrics["ndcg_at_k"])
    assert not ranking.samples.empty
    seen = split.train.user_items()
    for row in ranking.samples.itertuples(index=False):
        assert row.item_id not in seen.get(row.user_id, frozenset())

    global_mean = PopularityRecommender()
    global_mean.fit(split.train)
    with np.testing.assert_raises(Exception):
        evaluate_rating(global_mean, split.test)


def test_cold_start_split_and_evaluation():
    dataset = make_dataset(12, 20, 15)
    cold_users = {11, 12}
    keep = ~dataset.ratings["user_id"].isin(cold_users)
    remaining = dataset.ratings[keep]
    cold_rows = []
    for user_id in cold_users:
        user_rows = dataset.ratings[dataset.ratings["user_id"] == user_id].head(3)
        cold_rows.append(user_rows)
    ratings = pd.concat([remaining, *cold_rows]).sort_values(
        ["timestamp", "user_id", "item_id"], kind="mergesort"
    )
    reduced = Dataset(
        name="cold-tiny",
        ratings=ratings,
        id_map=dataset.id_map,
        item_features=dataset.item_features,
        metadata=dataset.metadata,
    )
    split = build_split(
        reduced,
        "leave_one_out",
        min_user_ratings=5,
        min_item_ratings=5,
    )
    assert split.cold_start_test is not None
    assert split.cold_start_test.n_users == 2
    model = PopularityRecommender()
    model.fit(split.train)
    result = evaluate_ranking(
        model,
        split.cold_start_test,
        split.train,
        k=3,
        relevance_threshold=1.0,
        split_name="cold_start",
    )
    assert not result.per_user.empty
    assert result.global_metrics["fallback_rate"] >= 0.0
