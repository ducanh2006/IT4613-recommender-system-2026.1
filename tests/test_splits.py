from __future__ import annotations

import pandas as pd

from evaluation.splits import build_split, split_stats
from tests.conftest import make_dataset


def pair_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(map(tuple, frame[["user_id", "item_id"]].to_numpy().tolist()))


def test_temporal_split_is_disjoint_and_filtered():
    dataset = make_dataset(20, 20, 15)
    split = build_split(dataset, "temporal", min_user_ratings=5, min_item_ratings=5)
    assert not pair_set(split.train.ratings).intersection(pair_set(split.test.ratings))
    assert split.validation is not None
    assert not pair_set(split.train.ratings).intersection(
        pair_set(split.validation.ratings)
    )
    assert split.test.n_users > 1
    assert split.test.n_ratings > 0
    train_counts = split.train.ratings.groupby("user_id").size()
    assert train_counts.min() >= 5
    stats = split_stats(split)
    assert stats["protocol"] == "temporal"
    assert stats["train_ratings"] == split.train.n_ratings


def test_leave_one_out_has_one_test_per_eligible_user():
    dataset = make_dataset(12, 20, 10)
    split = build_split(dataset, "leave_one_out", min_user_ratings=5, min_item_ratings=5)
    per_user_test = split.test.ratings.groupby("user_id").size()
    assert (per_user_test == 1).all()
    assert split.test.n_users == split.test.n_ratings
    split_pairs = pair_set(split.train.ratings).intersection(pair_set(split.test.ratings))
    assert not split_pairs


def test_split_respects_train_only_catalog():
    dataset = make_dataset(20, 20, 15)
    split = build_split(dataset, "temporal", min_user_ratings=5, min_item_ratings=8)
    train_item_counts = split.train.ratings.groupby("item_id").size()
    assert train_item_counts.min() >= 8
    test_item_counts = split.test.ratings.groupby("item_id").size()
    assert test_item_counts.index.isin(train_item_counts.index).all()


def test_temporal_split_covers_every_eligible_user():
    dataset = make_dataset(20, 20, 15)
    split = build_split(dataset, "temporal", min_user_ratings=5, min_item_ratings=5)
    eligible = set(dataset.ratings["user_id"].unique())
    assert set(split.test.ratings["user_id"]) == eligible
    assert set(split.validation.ratings["user_id"]) == eligible
    train_times = split.train.ratings.groupby("user_id")["timestamp"].max()
    test_times = split.test.ratings.groupby("user_id")["timestamp"].min()
    assert (train_times <= test_times).all()


def test_cold_user_holdout_produces_cold_start_split():
    dataset = make_dataset(20, 20, 15)
    split = build_split(
        dataset,
        "temporal",
        min_user_ratings=5,
        min_item_ratings=5,
        cold_user_ratio=0.2,
    )
    assert split.cold_start_test is not None
    cold_users = set(split.cold_start_test.ratings["user_id"])
    assert cold_users
    assert not cold_users & set(split.train.ratings["user_id"])
    assert not cold_users & set(split.test.ratings["user_id"])


def test_leave_one_out_cold_user_holdout():
    dataset = make_dataset(20, 20, 15)
    split = build_split(
        dataset,
        "leave_one_out",
        min_user_ratings=5,
        min_item_ratings=5,
        cold_user_ratio=0.2,
    )
    assert split.cold_start_test is not None
    cold_users = set(split.cold_start_test.ratings["user_id"])
    assert not cold_users & set(split.train.ratings["user_id"])
    per_user_test = split.test.ratings.groupby("user_id").size()
    assert (per_user_test == 1).all()
