from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from backend.app.recommender.contracts import Dataset, DatasetSplit

MIN_RATINGS_PER_USER = 3


def _counts(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(column, sort=False).size()


def _select_cold_users(
    user_ids: Iterable[object],
    ratio: float,
    seed: int,
) -> set[object]:
    if ratio <= 0:
        return set()
    unique = sorted({user_id for user_id in user_ids}, key=str)
    count = int(round(ratio * len(unique)))
    if count <= 0:
        return set()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(unique), size=count, replace=False)
    return {unique[int(index)] for index in indices}


def _filter_splits(
    dataset: Dataset,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    min_user_ratings: int,
    min_item_ratings: int,
    protocol: str,
    seed: int,
) -> DatasetSplit:
    user_counts = _counts(train, "user_id")
    item_counts = _counts(train, "item_id")
    kept_users = set(user_counts[user_counts >= min_user_ratings].index)
    kept_items = set(item_counts[item_counts >= min_item_ratings].index)
    filtered_train = train[
        train["user_id"].isin(kept_users) & train["item_id"].isin(kept_items)
    ]
    filtered_validation = validation[
        validation["user_id"].isin(kept_users) & validation["item_id"].isin(kept_items)
    ]
    filtered_test = test[
        test["user_id"].isin(kept_users) & test["item_id"].isin(kept_items)
    ]
    if filtered_train.empty:
        raise ValueError("Training split is empty after filtering")
    if filtered_test.empty:
        raise ValueError("Test split is empty after filtering")

    cold_users = set(user_counts[user_counts < min_user_ratings].index)
    cold_users.update(set(test["user_id"]).difference(user_counts.index))
    cold_start_test = test[
        test["user_id"].isin(cold_users) & test["item_id"].isin(kept_items)
    ]

    stats = {
        "protocol": protocol,
        "seed": seed,
        "raw_train": int(len(train)),
        "raw_validation": int(len(validation)),
        "raw_test": int(len(test)),
        "train": int(len(filtered_train)),
        "validation": int(len(filtered_validation)),
        "test": int(len(filtered_test)),
        "cold_start_test": int(len(cold_start_test)),
        "train_users": int(filtered_train["user_id"].nunique()),
        "train_items": int(filtered_train["item_id"].nunique()),
        "test_users": int(filtered_test["user_id"].nunique()),
        "test_items": int(filtered_test["item_id"].nunique()),
        "min_user_ratings": min_user_ratings,
        "min_item_ratings": min_item_ratings,
        "dropped_users": int(len(user_counts) - len(kept_users)),
        "dropped_items": int(len(item_counts) - len(kept_items)),
    }
    train_dataset = dataset.subset(filtered_train, name=f"{dataset.name}:train")
    validation_dataset = (
        dataset.subset(filtered_validation, name=f"{dataset.name}:validation")
        if not filtered_validation.empty
        else None
    )
    test_dataset = dataset.subset(filtered_test, name=f"{dataset.name}:test")
    cold_dataset = (
        dataset.subset(cold_start_test, name=f"{dataset.name}:cold-start-test")
        if not cold_start_test.empty
        else None
    )
    train_dataset.metadata.update(stats)
    test_dataset.metadata.update(stats)
    return DatasetSplit(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
        protocol=protocol,
        seed=seed,
        cold_start_test=cold_dataset,
    )


def temporal_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    seed: int = 42,
    cold_user_ratio: float = 0.0,
) -> DatasetSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    test_ratio = 1.0 - train_ratio - validation_ratio
    ratings = dataset.ratings.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = ratings.groupby("user_id", sort=False)["item_id"].transform("size")
    rank_from_end = ratings.groupby("user_id", sort=False).cumcount(ascending=False)
    eligible = sizes >= MIN_RATINGS_PER_USER
    test_size = np.maximum(1, np.ceil(sizes * test_ratio)).astype("int64")
    validation_size = np.minimum(
        np.maximum(1, np.ceil(sizes * validation_ratio)).astype("int64"),
        np.maximum(0, sizes - test_size - 1).astype("int64"),
    )
    is_test = eligible & (rank_from_end < test_size)
    is_validation = eligible & (rank_from_end >= test_size) & (
        rank_from_end < test_size + validation_size
    )
    is_train = eligible & ~is_test & ~is_validation

    cold_users = _select_cold_users(
        ratings.loc[eligible, "user_id"].unique(), cold_user_ratio, seed
    )
    keep = ~ratings["user_id"].isin(cold_users)
    train = ratings[is_train & keep]
    validation = ratings[is_validation & keep]
    test = ratings[is_test]
    return _filter_splits(
        dataset,
        train,
        validation,
        test,
        min_user_ratings,
        min_item_ratings,
        "temporal",
        seed,
    )


def leave_one_out_split(
    dataset: Dataset,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    seed: int = 42,
    cold_user_ratio: float = 0.0,
) -> DatasetSplit:
    ratings = dataset.ratings.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = ratings.groupby("user_id", sort=False)["item_id"].transform("size")
    eligible = ratings[sizes >= MIN_RATINGS_PER_USER].copy()
    eligible["_rank_from_end"] = eligible.groupby("user_id", sort=False).cumcount(
        ascending=False
    )
    train = eligible[eligible["_rank_from_end"] >= 2]
    validation = eligible[eligible["_rank_from_end"] == 1]
    test = eligible[eligible["_rank_from_end"] == 0]

    cold_users = _select_cold_users(
        eligible["user_id"].unique(), cold_user_ratio, seed
    )
    keep = ~train["user_id"].isin(cold_users)
    train = train[keep]
    validation = validation[~validation["user_id"].isin(cold_users)]
    return _filter_splits(
        dataset,
        train,
        validation,
        test,
        min_user_ratings,
        min_item_ratings,
        "leave_one_out",
        seed,
    )


def build_split(
    dataset: Dataset,
    protocol: str,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    seed: int = 42,
    cold_user_ratio: float = 0.0,
) -> DatasetSplit:
    if protocol == "temporal":
        return temporal_split(
            dataset,
            min_user_ratings=min_user_ratings,
            min_item_ratings=min_item_ratings,
            seed=seed,
            cold_user_ratio=cold_user_ratio,
        )
    if protocol == "leave_one_out":
        return leave_one_out_split(
            dataset,
            min_user_ratings=min_user_ratings,
            min_item_ratings=min_item_ratings,
            seed=seed,
            cold_user_ratio=cold_user_ratio,
        )
    raise ValueError(f"Unsupported split protocol: {protocol}")


def split_stats(split: DatasetSplit) -> dict[str, int | float | str]:
    stats: dict[str, int | float | str] = {
        "protocol": split.protocol,
        "seed": split.seed,
        "train_ratings": split.train.n_ratings,
        "train_users": split.train.n_users,
        "train_items": split.train.n_items,
        "test_ratings": split.test.n_ratings,
        "test_users": split.test.n_users,
        "test_items": split.test.n_items,
    }
    if split.validation is not None:
        stats["validation_ratings"] = split.validation.n_ratings
        stats["validation_users"] = split.validation.n_users
        stats["validation_items"] = split.validation.n_items
    else:
        stats["validation_ratings"] = 0
        stats["validation_users"] = 0
        stats["validation_items"] = 0
    if split.cold_start_test is not None:
        stats["cold_start_ratings"] = split.cold_start_test.n_ratings
        stats["cold_start_users"] = split.cold_start_test.n_users
    else:
        stats["cold_start_ratings"] = 0
        stats["cold_start_users"] = 0
    return stats
