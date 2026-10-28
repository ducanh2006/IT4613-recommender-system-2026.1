from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping

import numpy as np
import pandas as pd

RATING_COLUMNS = ("user_id", "item_id", "rating", "timestamp")


@dataclass(frozen=True)
class IdMap:
    user_to_idx: Mapping[Hashable, int]
    idx_to_user: tuple[Hashable, ...]
    item_to_idx: Mapping[Hashable, int]
    idx_to_item: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        self._validate_side("user", self.user_to_idx, self.idx_to_user)
        self._validate_side("item", self.item_to_idx, self.idx_to_item)

    @staticmethod
    def _validate_side(
        name: str,
        mapping: Mapping[Hashable, int],
        reverse: tuple[Hashable, ...],
    ) -> None:
        if len(mapping) != len(reverse):
            raise ValueError(f"{name} map length mismatch")
        if len(set(reverse)) != len(reverse):
            raise ValueError(f"{name} reverse map contains duplicates")
        if set(mapping.values()) != set(range(len(reverse))):
            raise ValueError(f"{name} indices must be contiguous from 0")
        for external_id, idx in mapping.items():
            if not isinstance(idx, int):
                raise TypeError(f"{name} index must be int")
            if idx < 0 or idx >= len(reverse):
                raise ValueError(f"{name} index out of range")
            if reverse[idx] != external_id:
                raise ValueError(f"{name} map is not bijective")

    @classmethod
    def from_ids(
        cls,
        user_ids: list[Hashable] | tuple[Hashable, ...],
        item_ids: list[Hashable] | tuple[Hashable, ...],
    ) -> "IdMap":
        users = tuple(dict.fromkeys(user_ids))
        items = tuple(dict.fromkeys(item_ids))
        return cls(
            user_to_idx={value: idx for idx, value in enumerate(users)},
            idx_to_user=users,
            item_to_idx={value: idx for idx, value in enumerate(items)},
            idx_to_item=items,
        )

    def user_to_internal(self, user_id: Hashable) -> int:
        try:
            return self.user_to_idx[user_id]
        except KeyError as exc:
            raise KeyError(f"Unknown user_id: {user_id!r}") from exc

    def user_to_external(self, index: int) -> Hashable:
        if index < 0 or index >= len(self.idx_to_user):
            raise IndexError(f"User index out of range: {index}")
        return self.idx_to_user[index]

    def item_to_internal(self, item_id: Hashable) -> int:
        try:
            return self.item_to_idx[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id: {item_id!r}") from exc

    def item_to_external(self, index: int) -> Hashable:
        if index < 0 or index >= len(self.idx_to_item):
            raise IndexError(f"Item index out of range: {index}")
        return self.idx_to_item[index]

    @property
    def n_users(self) -> int:
        return len(self.idx_to_user)

    @property
    def n_items(self) -> int:
        return len(self.idx_to_item)

    @property
    def item_ids(self) -> tuple[Hashable, ...]:
        return self.idx_to_item


@dataclass(frozen=True)
class ItemFeatures:
    item_ids: tuple[Hashable, ...]
    feature_names: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("ItemFeatures.matrix must be 2-dimensional")
        if matrix.shape != (len(self.item_ids), len(self.feature_names)):
            raise ValueError("ItemFeatures matrix shape does not match metadata")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("ItemFeatures.item_ids contains duplicates")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("ItemFeatures.feature_names contains duplicates")
        if not np.isfinite(matrix).all():
            raise ValueError("ItemFeatures.matrix contains non-finite values")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(
            self, "_index", {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        )

    def vector(self, item_id: Hashable) -> np.ndarray:
        try:
            idx = self._index[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id: {item_id!r}") from exc
        return self.matrix[idx]

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class Dataset:
    name: str
    ratings: pd.DataFrame
    id_map: IdMap
    item_features: ItemFeatures | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset.name must not be empty")
        ratings = self.ratings.copy(deep=True).reset_index(drop=True)
        missing = [column for column in RATING_COLUMNS if column not in ratings.columns]
        if missing:
            raise ValueError(f"Dataset.ratings is missing columns: {missing}")
        ratings = ratings.loc[:, list(RATING_COLUMNS)]
        if ratings.empty:
            raise ValueError("Dataset.ratings must not be empty")
        if ratings[list(RATING_COLUMNS)].isna().any().any():
            raise ValueError("Dataset.ratings contains null values")
        if not pd.api.types.is_numeric_dtype(ratings["rating"]):
            raise TypeError("Dataset.ratings['rating'] must be numeric")
        if not pd.api.types.is_numeric_dtype(ratings["timestamp"]):
            raise TypeError("Dataset.ratings['timestamp'] must be numeric")
        if ratings.duplicated(["user_id", "item_id"]).any():
            raise ValueError("Dataset.ratings contains duplicate user-item pairs")
        unknown_users = set(ratings["user_id"]).difference(self.id_map.user_to_idx)
        if unknown_users:
            raise ValueError(f"Dataset.ratings contains unknown users: {unknown_users}")
        unknown_items = set(ratings["item_id"]).difference(self.id_map.item_to_idx)
        if unknown_items:
            raise ValueError(f"Dataset.ratings contains unknown items: {unknown_items}")
        rating_min = float(self.metadata.get("rating_min", 1.0))
        rating_max = float(self.metadata.get("rating_max", 5.0))
        if not ratings["rating"].between(rating_min, rating_max).all():
            raise ValueError(f"Ratings must be between {rating_min} and {rating_max}")
        object.__setattr__(self, "ratings", ratings)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def items(self) -> tuple[Hashable, ...]:
        return tuple(pd.unique(self.ratings["item_id"]).tolist())

    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @property
    def n_users(self) -> int:
        return self.ratings["user_id"].nunique()

    @property
    def n_items(self) -> int:
        return self.ratings["item_id"].nunique()

    def subset(self, ratings: pd.DataFrame, name: str | None = None) -> "Dataset":
        metadata = dict(self.metadata)
        metadata["parent_name"] = self.name
        if name is not None:
            metadata["subset_name"] = name
        return Dataset(
            name=name or self.name,
            ratings=ratings,
            id_map=self.id_map,
            item_features=self.item_features,
            metadata=metadata,
        )

    def user_items(self) -> dict[Hashable, frozenset[Hashable]]:
        grouped = self.ratings.groupby("user_id", sort=False)["item_id"].apply(frozenset)
        return grouped.to_dict()


@dataclass(frozen=True)
class DatasetSplit:
    train: Dataset
    validation: Dataset | None
    test: Dataset
    protocol: str
    seed: int
    cold_start_test: Dataset | None = None

    def __post_init__(self) -> None:
        if self.protocol not in {"temporal", "leave_one_out"}:
            raise ValueError(f"Unsupported split protocol: {self.protocol}")
        train_pairs = set(
            map(tuple, self.train.ratings[["user_id", "item_id"]].to_numpy().tolist())
        )
        test_pairs = set(
            map(tuple, self.test.ratings[["user_id", "item_id"]].to_numpy().tolist())
        )
        if train_pairs.intersection(test_pairs):
            raise ValueError("Train and test contain overlapping user-item pairs")
        if self.validation is not None:
            validation_pairs = set(
                map(
                    tuple,
                    self.validation.ratings[["user_id", "item_id"]].to_numpy().tolist(),
                )
            )
            if train_pairs.intersection(validation_pairs):
                raise ValueError("Train and validation contain overlapping user-item pairs")
            if test_pairs.intersection(validation_pairs):
                raise ValueError("Validation and test contain overlapping user-item pairs")
        if self.cold_start_test is not None:
            cold_pairs = set(
                map(
                    tuple,
                    self.cold_start_test.ratings[["user_id", "item_id"]]
                    .to_numpy()
                    .tolist(),
                )
            )
            if train_pairs.intersection(cold_pairs):
                raise ValueError("Train and cold_start_test contain overlapping pairs")
