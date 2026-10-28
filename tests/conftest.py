from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.recommender.contracts import Dataset, IdMap, ItemFeatures


def make_dataset(
    n_users: int = 6,
    n_items: int = 8,
    ratings_per_user: int = 5,
    n_features: int = 4,
    seed: int = 7,
) -> Dataset:
    rng = np.random.default_rng(seed)
    user_ids = tuple(range(1, n_users + 1))
    item_ids = tuple(range(101, 101 + n_items))
    rows = []
    for user_id in user_ids:
        chosen = rng.choice(item_ids, size=min(ratings_per_user, n_items), replace=False)
        for item_id in chosen:
            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "rating": float(rng.integers(1, 6)),
                }
            )
    timestamps = 1_000 + rng.permutation(len(rows))
    for row, timestamp in zip(rows, timestamps):
        row["timestamp"] = int(timestamp)
    ratings = pd.DataFrame(rows).sort_values(
        ["timestamp", "user_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    id_map = IdMap.from_ids(user_ids, item_ids)
    matrix = np.zeros((len(item_ids), n_features), dtype=np.float32)
    for index, item_id in enumerate(item_ids):
        matrix[index, item_id % n_features] = 1.0
        matrix[index, (item_id + 1) % n_features] = 0.5
    features = ItemFeatures(
        item_ids=item_ids,
        feature_names=tuple(f"feature_{index}" for index in range(n_features)),
        matrix=matrix,
    )
    return Dataset(
        name="tiny",
        ratings=ratings,
        id_map=id_map,
        item_features=features,
        metadata={
            "rating_min": 1.0,
            "rating_max": 5.0,
            "item_titles": {item_id: f"Movie {item_id}" for item_id in item_ids},
        },
    )


@pytest.fixture
def tiny_dataset() -> Dataset:
    return make_dataset()


def write_movielens_dir(
    path: Path,
    ratings_text: str,
    item_rows: list[tuple[int, str, list[str]]],
) -> None:
    """Write a MovieLens directory with a u.data file and a u.item file."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "u.data").write_text(ratings_text, encoding="latin-1")
    lines = [
        "|".join(
            [
                str(item_id),
                title,
                "01-Jan-1995",
                "",
                "http://example.test",
                *genres,
            ]
        )
        for item_id, title, genres in item_rows
    ]
    (path / "u.item").write_text("\n".join(lines) + "\n", encoding="latin-1")
