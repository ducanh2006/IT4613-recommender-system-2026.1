from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import Dataset, IdMap, ItemFeatures

GENRE_COLUMNS = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
ITEM_COLUMNS = (
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *GENRE_COLUMNS,
)
RATING_COLUMNS = ("user_id", "item_id", "rating", "timestamp")


class MovieLensLoader:
    name = "ml-100k"
    version = "1.0.0"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Dataset:
        ratings_bytes, items_bytes = self._read_members()
        ratings = pd.read_csv(
            io.BytesIO(ratings_bytes),
            sep="\t",
            names=list(RATING_COLUMNS),
            encoding="latin-1",
            engine="python",
        )
        ratings = ratings.loc[:, list(RATING_COLUMNS)].dropna()
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["item_id"] = ratings["item_id"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype(int)
        ratings = ratings.sort_values(
            ["timestamp", "user_id", "item_id"], kind="mergesort"
        ).reset_index(drop=True)

        user_ids = tuple(sorted(ratings["user_id"].unique().tolist()))
        item_ids = tuple(sorted(ratings["item_id"].unique().tolist()))
        id_map = IdMap.from_ids(user_ids, item_ids)

        items = pd.read_csv(
            io.BytesIO(items_bytes),
            sep="|",
            names=list(ITEM_COLUMNS),
            encoding="latin-1",
            engine="python",
            quoting=csv.QUOTE_NONE,
        )
        items["item_id"] = items["item_id"].astype(int)
        items = items.drop_duplicates("item_id").reset_index(drop=True)
        feature_matrix = items.loc[:, list(GENRE_COLUMNS)].to_numpy(dtype=np.float32)
        item_features = ItemFeatures(
            item_ids=tuple(items["item_id"].astype(int).tolist()),
            feature_names=tuple(GENRE_COLUMNS),
            matrix=feature_matrix,
        )
        titles = {
            int(row.item_id): str(row.title)
            for row in items[["item_id", "title"]].itertuples(index=False)
        }
        release_years = {
            int(row.item_id): self._parse_year(row.release_date)
            for row in items[["item_id", "release_date"]].itertuples(index=False)
        }
        dataset = Dataset(
            name=self.name,
            ratings=ratings,
            id_map=id_map,
            item_features=item_features,
            metadata={
                "source": str(self.path),
                "rating_min": 1.0,
                "rating_max": 5.0,
                "item_titles": titles,
                "release_years": release_years,
                "genre_columns": GENRE_COLUMNS,
            },
        )
        return dataset

    def _read_members(self) -> tuple[bytes, bytes]:
        if self.path.is_dir():
            ratings_path = self.path / "u.data"
            items_path = self.path / "u.item"
            if not ratings_path.exists():
                raise FileNotFoundError(ratings_path)
            if not items_path.exists():
                raise FileNotFoundError(items_path)
            return ratings_path.read_bytes(), items_path.read_bytes()
        if self.path.is_file() and zipfile.is_zipfile(self.path):
            with zipfile.ZipFile(self.path) as archive:
                names = {Path(name).name: name for name in archive.namelist()}
                if "u.data" not in names or "u.item" not in names:
                    raise FileNotFoundError("u.data or u.item not found in archive")
                return archive.read(names["u.data"]), archive.read(names["u.item"])
        raise FileNotFoundError(f"MovieLens path not found: {self.path}")

    @staticmethod
    def _parse_year(value: object) -> int | None:
        if pd.isna(value):
            return None
        text = str(value)
        parts = text.split("-")
        if len(parts) != 3:
            return None
        try:
            return int(parts[2])
        except ValueError:
            return None
