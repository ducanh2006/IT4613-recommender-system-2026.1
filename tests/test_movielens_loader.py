from __future__ import annotations

import numpy as np

from backend.app.recommender.loaders import MovieLensLoader
from tests.conftest import write_movielens_dir


def test_movielens_loader(tmp_path):
    genres = ["0"] * 19
    genres[0] = "1"
    genres[1] = "1"
    write_movielens_dir(
        tmp_path,
        "1\t10\t5\t100\n1\t11\t4\t101\n2\t10\t3\t102\n2\t11\t5\t103\n",
        [(10, "Ten", genres), (11, "Eleven", genres)],
    )
    dataset = MovieLensLoader(tmp_path).load()
    assert dataset.n_ratings == 4
    assert dataset.n_users == 2
    assert dataset.n_items == 2
    assert dataset.ratings["timestamp"].tolist() == [100, 101, 102, 103]
    assert dataset.item_features is not None
    assert dataset.item_features.n_features == 19
    assert dataset.item_features.n_items == 2
    assert dataset.metadata["item_titles"][10] == "Ten"
    assert dataset.metadata["release_years"][10] == 1995
    assert np.allclose(dataset.item_features.vector(10)[:2], [1.0, 1.0])
