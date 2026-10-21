from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.recommender.loaders import MovieLensLoader


def write_movielens(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "u.data").write_text(
        "1\t10\t5\t100\n1\t11\t4\t101\n2\t10\t3\t102\n2\t11\t5\t103\n",
        encoding="latin-1",
    )
    item_lines = []
    for item_id, title in ((10, "Ten"), (11, "Eleven")):
        genres = ["0"] * 19
        genres[0] = "1"
        genres[1] = "1"
        item_lines.append(
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
        )
    (path / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")


def test_movielens_loader(tmp_path):
    write_movielens(tmp_path)
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
