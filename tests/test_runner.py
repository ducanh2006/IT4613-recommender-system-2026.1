from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.runner import run_benchmark
from tests.conftest import make_dataset


def write_dataset(path: Path, dataset) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dataset.ratings.loc[:, ["user_id", "item_id", "rating", "timestamp"]].to_csv(
        path / "u.data",
        sep="\t",
        header=False,
        index=False,
        encoding="latin-1",
    )
    lines = []
    for item_id in dataset.id_map.item_ids:
        genres = ["0"] * 19
        genres[item_id % 19] = "1"
        lines.append(
            "|".join(
                [
                    str(item_id),
                    f"Movie {item_id}",
                    "01-Jan-1995",
                    "",
                    "http://example.test",
                    *genres,
                ]
            )
        )
    (path / "u.item").write_text("\n".join(lines) + "\n", encoding="latin-1")


def test_runner_end_to_end(tmp_path):
    dataset = make_dataset(20, 20, 15)
    data_dir = tmp_path / "ml-100k"
    out_dir = tmp_path / "results"
    write_dataset(data_dir, dataset)
    result_dir = run_benchmark(
        data_dir=data_dir,
        out_dir=out_dir,
        protocol="leave_one_out",
        models=["random", "popularity", "global_mean"],
        k=3,
        seed=7,
        min_user_ratings=5,
        min_item_ratings=5,
        relevance_threshold=4.0,
        negative_samples=None,
        n_boot=20,
    )
    assert result_dir == out_dir
    for name in (
        "metrics_summary.csv",
        "global_metrics.csv",
        "per_user_metrics.csv",
        "recommendations_sample.csv",
        "efficiency.csv",
        "run_metadata.json",
        "report.md",
    ):
        assert (out_dir / name).exists(), name
    summary = pd.read_csv(out_dir / "metrics_summary.csv")
    assert not summary.empty
    assert set(summary["model"]) == {"random", "popularity", "global_mean"}
    assert "scope" in summary.columns
    assert set(summary["scope"]) <= {"user_macro", "global"}
    global_metrics = pd.read_csv(out_dir / "global_metrics.csv")
    rating_rows = global_metrics[global_metrics["task"] == "rating"]
    assert not rating_rows.empty
    macro = summary[
        (summary["model"] == "global_mean")
        & (summary["task"] == "rating")
        & (summary["metric"] == "rmse")
    ]["mean"].iloc[0]
    pooled = rating_rows[rating_rows["model"] == "global_mean"]["rmse"].iloc[0]
    assert pooled != macro
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Offline Benchmark Report" in report
    assert "rmse_pooled" in report
    samples = pd.read_csv(out_dir / "recommendations_sample.csv")
    if not samples.empty:
        assert samples["rank"].min() >= 1
        assert samples.groupby(["model", "user_id"])["item_id"].nunique().max() <= 3
