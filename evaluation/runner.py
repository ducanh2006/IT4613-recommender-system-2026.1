from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backend.app.recommender.contracts import Dataset, DatasetSplit
from backend.app.recommender.loaders import MovieLensLoader
from backend.app.recommender.models import (
    ContentBasedRecommender,
    FunkSVDRecommender,
    GlobalMeanRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    PopularityRecommender,
    RandomRecommender,
    UserKNNRecommender,
)

from evaluation.metrics import (
    EvaluationResult,
    evaluate_ranking,
    evaluate_rating,
    intra_list_diversity,
    novelty,
)
from evaluation.splits import build_split, split_stats
from evaluation.stats import (
    cliffs_delta,
    holm_bonferroni,
    paired_bootstrap_delta,
    summarize_values,
    wilcoxon_p,
)

MODEL_NAMES = (
    "random",
    "global_mean",
    "most_popular",
    "popularity",
    "content_based",
    "user_knn",
    "item_knn",
    "funk_svd",
    "hybrid",
)

RANKING_METRICS = (
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "map_at_k",
    "hit_rate_at_k",
    "mrr_at_k",
)
RATING_METRICS = ("rmse", "mae")
GLOBAL_RANKING_METRICS = ("coverage", "diversity", "novelty", "empty_rate", "fallback_rate")


def build_model(name: str, seed: int):
    factories = {
        "random": lambda: RandomRecommender(seed=seed),
        "global_mean": GlobalMeanRecommender,
        "most_popular": MostPopularRecommender,
        "popularity": PopularityRecommender,
        "content_based": ContentBasedRecommender,
        "user_knn": UserKNNRecommender,
        "item_knn": ItemKNNRecommender,
        "funk_svd": lambda: FunkSVDRecommender(seed=seed),
        "hybrid": lambda: HybridRecommender(seed=seed),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODEL_NAMES)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return names


def compute_diversity_novelty(
    samples: pd.DataFrame,
    train: Dataset,
    item_features,
) -> tuple[float, float]:
    if samples.empty:
        return 0.0, 0.0
    lists = samples.groupby("user_id", sort=False)["item_id"].apply(list).tolist()
    diversity = intra_list_diversity(lists, item_features)
    item_counts = train.ratings.groupby("item_id").size().to_dict()
    novelty_value = novelty(lists, item_counts, train.n_users)
    return diversity, novelty_value


def summarize_result(
    model_name: str,
    split_name: str,
    task: str,
    result: EvaluationResult,
    fit_time_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    metrics = dict(result.global_metrics)
    if task == "ranking":
        for metric in RANKING_METRICS:
            values = result.per_user[metric].tolist() if metric in result.per_user else []
            stats = summarize_values(values)
            summary_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "task": task,
                    "metric": metric,
                    "scope": "user_macro",
                    **stats,
                }
            )
        for metric in GLOBAL_RANKING_METRICS:
            if metric in metrics:
                summary_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "task": task,
                        "metric": metric,
                        "scope": "global",
                        "mean": metrics[metric],
                        "std": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": float("nan"),
                    }
                )
    else:
        for metric in RATING_METRICS:
            values = result.per_user[metric].tolist() if metric in result.per_user else []
            stats = summarize_values(values)
            summary_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "task": task,
                    "metric": metric,
                    "scope": "user_macro",
                    **stats,
                }
            )
    global_row = {
        "model": model_name,
        "split": split_name,
        "task": task,
        "fit_time_s": fit_time_s,
        **metrics,
    }
    return summary_rows, global_row


def make_comparisons(
    per_user_by_model: dict[tuple[str, str, str], pd.DataFrame],
    model_names: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in sorted(
        {key[1] for key in per_user_by_model}
    ):
        for task, baseline_name, metric_names in (
            ("ranking", "popularity", RANKING_METRICS),
            ("rating", "global_mean", RATING_METRICS),
        ):
            baseline = per_user_by_model.get((baseline_name, split_name, task))
            if baseline is None or baseline.empty:
                continue
            baseline = baseline.set_index("user_id")
            for model_name in model_names:
                if model_name == baseline_name:
                    continue
                frame = per_user_by_model.get((model_name, split_name, task))
                if frame is None or frame.empty:
                    continue
                frame = frame.set_index("user_id")
                joined = frame.join(
                    baseline[list(metric_names)],
                    how="inner",
                    lsuffix="_model",
                    rsuffix="_baseline",
                )
                if joined.empty:
                    continue
                for metric in metric_names:
                    model_values = joined[f"{metric}_model"].to_numpy(dtype=float)
                    baseline_values = joined[f"{metric}_baseline"].to_numpy(dtype=float)
                    delta, ci_low, ci_high = paired_bootstrap_delta(
                        model_values,
                        baseline_values,
                        n_boot=n_boot,
                        seed=seed,
                    )
                    p_value = wilcoxon_p(model_values, baseline_values)
                    higher_is_better = metric not in {"rmse", "mae"}
                    baseline_mean = float(np.nanmean(baseline_values))
                    model_mean = float(np.nanmean(model_values))
                    if baseline_mean == 0.0 or not np.isfinite(baseline_mean):
                        relative = float("nan")
                    elif higher_is_better:
                        relative = (model_mean - baseline_mean) / abs(baseline_mean)
                    else:
                        relative = (baseline_mean - model_mean) / abs(baseline_mean)
                    favorable = delta > 0.0 if higher_is_better else delta < 0.0
                    rows.append(
                        {
                            "split": split_name,
                            "task": task,
                            "model": model_name,
                            "baseline": baseline_name,
                            "metric": metric,
                            "higher_is_better": higher_is_better,
                            "n_pairs": int(len(model_values)),
                            "model_mean": model_mean,
                            "baseline_mean": baseline_mean,
                            "delta": delta,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_value,
                            "cliffs_delta": cliffs_delta(model_values, baseline_values),
                            "relative_improvement": relative,
                            "favorable": favorable,
                            "primary": metric in {"ndcg_at_k", "rmse"},
                        }
                    )
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return comparisons
    comparisons["family"] = comparisons["split"] + "|" + comparisons["task"]
    comparisons["p_adjusted"] = float("nan")
    for _, group in comparisons.groupby("family", sort=False):
        adjusted = holm_bonferroni(group["p_value"].fillna(1.0).tolist())
        comparisons.loc[group.index, "p_adjusted"] = adjusted
    comparisons["family_size"] = comparisons.groupby("family")["p_value"].transform("size")
    comparisons["significant"] = (
        (comparisons["p_adjusted"] < 0.05)
        & (
            ((comparisons["ci_low"] > 0.0) & (comparisons["ci_high"] > 0.0))
            | ((comparisons["ci_low"] < 0.0) & (comparisons["ci_high"] < 0.0))
        )
    )
    comparisons["better"] = comparisons["significant"] & comparisons["favorable"]
    comparisons["worse"] = comparisons["significant"] & ~comparisons["favorable"]
    return comparisons


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for name in ("u.data", "u.item"):
            file_path = path / name
            if file_path.exists():
                digest.update(file_path.read_bytes())
        return digest.hexdigest()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    return ""


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("pandas", "numpy", "sklearn", "scipy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def format_count(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "N/A"
    return str(int(round(number)))


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_Không có dữ liệu._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _metric_frame(
    summary: pd.DataFrame,
    split_name: str,
    task: str,
    metric_names: list[str],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary[
        (summary["split"] == split_name)
        & (summary["task"] == task)
        & (summary["metric"].isin(metric_names))
    ]


def _evaluated_users(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    macro = frame[frame["scope"] == "user_macro"]
    if macro.empty:
        return pd.Series(dtype=float)
    return macro.groupby("model")["n"].max()


def build_report(
    summary: pd.DataFrame,
    global_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    efficiency: pd.DataFrame,
    split: DatasetSplit,
    dataset: Dataset,
    k: int,
    protocol: str,
) -> str:
    stats = split_stats(split)
    lines = [
        "# Offline Benchmark Report",
        "",
        "## Dataset & Split",
        "",
        f"- Dataset: `{dataset.name}`",
        f"- Protocol: `{protocol}`",
        f"- K: `{k}`",
        f"- Train: {stats['train_ratings']} ratings / {stats['train_users']} users / {stats['train_items']} items",
        f"- Validation: {stats['validation_ratings']} ratings",
        f"- Test: {stats['test_ratings']} ratings / {stats['test_users']} users",
        f"- Cold-start test: {stats['cold_start_ratings']} ratings / {stats['cold_start_users']} users",
        "",
        "## Ranking on Test",
        "",
    ]
    ranking = _metric_frame(
        summary,
        "test",
        "ranking",
        list(RANKING_METRICS) + list(GLOBAL_RANKING_METRICS),
    )
    if not ranking.empty:
        pivot = ranking.pivot_table(
            index="model", columns="metric", values="mean", aggfunc="first"
        ).reset_index()
        display = [
            "ndcg_at_k",
            "precision_at_k",
            "recall_at_k",
            "hit_rate_at_k",
            "map_at_k",
            "mrr_at_k",
            "coverage",
            "diversity",
            "novelty",
        ]
        columns = ["model"] + [metric for metric in display if metric in pivot.columns]
        pivot = pivot[columns]
        pivot["users"] = pivot["model"].map(_evaluated_users(ranking))
        if "ndcg_at_k" in pivot.columns:
            pivot = pivot.sort_values("ndcg_at_k", ascending=False)
        for column in columns[1:]:
            pivot[column] = pivot[column].map(lambda value: format_float(value))
        pivot["users"] = pivot["users"].map(lambda value: format_count(value))
        lines.append(markdown_table(pivot, list(pivot.columns)))
    else:
        lines.append("_Không có dữ liệu ranking._")
    lines.extend(["", "## Rating Prediction on Test", ""])
    rating = _metric_frame(summary, "test", "rating", list(RATING_METRICS))
    if not rating.empty:
        table = rating.pivot_table(
            index="model", columns="metric", values="mean", aggfunc="first"
        ).reset_index()
        table = table.rename(
            columns={"rmse": "rmse_user_mean", "mae": "mae_user_mean"}
        )
        if not global_metrics.empty:
            pooled = global_metrics[
                (global_metrics["split"] == "test")
                & (global_metrics["task"] == "rating")
            ].set_index("model")
            if not pooled.empty:
                table["rmse_pooled"] = table["model"].map(pooled["rmse"])
                table["mae_pooled"] = table["model"].map(pooled["mae"])
                table["ratings"] = table["model"].map(pooled["n_ratings"])
        table["users"] = table["model"].map(_evaluated_users(rating))
        columns = [
            "model",
            "rmse_pooled",
            "mae_pooled",
            "rmse_user_mean",
            "mae_user_mean",
            "users",
            "ratings",
        ]
        columns = [column for column in columns if column in table.columns]
        table = table[columns]
        sort_column = (
            "rmse_pooled" if "rmse_pooled" in columns else "rmse_user_mean"
        )
        table = table.sort_values(sort_column)
        for column in columns[1:]:
            if column in {"users", "ratings"}:
                table[column] = table[column].map(lambda value: format_count(value))
            else:
                table[column] = table[column].map(lambda value: format_float(value))
        lines.append(markdown_table(table, columns))
    else:
        lines.append("_Không có model nào hỗ trợ rating prediction._")

    lines.extend(["", "## So sánh với baseline", ""])
    primary = comparisons[
        (comparisons["split"] == "test") & (comparisons["primary"])
    ] if not comparisons.empty else pd.DataFrame()
    if not primary.empty:
        table = primary.copy()
        table["delta_ci"] = table.apply(
            lambda row: f"{format_float(row['delta'])} [{format_float(row['ci_low'])}, {format_float(row['ci_high'])}]",
            axis=1,
        )
        table["p_adj"] = table["p_adjusted"].map(lambda value: format_float(value))
        table["relative"] = table["relative_improvement"].map(lambda value: format_float(value))
        table["outcome"] = table.apply(
            lambda row: (
                "tốt hơn"
                if row["better"]
                else "kém hơn"
                if row["worse"]
                else "không rõ"
            ),
            axis=1,
        )
        table = table[
            [
                "model",
                "baseline",
                "metric",
                "delta_ci",
                "p_adj",
                "family_size",
                "relative",
                "outcome",
            ]
        ].sort_values(["metric", "model"])
        lines.append(markdown_table(table, list(table.columns)))
    else:
        lines.append("_Không có so sánh baseline._")

    lines.extend(["", "## Cold-start", ""])
    cold = _metric_frame(summary, "cold_start", "ranking", ["ndcg_at_k"])
    if not cold.empty:
        table = cold[["model", "mean", "ci_low", "ci_high"]].copy()
        table.columns = ["model", "NDCG@K", "CI low", "CI high"]
        for column in ["NDCG@K", "CI low", "CI high"]:
            table[column] = table[column].map(lambda value: format_float(value))
        table["users"] = table["model"].map(_evaluated_users(cold))
        table["users"] = table["users"].map(lambda value: format_count(value))
        table = table.sort_values("NDCG@K", ascending=False)
        lines.append(markdown_table(table, list(table.columns)))
    else:
        lines.append("_Không có cold-start users._")

    lines.extend(["", "## Efficiency", ""])
    if not efficiency.empty:
        table = efficiency.copy()
        for column in ["fit_time_s", "recommend_latency_mean_ms", "recommend_latency_p95_ms"]:
            if column in table:
                table[column] = table[column].map(lambda value: format_float(value))
        lines.append(markdown_table(table, list(table.columns)))
    else:
        lines.append("_Không có dữ liệu hiệu năng._")

    lines.extend(["", "## Kết luận", ""])
    if comparisons.empty:
        lines.append("- Không đủ dữ liệu so sánh.")
    else:
        for model_name in sorted(comparisons["model"].unique()):
            model_rows = comparisons[
                (comparisons["model"] == model_name)
                & (comparisons["split"] == "test")
                & (comparisons["primary"])
            ]
            if model_rows.empty:
                continue
            better = model_rows[model_rows["better"]]
            worse = model_rows[model_rows["worse"]]
            if not better.empty:
                metrics_text = ", ".join(better["metric"].tolist())
                lines.append(
                    f"- `{model_name}`: tốt hơn baseline có ý nghĩa ở {metrics_text}."
                )
            elif not worse.empty:
                metrics_text = ", ".join(worse["metric"].tolist())
                lines.append(
                    f"- `{model_name}`: không tốt hơn baseline; kém hơn có ý nghĩa ở {metrics_text}."
                )
            else:
                lines.append(
                    f"- `{model_name}`: chưa đủ bằng chứng tốt hơn baseline."
                )
    lines.extend(
        [
            "",
            "## Ghi chú",
            "",
            "- `rmse_pooled`/`mae_pooled` tính trên toàn bộ rating của tập test; `rmse_user_mean`/`mae_user_mean` là trung bình theo từng user (macro-average).",
            "- Metric ranking là trung bình theo từng user; cột `users` là số user thực sự được đánh giá.",
            "- `significance` dùng paired bootstrap 95% CI và Wilcoxon, hiệu chỉnh Holm–Bonferroni trong từng family (split + task); cột `family_size` là số phép so sánh trong family đó.",
            "- Primary metric: `ndcg_at_k` cho ranking, `rmse` cho rating.",
            "- Kết quả chỉ có ý nghĩa trong phạm vi protocol và dữ liệu đã dùng.",
            "",
        ]
    )
    return "\n".join(lines)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        return pd.DataFrame()
    return pd.concat(populated, ignore_index=True)


def run_benchmark(
    data_dir: Path,
    out_dir: Path,
    protocol: str,
    models: list[str],
    k: int,
    seed: int,
    min_user_ratings: int,
    min_item_ratings: int,
    relevance_threshold: float,
    negative_samples: int | None,
    n_boot: int,
    cold_user_ratio: float = 0.1,
) -> Path:
    loader = MovieLensLoader(data_dir)
    dataset = loader.load()
    split = build_split(
        dataset,
        protocol=protocol,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
        seed=seed,
        cold_user_ratio=cold_user_ratio,
    )
    catalog_item_ids = tuple(
        sorted(
            split.train.ratings.groupby("item_id").size().loc[lambda series: series >= min_item_ratings].index.tolist(),
            key=str,
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    per_user_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    efficiency_rows: list[dict[str, Any]] = []
    per_user_by_model: dict[tuple[str, str, str], pd.DataFrame] = {}

    for model_name in models:
        model = build_model(model_name, seed)

        fit_start = time.perf_counter()
        model.fit(split.train, split.validation)
        fit_time = time.perf_counter() - fit_start

        rating_result = None
        if model.capabilities.supports_rating_prediction:
            rating_result = evaluate_rating(model, split.test, "test")
            rows, global_row = summarize_result(
                model_name, "test", "rating", rating_result, fit_time
            )
            summary_rows.extend(rows)
            global_rows.append(global_row)
            rating_frame = rating_result.per_user.copy()
            rating_frame["task"] = "rating"
            per_user_frames.append(rating_frame)
            per_user_by_model[(model_name, "test", "rating")] = rating_frame

        ranking_result = evaluate_ranking(
            model,
            split.test,
            split.train,
            k=k,
            relevance_threshold=relevance_threshold,
            catalog_item_ids=catalog_item_ids,
            negative_samples=negative_samples,
            seed=seed,
            split_name="test",
        )
        diversity, novelty_value = compute_diversity_novelty(
            ranking_result.samples, split.train, split.train.item_features
        )
        ranking_result.global_metrics["diversity"] = diversity
        ranking_result.global_metrics["novelty"] = novelty_value
        rows, global_row = summarize_result(
            model_name, "test", "ranking", ranking_result, fit_time
        )
        summary_rows.extend(rows)
        global_rows.append(global_row)
        ranking_frame = ranking_result.per_user.copy()
        ranking_frame["task"] = "ranking"
        per_user_frames.append(ranking_frame)
        per_user_by_model[(model_name, "test", "ranking")] = ranking_frame
        sample_frames.append(ranking_result.samples)

        if split.cold_start_test is not None and not split.cold_start_test.ratings.empty:
            cold_result = evaluate_ranking(
                model,
                split.cold_start_test,
                split.train,
                k=k,
                relevance_threshold=relevance_threshold,
                catalog_item_ids=catalog_item_ids,
                negative_samples=negative_samples,
                seed=seed,
                split_name="cold_start",
            )
            cold_diversity, cold_novelty = compute_diversity_novelty(
                cold_result.samples, split.train, split.train.item_features
            )
            cold_result.global_metrics["diversity"] = cold_diversity
            cold_result.global_metrics["novelty"] = cold_novelty
            rows, global_row = summarize_result(
                model_name, "cold_start", "ranking", cold_result, fit_time
            )
            summary_rows.extend(rows)
            global_rows.append(global_row)
            cold_frame = cold_result.per_user.copy()
            cold_frame["task"] = "ranking"
            per_user_frames.append(cold_frame)
            per_user_by_model[(model_name, "cold_start", "ranking")] = cold_frame
            sample_frames.append(cold_result.samples)

        recommend_latencies = (
            ranking_result.per_user["latency_ms"].tolist()
            if not ranking_result.per_user.empty
            else []
        )
        efficiency_rows.append(
            {
                "model": model_name,
                "fit_time_s": fit_time,
                "recommend_latency_mean_ms": float(np.mean(recommend_latencies))
                if recommend_latencies
                else float("nan"),
                "recommend_latency_p95_ms": float(np.percentile(recommend_latencies, 95))
                if recommend_latencies
                else float("nan"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    global_metrics = pd.DataFrame(global_rows)
    per_user = _concat_frames(per_user_frames)
    samples = _concat_frames(sample_frames)
    efficiency = pd.DataFrame(efficiency_rows)
    comparisons = make_comparisons(per_user_by_model, models, n_boot=n_boot, seed=seed)

    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    if not global_metrics.empty:
        global_metrics.to_csv(out_dir / "global_metrics.csv", index=False)
    per_user.to_csv(out_dir / "per_user_metrics.csv", index=False)
    samples.to_csv(out_dir / "recommendations_sample.csv", index=False)
    efficiency.to_csv(out_dir / "efficiency.csv", index=False)
    if not comparisons.empty:
        comparisons.to_csv(out_dir / "comparisons.csv", index=False)
    if not summary.empty:
        cold = summary[
            (summary["split"] == "cold_start") & (summary["task"] == "ranking")
        ]
        cold.to_csv(out_dir / "cold_start_metrics.csv", index=False)

    metadata = {
        "dataset": dataset.name,
        "dataset_path": str(data_dir),
        "dataset_hash": dataset_hash(data_dir),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "seed": seed,
        "protocol": protocol,
        "k": k,
        "relevance_threshold": relevance_threshold,
        "negative_samples": negative_samples,
        "min_user_ratings": min_user_ratings,
        "min_item_ratings": min_item_ratings,
        "cold_user_ratio": cold_user_ratio,
        "models": models,
        "split": split_stats(split),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(
        summary,
        global_metrics,
        comparisons,
        efficiency,
        split,
        dataset,
        k,
        protocol,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens offline benchmark")
    parser.add_argument("--dataset", default="ml-100k")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument("--out", type=Path, default=Path("results/latest"))
    parser.add_argument(
        "--protocol",
        choices=("temporal", "leave_one_out"),
        default="temporal",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="all")
    parser.add_argument("--min-user-ratings", type=int, default=5)
    parser.add_argument("--min-item-ratings", type=int, default=5)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--negative-samples", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--cold-user-ratio",
        type=float,
        default=0.1,
        help="Fraction of users held out of training to simulate cold start.",
    )
    args = parser.parse_args()

    if args.dataset != "ml-100k":
        raise SystemExit("Only ml-100k is supported")
    if args.k <= 0:
        raise SystemExit("--k must be positive")
    if not 0.0 <= args.cold_user_ratio < 1.0:
        raise SystemExit("--cold-user-ratio must be between 0 and 1")
    models = parse_models(args.models)
    out_dir = run_benchmark(
        data_dir=args.data_dir,
        out_dir=args.out,
        protocol=args.protocol,
        models=models,
        k=args.k,
        seed=args.seed,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings,
        relevance_threshold=args.relevance_threshold,
        negative_samples=args.negative_samples,
        n_boot=args.n_boot,
        cold_user_ratio=args.cold_user_ratio,
    )
    print(f"Benchmark written to {out_dir}")


if __name__ == "__main__":
    main()
