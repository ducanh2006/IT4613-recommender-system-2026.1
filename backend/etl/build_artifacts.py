"""
ETL offline: precompute artifact gợi ý từ MovieLens-32m cho 20 phim demo.

Chạy (từ thư mục gốc project):
    python -m backend.etl.build_artifacts [--backfill-db]

Đọc ml-32m/ratings.csv (32M dòng) theo chunk, chỉ giữ 20 movieId đã map,
tính:
  1. Popularity  : vote_count / mean_rating / weighted_rating (IMDB WR) cho 20 phim demo.
  2. Item-item   : adjusted cosine similarity (mean-centered) + shrinkage, 20x20.
  3. Ghi artifact vào backend/artifacts/ (demo_popularity.json, demo_similarity.npz,
     ml_mapping.json).
  4. --backfill-db: ghi vote_count/rating_avg thật vào database/app.db.

Không import 32M dòng vào SQLite; artifact rất nhỏ (~vài KB), API nạp 1 lần vào RAM.
"""
import os
import sys
import json
import argparse

# Cho phép import backend.* khi chạy trực tiếp bằng `python backend/etl/build_artifacts.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from backend.app.config import (
    RATINGS_CSV,
    ARTIFACTS_DIR,
    SIMILARITY_NPZ,
    POPULARITY_JSON,
    MAPPING_JSON,
    SHRINKAGE_LAMBDA,
    POPULARITY_M,
)
from backend.app.recommender.ml_mapping import DEMO_TO_ML, ML_TO_DEMO

CHUNK_SIZE = 2_000_000


def load_filtered_ratings():
    """Đọc ratings.csv theo chunk, trả DataFrame chỉ gồm 20 movieId đã map (và tổng thống kê toàn cục)."""
    ml_ids = set(DEMO_TO_ML.values())
    frames = []
    global_sum = 0.0
    global_count = 0
    for chunk in pd.read_csv(
        RATINGS_CSV,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        chunksize=CHUNK_SIZE,
    ):
        global_sum += float(chunk["rating"].sum())
        global_count += int(len(chunk))
        sub = chunk[chunk["movieId"].isin(ml_ids)]
        if len(sub):
            frames.append(sub[["userId", "movieId", "rating"]])
    filtered = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["userId", "movieId", "rating"]
    )
    C = (global_sum / global_count) if global_count else 4.0
    return filtered, C


def build_popularity(filtered, C):
    """vote_count, mean_rating, weighted_rating cho từng phim demo."""
    if len(filtered) == 0:
        return {}
    grouped = filtered.groupby("movieId")["rating"].agg(["count", "mean"])
    pop = {}
    for demo_id, ml_id in DEMO_TO_ML.items():
        if ml_id not in grouped.index:
            continue
        v = int(grouped.loc[ml_id, "count"])
        r = float(grouped.loc[ml_id, "mean"])
        wr = (v / (v + POPULARITY_M)) * r + (POPULARITY_M / (v + POPULARITY_M)) * C
        pop[demo_id] = {
            "ml_id": ml_id,
            "vote_count": v,
            "mean_rating": round(r, 3),
            "weighted_rating": round(wr, 4),
        }
    return pop


def build_similarity(filtered):
    """Adjusted cosine (mean-centered) + shrinkage giữa 20 phim demo."""
    ml_ids_sorted = sorted(DEMO_TO_ML.values())
    demo_ids_sorted = [ML_TO_DEMO[m] for m in ml_ids_sorted]
    n = len(ml_ids_sorted)

    matrix = filtered.pivot(index="userId", columns="movieId", values="rating")
    matrix = matrix.reindex(columns=ml_ids_sorted)  # đảm bảo đủ 20 cột, thiếu -> NaN

    R = matrix.notna().to_numpy(dtype="float32")      # users x n (0/1)
    support = (R.T @ R).astype("int32")               # n x n: số user cùng đánh giá

    M = matrix.to_numpy(dtype="float32")              # users x n, có NaN
    col_mean = np.nan_to_num(np.nanmean(M, axis=0), nan=0.0)
    Mc = np.nan_to_num(M - col_mean, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.sqrt((Mc ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    Mn = Mc / norms
    sim = Mn.T @ Mn                                   # n x n adjusted cosine
    sim = np.nan_to_num(sim, nan=0.0)

    support_f = support.astype("float32")
    sim_shrunk = (support_f / (support_f + SHRINKAGE_LAMBDA)) * sim
    np.fill_diagonal(sim_shrunk, 0.0)

    return demo_ids_sorted, ml_ids_sorted, sim_shrunk, support


def backfill_db(pop):
    """Ghi vote_count/rating_avg thật (từ ML-32m) vào database/app.db cho 20 phim demo."""
    import sqlite3
    from backend.app.database import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for demo_id, st in pop.items():
        cur.execute(
            "UPDATE movies SET rating_avg = ?, vote_count = ? WHERE id = ?;",
            (st["mean_rating"], st["vote_count"], demo_id),
        )
    conn.commit()
    conn.close()
    print(f"[ETL] Đã backfill vote_count/rating_avg cho {len(pop)} phim demo vào {DB_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Precompute ML-32m artifacts cho 20 phim demo.")
    parser.add_argument("--backfill-db", action="store_true", help="Ghi số liệu thật vào app.db")
    args = parser.parse_args()

    if not os.path.exists(RATINGS_CSV):
        print(f"[ETL] Không tìm thấy {RATINGS_CSV} -> thoát.")
        sys.exit(1)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("[ETL] Đang đọc ratings.csv (chunk) ...")
    filtered, C = load_filtered_ratings()
    print(f"[ETL] Giữ lại {len(filtered)} dòng của 20 phim demo. C(global mean) = {C:.4f}")

    pop = build_popularity(filtered, C)
    print(f"[ETL] Popularity: {len(pop)}/20 phim demo có dữ liệu.")

    demo_ids, ml_ids, sim, support = build_similarity(filtered)
    print(f"[ETL] Similarity: ma trận {sim.shape[0]}x{sim.shape[1]} (support 0..{support.max()}).")

    # Ghi artifact
    np.savez(
        SIMILARITY_NPZ,
        demo_ids=np.array(demo_ids, dtype=np.int32),
        ml_ids=np.array(ml_ids, dtype=np.int32),
        sim=sim.astype("float32"),
        support=support,
    )
    with open(POPULARITY_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in pop.items()}, f, ensure_ascii=False, indent=2)
    with open(MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in DEMO_TO_ML.items()}, f, ensure_ascii=False, indent=2)

    print(f"[ETL] Đã ghi artifact vào {ARTIFACTS_DIR}")

    if args.backfill_db:
        backfill_db(pop)

    print("[ETL] Hoàn tất.")


if __name__ == "__main__":
    main()
