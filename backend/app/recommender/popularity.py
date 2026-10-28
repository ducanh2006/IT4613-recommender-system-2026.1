import sqlite3
from typing import List, Dict, Any

from .base import BaseRecommender
from .utils import fetch_movies_with_genres, build_movie_payload
from .loader import get_popularity


class PopularityRecommender(BaseRecommender):
    """
    Gợi ý theo độ phổ biến (Popularity / Trending).

    - Ưu tiên: dùng điểm Weighted Rating đã precompute từ MovieLens-32m
      (vote_count / rating_avg thật, hàng chục-nghìn lượt), map về phim demo.
    - Fallback (chưa có artifact): tự tính IMDB Weighted Rating từ SQLite
      với vote_count/rating_avg thật (mặc định m=5), không dùng số giả.
    """

    def __init__(self, m: int = 5):
        self.m = m  # chỉ dùng cho nhánh fallback SQLite

    def recommend(
        self,
        conn: sqlite3.Connection,
        user_id: int = None,
        movie_id: int = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        movies = fetch_movies_with_genres(conn)
        pop = get_popularity()

        # --- Nhánh chính: dữ liệu thật từ MovieLens-32m ---
        if pop:
            scored = []
            for row in movies:
                stats = pop.get(row["id"])
                if stats is None:
                    continue
                # Ghi đè hiển thị bằng số liệu thật (vote_count/rating_avg từ ML-32m)
                row["rating_avg"] = stats.get("mean_rating") or 0.0
                row["vote_count"] = stats.get("vote_count") or 0
                scored.append(
                    build_movie_payload(
                        row,
                        "score",
                        round(float(stats.get("weighted_rating") or 0.0), 3),
                    )
                )
            scored.sort(key=lambda x: (-x["score"], -x["vote_count"], x["id"]))
            return scored[:top_k]

        # --- Fallback: tự tính IMDB Weighted Rating từ SQLite thật ---
        cursor = conn.cursor()
        avg_row = cursor.execute(
            "SELECT AVG(rating_avg) FROM movies WHERE vote_count > 0;"
        ).fetchone()
        C = avg_row[0] if (avg_row and avg_row[0]) else 4.0

        scored = []
        for row in movies:
            v = row["vote_count"] or 0
            R = row["rating_avg"] or 0.0
            wr = (v / (v + self.m)) * R + (self.m / (v + self.m)) * C
            scored.append(build_movie_payload(row, "score", round(wr, 3)))

        scored.sort(key=lambda x: (-x["score"], -x["vote_count"], x["id"]))
        return scored[:top_k]
