import sqlite3
from typing import List, Dict, Any

from ..config import LIKE_THRESHOLD, W_CF, W_CONTENT, W_POP, RATING_SCALE
from .base import BaseRecommender
from .popularity import PopularityRecommender
from .utils import fetch_movies_with_genres, build_movie_payload, minmax
from .loader import get_similarity, get_popularity


def _genre_cosine(user_genre: Dict[int, float], movie_genre_ids: List[int]) -> float:
    """Cosine giữa vector sở thích thể loại của user và one-hot thể loại của phim (đã có giới hạn [0,1])."""
    if not user_genre or not movie_genre_ids:
        return 0.0
    num = sum(user_genre.get(g, 0.0) for g in movie_genre_ids)
    if num <= 0:
        return 0.0
    den_user = sum(v * v for v in user_genre.values()) ** 0.5
    den_movie = len(movie_genre_ids) ** 0.5
    den = den_user * den_movie
    return float(num / den) if den > 0 else 0.0


class CollaborativeRecommender(BaseRecommender):
    """
    Gợi ý cá nhân hóa "dành cho bạn" (for-you).

    Đây là Item-based Collaborative Filtering lai:
      - cf_score: điểm dự đoán từ item-item similarity (cosine + shrinkage) precompute
        từ MovieLens-32m, kết hợp rating của user trên các phim đã thích.
      - content_score: cosine giữa hồ sơ thể loại của user và thể loại phim.
      - popularity_score: weighted rating (normalized) của phim.
    Điểm cuối: weighted = W_CF*cf + W_CONTENT*content + W_POP*pop, ánh xạ về thang 1-5.

    - Cold-start (user chưa có rating): fallback sang PopularityRecommender.
    - Chưa có artifact ML-32m: cf=0, vẫn chạy được nhờ content + popularity.
    """

    def __init__(self):
        self.popularity_engine = PopularityRecommender()

    def _fallback_to_popularity(self, conn, top_k):
        """Cold-start / hết ứng viên: dùng popularity nhưng giữ field predicted_score cho nhất quán."""
        recs = self.popularity_engine.recommend(conn, top_k=top_k)
        for r in recs:
            r["predicted_score"] = r.pop("score", round(r.get("rating_avg") or 0.0, 2))
        return recs

    def recommend(
        self,
        conn: sqlite3.Connection,
        user_id: int = None,
        movie_id: int = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not user_id:
            return self._fallback_to_popularity(conn, top_k)

        cursor = conn.cursor()
        user_ratings = cursor.execute(
            "SELECT movie_id, rating FROM ratings WHERE user_id = ?;", (user_id,)
        ).fetchall()

        if not user_ratings:
            return self._fallback_to_popularity(conn, top_k)

        rated = {r["movie_id"]: r["rating"] for r in user_ratings}
        liked = {mid: r for mid, r in rated.items() if r >= LIKE_THRESHOLD}
        if not liked:
            liked = dict(rated)

        movies = fetch_movies_with_genres(conn)
        genre_map = {m["id"]: m["genre_ids"] for m in movies}
        sim_mat = get_similarity()
        pop = get_popularity()

        # Hồ sơ thể loại của user (trung bình rating theo từng thể loại đã thích)
        genre_sum: Dict[int, float] = {}
        genre_cnt: Dict[int, int] = {}
        for mid, r in liked.items():
            for gid in genre_map.get(mid, []):
                genre_sum[gid] = genre_sum.get(gid, 0.0) + r
                genre_cnt[gid] = genre_cnt.get(gid, 0) + 1
        user_genre = {g: genre_sum[g] / genre_cnt[g] for g in genre_sum}

        candidates = []
        for row in movies:
            mid = row["id"]
            if mid in rated:
                continue  # đã đánh giá thì không gợi ý lại

            # 1. Content score (cosine thể loại, [0,1])
            content = _genre_cosine(user_genre, genre_map.get(mid, []))

            # 2. Collaborative score (item-item, chuẩn hóa về [0,1])
            if sim_mat is not None:
                num = 0.0
                den = 0.0
                for j, rj in liked.items():
                    s = sim_mat.get(mid, j)
                    num += s * rj
                    den += abs(s)
                cf_raw = (num / den) if den > 0 else 0.0
                cf = max(0.0, min(RATING_SCALE, cf_raw)) / RATING_SCALE
            else:
                cf = 0.0

            # 3. Popularity (giá trị thô, sẽ min-max ở bước sau)
            if pop and mid in pop:
                pop_val = pop[mid].get("weighted_rating") or 0.0
                row["rating_avg"] = pop[mid].get("mean_rating") or 0.0
                row["vote_count"] = pop[mid].get("vote_count") or 0
            else:
                pop_val = row["rating_avg"] or 0.0

            candidates.append(
                {"row": row, "content": content, "cf": cf, "pop_val": pop_val}
            )

        if not candidates:
            return self._fallback_to_popularity(conn, top_k)

        # Chuẩn hóa popularity theo min-max trong tập ứng viên
        pop_norm = minmax([c["pop_val"] for c in candidates])

        results = []
        for c, pn in zip(candidates, pop_norm):
            weighted = W_CF * c["cf"] + W_CONTENT * c["content"] + W_POP * pn
            predicted = round(1 + 4 * weighted, 2)
            results.append(
                build_movie_payload(c["row"], "predicted_score", predicted)
            )

        results.sort(key=lambda x: (-x["predicted_score"], x["id"]))
        return results[:top_k]
