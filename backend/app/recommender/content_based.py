import sqlite3
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender
from .utils import fetch_movies_with_genres, build_movie_payload
from .loader import get_similarity, get_popularity


class ContentBasedRecommender(BaseRecommender):
    """
    "Phim tương tự" / "Vì bạn đã xem phim X".

    - Ưu tiên: item-item collaborative similarity (cosine + shrinkage) precompute
      từ MovieLens-32m -> các phim demo được người dùng thực cùng đánh giá.
    - Fallback: content-based theo thể loại (one-hot genre + cosine), như bản cũ.
    """

    def recommend(
        self,
        conn: sqlite3.Connection,
        user_id: int = None,
        movie_id: int = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not movie_id:
            return []

        movies = fetch_movies_with_genres(conn)
        pop = get_popularity()

        def _decorate(row, score):
            if pop and row["id"] in pop:
                row["rating_avg"] = pop[row["id"]].get("mean_rating") or 0.0
                row["vote_count"] = pop[row["id"]].get("vote_count") or 0
            return build_movie_payload(row, "similarity_score", round(float(score), 3))

        # --- Nhánh chính: item-item similarity từ ML-32m ---
        sim_mat = get_similarity()
        if sim_mat is not None:
            candidates = [
                (row, sim_mat.get(movie_id, row["id"]))
                for row in movies
                if row["id"] != movie_id and sim_mat.get(movie_id, row["id"]) > 0.0
            ]
            if candidates:
                results = [_decorate(row, score) for row, score in candidates]
                results.sort(
                    key=lambda x: (-x["similarity_score"], -x["rating_avg"], x["id"])
                )
                return results[:top_k]

        # --- Fallback: content-based theo thể loại (giữ logic cũ) ---
        cursor = conn.cursor()
        all_genres = [r[0] for r in cursor.execute("SELECT id FROM genres ORDER BY id;").fetchall()]
        genre_to_idx = {gid: idx for idx, gid in enumerate(all_genres)}
        num_genres = len(all_genres)

        vectors = []
        target_vec = None
        for row in movies:
            vec = np.zeros(num_genres)
            for gid in row.get("genre_ids") or []:
                if gid in genre_to_idx:
                    vec[genre_to_idx[gid]] = 1.0
            vectors.append(vec)
            if row["id"] == movie_id:
                target_vec = vec.reshape(1, -1)

        if target_vec is None:
            return []

        matrix = np.array(vectors)
        sim_scores = cosine_similarity(target_vec, matrix)[0]

        results = []
        for idx, row in enumerate(movies):
            if row["id"] == movie_id:
                continue
            results.append(_decorate(row, sim_scores[idx]))

        results.sort(key=lambda x: (-x["similarity_score"], -x["rating_avg"], x["id"]))
        return results[:top_k]
