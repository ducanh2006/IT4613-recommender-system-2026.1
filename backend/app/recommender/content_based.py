import sqlite3
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    """
    Thuật toán lọc dựa trên nội dung (Content-based Filtering)
    Tính toán độ tương đồng Cosine Similarity giữa các phim dựa trên vector đặc trưng Thể loại (Genres).
    Dùng cho tính năng: "Vì bạn đã xem/thích phim X" hoặc "Phim tương tự".
    """
    def recommend(
            self,
            conn: sqlite3.Connection,
            user_id: int = None,
            movie_id: int = None,
            top_k: int = 10
        ) -> List[Dict[str, Any]]:
        if not movie_id:
            return []

        cursor = conn.cursor()
        # 1. Lấy danh sách tất cả thể loại để xây dựng vector one-hot
        all_genres = [r[0] for r in cursor.execute("SELECT id FROM genres ORDER BY id;").fetchall()]
        genre_to_idx = {gid: idx for idx, gid in enumerate(all_genres)}
        num_genres = len(all_genres)

        # 2. Lấy danh sách phim kèm thể loại
        movies_query = cursor.execute("""
            SELECT m.*, GROUP_CONCAT(mg.genre_id) AS genre_ids, GROUP_CONCAT(g.name, ', ') AS genre_names
            FROM movies m
            LEFT JOIN movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.id
            GROUP BY m.id
        """).fetchall()

        movie_rows = []
        movie_vectors = []
        target_idx = -1

        for idx, row in enumerate(movies_query):
            movie_rows.append(row)
            vec = np.zeros(num_genres)
            if row["genre_ids"]:
                gids = [int(x) for x in row["genre_ids"].split(",")]
                for gid in gids:
                    if gid in genre_to_idx:
                        vec[genre_to_idx[gid]] = 1.0
            movie_vectors.append(vec)
            if row["id"] == movie_id:
                target_idx = idx

        if target_idx == -1:
            return []

        # 3. Tính độ đo Cosine Similarity
        matrix = np.array(movie_vectors)
        target_vec = matrix[target_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vec, matrix)[0]

        # 4. Gom kết quả và sắp xếp (bỏ qua chính nó)
        results = []
        for idx, score in enumerate(sim_scores):
            if idx == target_idx:
                continue
            row = movie_rows[idx]
            genres_list = row["genre_names"].split(", ") if row["genre_names"] else []
            results.append({
                "id": row["id"],
                "title": row["title"],
                "overview": row["overview"],
                "release_year": row["release_year"],
                "poster_url": row["poster_url"],
                "backdrop_url": row["backdrop_url"],
                "trailer_youtube_id": row["trailer_youtube_id"],
                "duration_minutes": row["duration_minutes"],
                "rating_avg": round(row["rating_avg"], 1),
                "vote_count": row["vote_count"],
                "genres": genres_list,
                "similarity_score": round(float(score), 3)
            })

        # Sắp xếp giảm dần theo similarity_score rồi đến rating_avg
        results.sort(key=lambda x: (x["similarity_score"], x["rating_avg"]), reverse=True)
        return results[:top_k]
