import sqlite3
import numpy as np
from typing import List, Dict, Any
from .base import BaseRecommender
from .popularity import PopularityRecommender

class CollaborativeRecommender(BaseRecommender):
    """
    Gợi ý phim cá nhân hóa theo User (Personalized Recommender)
    - Nếu User chưa có đánh giá nào (Cold-start User): Tự động fallback sang PopularityRecommender.
    - Nếu User đã có đánh giá:
        + Tìm các thể loại và bộ phim người dùng yêu thích (rating >= 3.5)
        + Tìm các người dùng khác có sở thích tương đồng (User-based heuristic)
        + Dự đoán và xếp hạng các bộ phim chưa xem phù hợp nhất với hồ sơ người dùng.
    """
    def __init__(self):
        self.popularity_engine = PopularityRecommender()

    def recommend(self, conn: sqlite3.Connection, user_id: int = None, movie_id: int = None, top_k: int = 10) -> List[Dict[str, Any]]:
        if not user_id:
            return self.popularity_engine.recommend(conn, top_k=top_k)

        cursor = conn.cursor()
        
        # 1. Lấy lịch sử đánh giá của user
        user_ratings = cursor.execute("""
            SELECT movie_id, rating FROM ratings WHERE user_id = ?;
        """, (user_id,)).fetchall()

        # Xử lý bài toán Cold-Start
        if not user_ratings:
            print(f"[Recommender] User {user_id} chưa có rating (Cold-start) -> Dùng Popularity Recommender")
            return self.popularity_engine.recommend(conn, top_k=top_k)

        rated_movie_ids = set([r["movie_id"] for r in user_ratings])
        high_rated_movies = [r["movie_id"] for r in user_ratings if r["rating"] >= 3.5]
        if not high_rated_movies:
            high_rated_movies = [r["movie_id"] for r in user_ratings]

        # 2. Xác định các thể loại mà user yêu thích nhất
        fav_genres = cursor.execute(f"""
            SELECT mg.genre_id, COUNT(*) as cnt, AVG(r.rating) as avg_r
            FROM ratings r
            JOIN movie_genres mg ON r.movie_id = mg.movie_id
            WHERE r.user_id = ? AND r.rating >= 3.5
            GROUP BY mg.genre_id
            ORDER BY avg_r DESC, cnt DESC;
        """, (user_id,)).fetchall()
        
        fav_genre_ids = set([g["genre_id"] for g in fav_genres])

        # 3. Lấy tất cả các phim mà user CHƯA đánh giá
        candidate_movies = cursor.execute("""
            SELECT m.*, GROUP_CONCAT(mg.genre_id) AS genre_ids, GROUP_CONCAT(g.name, ', ') AS genre_names
            FROM movies m
            LEFT JOIN movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.id
            GROUP BY m.id
        """).fetchall()

        scored_candidates = []
        for movie in candidate_movies:
            mid = movie["id"]
            if mid in rated_movie_ids:
                continue # Đã đánh giá rồi thì không gợi ý lại

            # Tính điểm tương hợp thể loại (Genre Match Score)
            movie_genres = [int(x) for x in movie["genre_ids"].split(",")] if movie["genre_ids"] else []
            match_count = sum(1 for gid in movie_genres if gid in fav_genre_ids)
            genre_score = match_count * 1.5

            # Tìm xem có user nào khác cũng đánh giá cao phim này mà có chung sở thích với user hiện tại không
            similar_user_ratings = cursor.execute("""
                SELECT AVG(r2.rating)
                FROM ratings r1
                JOIN ratings r2 ON r1.user_id = r2.user_id
                WHERE r1.movie_id IN ({})
                  AND r1.user_id != ?
                  AND r2.movie_id = ?
                  AND r1.rating >= 4.0
            """.format(",".join("?" * len(high_rated_movies))), (*high_rated_movies, user_id, mid)).fetchone()

            cf_score = similar_user_ratings[0] if (similar_user_ratings and similar_user_ratings[0]) else 0.0

            # Tổng hợp điểm dự đoán (Predicted Preference Score)
            # Baseline: Điểm phim + Thể loại phù hợp + Đánh giá từ người dùng tương tự
            base_rating = movie["rating_avg"] or 4.0
            total_score = base_rating * 0.4 + genre_score * 0.4 + cf_score * 0.2

            genres_list = movie["genre_names"].split(", ") if movie["genre_names"] else []
            scored_candidates.append({
                "id": movie["id"],
                "title": movie["title"],
                "overview": movie["overview"],
                "release_year": movie["release_year"],
                "poster_url": movie["poster_url"],
                "backdrop_url": movie["backdrop_url"],
                "trailer_youtube_id": movie["trailer_youtube_id"],
                "duration_minutes": movie["duration_minutes"],
                "rating_avg": round(movie["rating_avg"], 1),
                "vote_count": movie["vote_count"],
                "genres": genres_list,
                "predicted_score": round(float(total_score), 2)
            })

        # Sắp xếp theo điểm dự đoán giảm dần
        scored_candidates.sort(key=lambda x: x["predicted_score"], reverse=True)

        # Nếu không còn phim nào chưa xem thì fallback lại trending
        if not scored_candidates:
            return self.popularity_engine.recommend(conn, top_k=top_k)

        return scored_candidates[:top_k]
