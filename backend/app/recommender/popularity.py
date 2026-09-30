import sqlite3
from typing import List, Dict, Any
from .base import BaseRecommender

class PopularityRecommender(BaseRecommender):
    """
    Thuật toán gợi ý dựa trên độ phổ biến (Popularity-based / Trending)
    Áp dụng công thức IMDB Weighted Rating (WR):
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    Xử lý rất tốt cho bài toán Cold-Start khi người dùng mới chưa có lịch sử đánh giá.
    """
    def __init__(self, m: int = 5):
        self.m = m # Ngưỡng số lượng đánh giá tối thiểu

    def recommend(self, conn: sqlite3.Connection, user_id: int = None, movie_id: int = None, top_k: int = 10) -> List[Dict[str, Any]]:
        cursor = conn.cursor()
        
        # 1. Tính điểm trung bình C toàn hệ thống
        avg_row = cursor.execute("SELECT AVG(rating_avg) FROM movies WHERE vote_count > 0;").fetchone()
        C = avg_row[0] if (avg_row and avg_row[0]) else 4.0

        # 2. Truy vấn tất cả phim cùng thể loại
        movies = cursor.execute("""
            SELECT m.*, GROUP_CONCAT(g.name, ', ') AS genre_names
            FROM movies m
            LEFT JOIN movie_genres mg ON m.id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.id
            GROUP BY m.id
        """).fetchall()

        scored_movies = []
        for row in movies:
            v = row["vote_count"] or 0
            R = row["rating_avg"] or 0.0
            # Công thức IMDB
            wr = (v / (v + self.m)) * R + (self.m / (v + self.m)) * C
            
            genres_list = row["genre_names"].split(", ") if row["genre_names"] else []
            movie_dict = {
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
                "score": round(wr, 3)
            }
            scored_movies.append(movie_dict)

        # Sắp xếp giảm dần theo điểm Weighted Rating
        scored_movies.sort(key=lambda x: x["score"], reverse=True)
        return scored_movies[:top_k]
