"""
Các hàm tiện ích dùng chung cho các recommender.
"""
import sqlite3
from typing import List, Dict, Any


def fetch_movies_with_genres(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Một query duy nhất lấy toàn bộ phim kèm thể loại.

    Trả về list các dict có sẵn:
      - các cột gốc của bảng movies (id, title, overview, ...)
      - "genre_ids": list[int]
      - "genres":    list[str] (tên thể loại)
    """
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT m.*, GROUP_CONCAT(mg.genre_id) AS genre_ids,
               GROUP_CONCAT(g.name, ', ') AS genre_names
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        GROUP BY m.id
    """).fetchall()

    movies = []
    for row in rows:
        d = dict(row)
        genre_ids = (
            [int(x) for x in row["genre_ids"].split(",")]
            if row["genre_ids"] else []
        )
        genre_names = (
            [x.strip() for x in row["genre_names"].split(", ")]
            if row["genre_names"] else []
        )
        d["genre_ids"] = genre_ids
        d["genres"] = genre_names
        movies.append(d)
    return movies


def minmax(values):
    """Chuẩn hóa min-max về [0, 1]; trả 0 nếu rỗng hoặc min == max (tránh chia 0)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_movie_payload(row: Dict[str, Any], score_field: str, score: float) -> Dict[str, Any]:
    """Đóng gói thông tin phim trả về cho API (giữ nguyên các field hiển thị)."""
    return {
        "id": row["id"],
        "title": row["title"],
        "overview": row["overview"],
        "release_year": row["release_year"],
        "poster_url": row["poster_url"],
        "backdrop_url": row["backdrop_url"],
        "trailer_youtube_id": row["trailer_youtube_id"],
        "duration_minutes": row["duration_minutes"],
        "rating_avg": round(row["rating_avg"] or 0.0, 1),
        "vote_count": row["vote_count"] or 0,
        "genres": row.get("genres") or [],
        score_field: score,
    }
