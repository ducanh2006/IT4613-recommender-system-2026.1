from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import sqlite3
from ..database import get_db
from ..models.db_models import RateMovieRequest

router = APIRouter(prefix="/api/interactions", tags=["Interactions"])

@router.post("/rate")
def rate_movie(req: RateMovieRequest, db: sqlite3.Connection = Depends(get_db)):
    """
    Người dùng chấm điểm cho một bộ phim (1.0 -> 5.0 sao)
    Ghi nhận tức thì vào SQLite để Recommender học sở thích mới.
    """
    if req.rating < 0.5 or req.rating > 5.0:
        raise HTTPException(status_code=400, detail="Điểm đánh giá phải từ 0.5 đến 5.0")

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO ratings (user_id, movie_id, rating, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET
            rating = excluded.rating,
            timestamp = CURRENT_TIMESTAMP;
    """, (req.user_id, req.movie_id, req.rating))

    # Cập nhật lại rating_avg của phim
    cursor.execute("""
        UPDATE movies
        SET rating_avg = (SELECT AVG(rating) FROM ratings WHERE movie_id = ?),
            vote_count = (SELECT COUNT(*) FROM ratings WHERE movie_id = ?) + 120
        WHERE id = ?;
    """, (req.movie_id, req.movie_id, req.movie_id))

    db.commit()
    return {"message": "Đánh giá thành công!", "user_id": req.user_id, "movie_id": req.movie_id, "rating": req.rating}

@router.get("/ratings")
def get_user_ratings(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    rows = cursor.execute("""
        SELECT r.movie_id, r.rating, r.timestamp, m.title, m.poster_url, m.rating_avg
        FROM ratings r
        JOIN movies m ON r.movie_id = m.id
        WHERE r.user_id = ?
        ORDER BY r.timestamp DESC;
    """, (user_id,)).fetchall()
    return [dict(r) for r in rows]

@router.post("/watchlist/toggle")
def toggle_watchlist(user_id: int, movie_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    existing = cursor.execute("SELECT 1 FROM watchlist WHERE user_id = ? AND movie_id = ?;", (user_id, movie_id)).fetchone()
    if existing:
        cursor.execute("DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?;", (user_id, movie_id))
        is_bookmarked = False
    else:
        cursor.execute("INSERT INTO watchlist (user_id, movie_id) VALUES (?, ?);", (user_id, movie_id))
        is_bookmarked = True
    db.commit()
    return {"user_id": user_id, "movie_id": movie_id, "is_bookmarked": is_bookmarked}

@router.get("/watchlist")
def get_watchlist(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    rows = cursor.execute("""
        SELECT m.*
        FROM watchlist w
        JOIN movies m ON w.movie_id = m.id
        WHERE w.user_id = ?
        ORDER BY w.created_at DESC;
    """, (user_id,)).fetchall()
    return [dict(r) for r in rows]
