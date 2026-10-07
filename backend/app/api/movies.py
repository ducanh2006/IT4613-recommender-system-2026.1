from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
import sqlite3
from ..database import get_db
from ..models.db_models import MovieResponse, GenreResponse

router = APIRouter(
    prefix="/api/movies",
    tags=["Movies"]
)

def _format_movie(
    row,
    user_rating=None
):
    genres_list = row["genre_names"].split(", ") if row["genre_names"] else []
    return {
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
        "user_rating": user_rating
    }

@router.get("/banner", response_model=MovieResponse)
def get_banner_movie(
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    # Chọn phim điểm cao, có backdrop đẹp làm banner (ví dụ Inception hoặc Avengers)
    row = cursor.execute("""
        SELECT m.*, GROUP_CONCAT(g.name, ', ') AS genre_names
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.backdrop_url IS NOT NULL AND m.backdrop_url != ''
        GROUP BY m.id
        ORDER BY m.rating_avg DESC
        LIMIT 1;
    """).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy phim banner")
    return _format_movie(row)

@router.get("/genres", response_model=List[GenreResponse])
def get_genres(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    rows = cursor.execute("SELECT id, name FROM genres ORDER BY id;").fetchall()
    return [
        {"id": r["id"], "name": r["name"]} for r in rows
    ]

@router.get("/by-genre/{genre_id}", response_model=List[MovieResponse])
def get_movies_by_genre(
    genre_id: int,
    user_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    rows = cursor.execute("""
        SELECT m.*, GROUP_CONCAT(g.name, ', ') AS genre_names
        FROM movies m
        JOIN movie_genres mg ON m.id = mg.movie_id
        JOIN genres g ON mg.genre_id = g.id
        WHERE m.id IN (SELECT movie_id FROM movie_genres WHERE genre_id = ?)
        GROUP BY m.id
        ORDER BY m.rating_avg DESC;
    """, (genre_id,)).fetchall()

    user_ratings = {}
    if user_id:
        ur_rows = cursor.execute("SELECT movie_id, rating FROM ratings WHERE user_id = ?;", (user_id,)).fetchall()
        user_ratings = {r["movie_id"]: r["rating"] for r in ur_rows}

    return [_format_movie(r, user_ratings.get(r["id"])) for r in rows]

@router.get("/search", response_model=List[MovieResponse])
def search_movies(
    q: str = Query(..., min_length=1),
    user_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    keyword = f"%{q.strip()}%"
    rows = cursor.execute("""
        SELECT m.*, GROUP_CONCAT(g.name, ', ') AS genre_names
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.title LIKE ? OR m.overview LIKE ? OR g.name LIKE ?
        GROUP BY m.id
        ORDER BY m.rating_avg DESC;
    """, (keyword, keyword, keyword)).fetchall()

    user_ratings = {}
    if user_id:
        ur_rows = cursor.execute("SELECT movie_id, rating FROM ratings WHERE user_id = ?;", (user_id,)).fetchall()
        user_ratings = {r["movie_id"]: r["rating"] for r in ur_rows}

    return [_format_movie(r, user_ratings.get(r["id"])) for r in rows]

@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie_detail(
    movie_id: int,
    user_id: Optional[int] = None,
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    row = cursor.execute("""
        SELECT m.*, GROUP_CONCAT(g.name, ', ') AS genre_names
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.id = ?
        GROUP BY m.id;
    """, (movie_id,)).fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy phim")

    user_rating = None
    if user_id:
        ur = cursor.execute("SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?;", (user_id, movie_id)).fetchone()
        if ur:
            user_rating = ur["rating"]

    return _format_movie(row, user_rating)
