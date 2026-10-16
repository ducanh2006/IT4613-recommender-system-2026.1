from fastapi import APIRouter, Depends, Query
from typing import List, Optional
import sqlite3
from ..database import get_db
from ..recommender.popularity import PopularityRecommender
from ..recommender.content_based import ContentBasedRecommender
from ..recommender.collaborative import CollaborativeRecommender

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])

popularity_engine = PopularityRecommender()
content_engine = ContentBasedRecommender()
hybrid_engine = CollaborativeRecommender()

@router.get("/trending")
def get_trending_movies(top_k: int = Query(10, ge=1, le=50), db: sqlite3.Connection = Depends(get_db)):
    """
    Hàng 1: Top thịnh hành.
    Điểm = IMDB Weighted Rating precompute từ MovieLens-32m (vote_count/rating_avg thật),
    fallback tự tính từ SQLite nếu chưa build artifact.
    """
    return popularity_engine.recommend(conn=db, top_k=top_k)

@router.get("/for-you")
def get_personalized_movies(user_id: Optional[int] = None, top_k: int = Query(10, ge=1, le=50), db: sqlite3.Connection = Depends(get_db)):
    """
    Hàng 2: Gợi ý cá nhân hóa (Item-based Collaborative Filtering lai).
    cf_score = item-item similarity (cosine + shrinkage) từ ML-32m;
    trộn với content (genre) + popularity, ánh xạ về thang 1-5.
    User chưa có rating (cold-start) -> tự động fallback sang Popularity.
    """
    return hybrid_engine.recommend(conn=db, user_id=user_id, top_k=top_k)

@router.get("/similar/{movie_id}")
def get_similar_movies(movie_id: int, top_k: int = Query(10, ge=1, le=50), db: sqlite3.Connection = Depends(get_db)):
    """
    Hàng 3: "Phim tương tự / Vì bạn đã xem phim X".
    Ưu tiên item-item similarity từ ML-32m; fallback content-based (genre cosine).
    """
    return content_engine.recommend(conn=db, movie_id=movie_id, top_k=top_k)
