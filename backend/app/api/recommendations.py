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
collab_engine = CollaborativeRecommender()

@router.get("/trending")
def get_trending_movies(top_k: int = Query(10, ge=1, le=50), db: sqlite3.Connection = Depends(get_db)):
    """
    Hàng 1: Top thịnh hành (Popularity-based Recommender - IMDB Weighted Rating)
    """
    return popularity_engine.recommend(conn=db, top_k=top_k)

@router.get("/for-you")
def get_personalized_movies(user_id: Optional[int] = None, top_k: int = Query(10, ge=1, le=50), db: sqlite3.Connection = Depends(get_db)):
    """
    Hàng 2: Gợi ý cá nhân hóa dành riêng cho User (Collaborative / Personalized)
    - Nếu là Cold-start User (chưa có rating) -> Tự động fallback sang Popularity
    """
    return collab_engine.recommend(conn=db, user_id=user_id, top_k=top_k)

@router.get("/similar/{movie_id}")
def get_similar_movies(movie_id: int, top_k: int = Query(10, ge=1, le=50), db: sqlite3.Connection = Depends(get_db)):
    """
    Hàng 3: Phim tương tự / Vì bạn đã xem phim X (Content-based Recommender - Cosine Similarity)
    """
    return content_engine.recommend(conn=db, movie_id=movie_id, top_k=top_k)
