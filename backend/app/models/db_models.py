from pydantic import BaseModel
from typing import List, Optional

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150"

class UserResponse(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    avatar_url: Optional[str]

class MovieResponse(BaseModel):
    id: int
    title: str
    overview: Optional[str]
    release_year: Optional[int]
    poster_url: Optional[str]
    backdrop_url: Optional[str]
    trailer_youtube_id: Optional[str]
    duration_minutes: Optional[int]
    rating_avg: float
    vote_count: int
    genres: List[str] = []
    user_rating: Optional[float] = None

class RateMovieRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float

class GenreResponse(BaseModel):
    id: int
    name: str
