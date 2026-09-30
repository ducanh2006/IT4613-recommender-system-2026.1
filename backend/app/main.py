from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import auth, movies, recommendations, interactions, users

app = FastAPI(
    title="Netflix Movie Recommender System (IT4613)",
    description="Hệ thống gợi ý phim trực tuyến phong cách Netflix - Môn học IT4613 Hệ Gợi Ý",
    version="1.0.0"
)

# Cấu hình CORS để Frontend ReactJS (Vite) có thể gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router API
app.include_router(auth.router)
app.include_router(movies.router)
app.include_router(recommendations.router)
app.include_router(interactions.router)
app.include_router(users.router)

@app.get("/")
def root():
    return {
        "message": "Chào mừng đến với API Hệ gợi ý phim IT4613!",
        "docs_url": "/docs",
        "status": "online"
    }
