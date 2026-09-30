from fastapi import APIRouter, Depends
from typing import List
import sqlite3
from ..database import get_db
from ..models.db_models import UserResponse

router = APIRouter(prefix="/api/users", tags=["Users"])

@router.get("", response_model=List[UserResponse])
def get_all_users(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    users = cursor.execute("SELECT id, username, full_name, avatar_url FROM users ORDER BY id;").fetchall()
    return [dict(u) for u in users]

@router.get("/{user_id}", response_model=UserResponse)
def get_user_detail(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    user = cursor.execute("SELECT id, username, full_name, avatar_url FROM users WHERE id = ?;", (user_id,)).fetchone()
    if not user:
        return None
    return dict(user)
