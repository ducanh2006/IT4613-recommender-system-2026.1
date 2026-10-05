from fastapi import APIRouter, HTTPException, Depends
import sqlite3
from ..database import get_db
from ..models.db_models import UserLogin, UserRegister, UserResponse
import bcrypt

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse)
def register(user_data: UserRegister, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    # Kiểm tra username đã tồn tại chưa
    existing = cursor.execute("SELECT id FROM users WHERE username = ?;", (user_data.username,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")

    hashed_pw = bcrypt.hashpw(
        user_data.password.encode(),
        bcrypt.gensalt(),
    ).decode()

    cursor.execute("""
        INSERT INTO users (username, password, full_name, avatar_url)
        VALUES (?, ?, ?, ?);
    """, (user_data.username, hashed_pw, user_data.full_name or user_data.username, user_data.avatar_url))
    db.commit()
    new_id = cursor.lastrowid

    row = cursor.execute("SELECT id, username, full_name, avatar_url FROM users WHERE id = ?;", (new_id,)).fetchone()
    return dict(row)

@router.post("/login", response_model=UserResponse)
def login(
    credentials: UserLogin,
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()

    user = cursor.execute("""
        SELECT id, username, password, full_name, avatar_url
        FROM users
        WHERE username = ?;
    """, (credentials.username,)).fetchone()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Sai ten dang nhap hoac mat khau",
        )
    try:
        ok = bcrypt.checkpw(
            credentials.password.encode(),
            user["password"].encode(),
        )
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Sai ten dang nhap hoac mat khau",
        )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="sai ten dang nhap hoac mat khau"
        )
    
    return dict(user)

