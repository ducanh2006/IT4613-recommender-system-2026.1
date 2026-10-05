-- Database Schema for Movie Recommender System (Netflix Style)
-- Course: IT4613 - Recommender Systems

-- 1. Bảng Người dùng (Hỗ trợ Đăng nhập / Đăng ký đơn giản, mật khẩu băm bcrypt)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL DEFAULT '123456',
    full_name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng Thể loại phim
CREATE TABLE IF NOT EXISTS genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

-- 3. Bảng Phim
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    overview TEXT,
    release_year INTEGER,
    poster_url TEXT,
    backdrop_url TEXT,
    trailer_youtube_id TEXT,
    duration_minutes INTEGER,
    rating_avg REAL DEFAULT 0.0,
    vote_count INTEGER DEFAULT 0
);

-- 4. Bảng liên kết Phim - Thể loại (N-N)
CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id INTEGER,
    genre_id INTEGER,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);

-- 5. Bảng Tương tác người dùng (Ratings phục vụ huấn luyện Recommender)
CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    movie_id INTEGER,
    rating REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE,
    UNIQUE(user_id, movie_id)
);

-- 6. Bảng Danh sách yêu thích / Xem sau (Watchlist)
CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER,
    movie_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
);
