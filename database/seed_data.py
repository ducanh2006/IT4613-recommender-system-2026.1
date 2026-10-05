import sqlite3
import os
import sys
import bcrypt

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_FILE = os.path.join(DB_DIR, "database.sql")
DB_FILE = os.path.join(DB_DIR, "app.db")

def hash_password(raw_password):
    """Băm mật khẩu bằng bcrypt để khớp với API đăng nhập."""
    return bcrypt.hashpw(raw_password.encode(), bcrypt.gensalt()).decode()

def init_database():
    print(f"[*] Đang khởi tạo CSDL tại: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        cursor.executescript(f.read())
    
    # 1. Thể loại
    genres = [
        (1, "Hành Động"),
        (2, "Khoa Học Viễn Tưởng"),
        (3, "Phiêu Lưu"),
        (4, "Hoạt Hình"),
        (5, "Tâm Lý"),
        (6, "Kinh Dị"),
        (7, "Hài Hước"),
        (8, "Lãng Mạn")
    ]
    cursor.executemany("INSERT OR IGNORE INTO genres (id, name) VALUES (?, ?);", genres)

    # 2. Người dùng mẫu (mật khẩu băm bcrypt, dùng để đăng nhập demo)
    demo_password = "123456"
    users = [
        (1, "alice", hash_password(demo_password), "Alice Nguyen (Mê Hành Động/Marvel)", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150"),
        (2, "bob", hash_password(demo_password), "Bob Tran (Mê Hoạt Hình/Anime)", "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150"),
        (3, "dave", hash_password(demo_password), "Dave Pham (Mê Viễn Tưởng/Không Gian)", "https://images.unsplash.com/photo-1570295999919-56ceb5ecca61?w=150"),
        (4, "charlie", hash_password(demo_password), "Charlie Le (User mới - Demo Cold Start)", "https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=150")
    ]
    cursor.executemany("INSERT OR IGNORE INTO users (id, username, password, full_name, avatar_url) VALUES (?, ?, ?, ?, ?);", users)

    # Nâng cấp mật khẩu plain text còn sót trong app.db sang hash bcrypt
    hashed_legacy = 0
    for user_id, stored_password in cursor.execute("SELECT id, password FROM users;").fetchall():
        if not stored_password.startswith("$2"):
            cursor.execute("UPDATE users SET password = ? WHERE id = ?;", (hash_password(stored_password), user_id))
            hashed_legacy += 1
    if hashed_legacy:
        print(f"[*] Đã băm lại {hashed_legacy} mật khẩu plain text cũ.")

    # 3. Danh sách phim (Dữ liệu chuẩn nét, poster và trailer youtube thật)
    movies = [
        # (id, title, overview, release_year, poster_url, backdrop_url, trailer_youtube_id, duration_minutes)
        (
            1, "Avengers: Endgame",
            "Sau những sự kiện tàn khốc của Infinity War, vũ trụ đang chìm trong đống đổ nát. Với sự trợ giúp của các đồng minh còn lại, biệt đội Avengers tập hợp một lần nữa để đảo ngược hành động của Thanos.",
            2019,
            "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg",
            "https://image.tmdb.org/t/p/original/7RyHsO4yDXtBv1zUU3mTpHeQ0d5.jpg",
            "TcMBFSGVi1c", 181
        ),
        (
            2, "Inception",
            "Một tên trộm có khả năng đi vào giấc mơ của người khác để đánh cắp bí mật được trao cơ hội xóa sạch hồ sơ tội phạm nếu có thể gieo một ý tưởng vào tâm trí của một CEO.",
            2010,
            "https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
            "https://image.tmdb.org/t/p/original/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
            "YoHD9XEInc0", 148
        ),
        (
            3, "Interstellar",
            "Khi Trái Đất dần trở nên không thể sinh sống được, một nhóm nhà thám hiểm du hành qua hố sâu không gian để tìm kiếm hành tinh mới cho nhân loại.",
            2014,
            "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
            "https://image.tmdb.org/t/p/original/xJHokMbljvjADYdit5fK5VQsXEG.jpg",
            "zSWdZVtXT7E", 169
        ),
        (
            4, "Spider-Man: No Way Home",
            "Với danh tính Người Nhện bị bại lộ, Peter Parker tìm kiếm sự giúp đỡ của Doctor Strange, vô tình mở ra đa vũ trụ đầy hiểm nguy.",
            2021,
            "https://image.tmdb.org/t/p/w500/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg",
            "https://image.tmdb.org/t/p/original/14QbnygCuTO0vl7CAFmPf1fgZfV.jpg",
            "JfVOs4VSpmA", 148
        ),
        (
            5, "Spirited Away (Vùng Đất Linh Hồn)",
            "Cô bé Chihiro 10 tuổi lạc vào thế giới của linh hồn và phù thủy sau khi cha mẹ bị biến thành heo. Cô phải dũng cảm tìm cách giải cứu gia đình.",
            2001,
            "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
            "https://image.tmdb.org/t/p/original/bSXfSemFtIanWEwoOpJDXL47jnE.jpg",
            "ByXuk9QqQkk", 125
        ),
        (
            6, "Your Name (Kimi no Na wa)",
            "Hai thanh thiếu niên xa lạ bỗng nhiên bị hoán đổi thân xác cho nhau qua những giấc mơ và cố gắng tìm kiếm danh tính của đối phương xuyên thời gian.",
            2016,
            "https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6babgKnONONX.jpg",
            "https://image.tmdb.org/t/p/original/dIWwZW7dJJ1742qSYElLqYYr3NV.jpg",
            "s0wTdCQoc8A", 106
        ),
        (
            7, "The Dark Knight",
            "Batman cùng Trung tá Gordon và Công tố viên Harvey Dent liên minh chống lại tên tội phạm tâm thần khét tiếng Joker đang gieo rắc hỗn loạn tại Gotham.",
            2008,
            "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
            "https://image.tmdb.org/t/p/original/dqK9Hag1054tghRQSqLSfrkvQnA.jpg",
            "EXeTwQWrcwY", 152
        ),
        (
            8, "Dune: Hành Tinh Cát",
            "Paul Atreides, một chàng trai trẻ thông minh và tài năng, phải du hành đến hành tinh nguy hiểm nhất vũ trụ để bảo vệ tương lai của gia đình và nhân dân.",
            2021,
            "https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg",
            "https://image.tmdb.org/t/p/original/lzWHmYZrBtTP9uc89NsiyvMrgei.jpg",
            "8g18jFHCLXk", 155
        ),
        (
            9, "Oppenheimer",
            "Câu chuyện về nhà vật lý lý thuyết J. Robert Oppenheimer, người đứng đầu Dự án Manhattan phát triển bom nguyên tử trong Thế chiến II.",
            2023,
            "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
            "https://image.tmdb.org/t/p/original/fm6K9vY9wgjhCwTzbamUZ20pmY9.jpg",
            "uYPbbksJxIg", 180
        ),
        (
            10, "Toy Story 4 (Câu Chuyện Đồ Chơi 4)",
            "Woody, Buzz Lightyear và nhóm đồ chơi bắt đầu một chuyến dã ngoại cùng người chủ mới Bonnie và một người bạn đồ chơi tự tạo tên là Forky.",
            2019,
            "https://image.tmdb.org/t/p/w500/w9kR8qbmQ01HwnvK4alvnQ2v0Zw.jpg",
            "https://image.tmdb.org/t/p/original/m67smI195viDQKa9GU04qUIKHdH.jpg",
            "wmiIUN-7qhE", 100
        ),
        (
            11, "Doctor Strange in the Multiverse of Madness",
            "Doctor Strange hợp tác với một thiếu nữ có khả năng du hành xuyên đa vũ trụ để chiến đấu chống lại hiểm họa đa chiều.",
            2022,
            "https://image.tmdb.org/t/p/w500/9Gtg2DzBhmYamXBS1hKAhiwbBKS.jpg",
            "https://image.tmdb.org/t/p/original/wcKFYgnvdvRURwhEjYVJRhq129b.jpg",
            "aWzlQ2N6qqg", 126
        ),
        (
            12, "Demon Slayer: Mugen Train",
            "Tanjiro và các đồng đội cùng Viêm Trụ Rengoku Kyoujurou lên chuyến tàu Vô Tận để điều tra các vụ mất tích bí ẩn do quỷ gây ra.",
            2020,
            "https://image.tmdb.org/t/p/w500/h8Rb9vKitWwhdUG7drqP13qEjM8.jpg",
            "https://image.tmdb.org/t/p/original/xPpXYnCWcuumYnG7wvgudNmRPVU.jpg",
            "ATJYac_dORw", 117
        ),
        (
            13, "The Matrix (Ma Trận)",
            "Một lập trình viên phát hiện ra thế giới thực chỉ là một mô phỏng máy tính tinh vi do máy móc thống trị tạo ra nhằm khai thác loài người.",
            1999,
            "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
            "https://image.tmdb.org/t/p/original/7u3fhRwF6NmIPq1g210uom999Mm.jpg",
            "vKQi3bBA1y8", 136
        ),
        (
            14, "Titanic",
            "Mối tình lãng mạn giữa một chàng nghệ sĩ nghèo và một tiểu thư quý tộc trên con tàu định mệnh Titanic vượt Đại Tây Dương.",
            1997,
            "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
            "https://image.tmdb.org/t/p/original/yDI6D5ZQh67YU4r2ms8qcSbAviZ.jpg",
            "kVrqfYjkTdQ", 194
        ),
        (
            15, "A Quiet Place (Vùng Đất Câm Lặng)",
            "Một gia đình phải sống trong sự im lặng tuyệt đối để trốn thoát những sinh vật mù nhưng có thính giác siêu phàm săn lùng theo âm thanh.",
            2018,
            "https://image.tmdb.org/t/p/w500/nAU74GmpUk7t5iklEp3bufwDq4n.jpg",
            "https://image.tmdb.org/t/p/original/roYyPiQDytnk0yuHI6RQ1uLUI5i.jpg",
            "WR7cc5t7tv8", 90
        ),
        (
            16, "Parasite (Ký Sinh Trùng)",
            "Một gia đình nghèo từng bước thâm nhập vào cuộc sống của một gia đình giàu có, dẫn đến chuỗi biến cố bi hài kịch không thể lường trước.",
            2019,
            "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
            "https://image.tmdb.org/t/p/original/TU9NIjwzjoKPwQHoHshkFcQUCG.jpg",
            "5xH0R_uieTY", 132
        ),
        (
            17, "Coco",
            "Cậu bé Miguel với niềm đam mê âm nhạc vô tình bước chân vào Vùng Đất Linh Hồn rực rỡ và khám phá bí mật lịch sử gia đình.",
            2017,
            "https://image.tmdb.org/t/p/w500/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg",
            "https://image.tmdb.org/t/p/original/askg3SMvhqEl4OL52YuvdtQw40Y.jpg",
            "Rvr68u6k5sI", 105
        ),
        (
            18, "Deadpool",
            "Một cựu lính đánh thuê được trao khả năng hồi phục phi thường sau một thí nghiệm bất hợp pháp, lên đường săn lùng kẻ đã hủy hoại diện mạo của mình.",
            2016,
            "https://image.tmdb.org/t/p/w500/3E53WEZJqP6aM84D8C4O2Z9ne.jpg",
            "https://image.tmdb.org/t/p/original/en971MEXui9vgirapIvgSYOkRJ1.jpg",
            "ONHBaC-pfsk", 108
        ),
        (
            19, "John Wick: Chapter 4",
            "Sát thủ John Wick tìm ra con đường để đánh bại High Table, nhưng trước khi tự do, anh phải đối đầu với một kẻ thù quyền lực mới.",
            2023,
            "https://image.tmdb.org/t/p/w500/vZloFAK7NKnMGKEslUsZjhC9ph8.jpg",
            "https://image.tmdb.org/t/p/original/7I6VUdPj6tQECNHdviJkUHD2389.jpg",
            "qEVUtrk8_B4", 169
        ),
        (
            20, "Avatar: The Way of Water",
            "Jake Sully và Ney'tiri cùng gia đình rời khỏi khu rừng để nương tựa vào các bộ tộc sống gần đại dương của hành tinh Pandora.",
            2022,
            "https://image.tmdb.org/t/p/w500/t6HIqrRAclMCA60NsSmeqe9RmNV.jpg",
            "https://image.tmdb.org/t/p/original/s16H6tpK2utvwDtzZ8Qy4qm5Emw.jpg",
            "d9MyW72ELq0", 192
        )
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO movies (id, title, overview, release_year, poster_url, backdrop_url, trailer_youtube_id, duration_minutes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, movies)

    # 4. Gán Thể loại cho Phim (movie_genres)
    # 1: Action, 2: Sci-Fi, 3: Adventure, 4: Animation, 5: Drama, 6: Horror, 7: Comedy, 8: Romance
    movie_genres_data = [
        (1, 1), (1, 2), (1, 3), # Avengers: Action, Sci-Fi, Adventure
        (2, 2), (2, 1), (2, 5), # Inception: Sci-Fi, Action, Drama
        (3, 2), (3, 5), (3, 3), # Interstellar: Sci-Fi, Drama, Adventure
        (4, 1), (4, 3), (4, 2), # Spider-Man: Action, Adventure, Sci-Fi
        (5, 4), (5, 3), (5, 5), # Spirited Away: Animation, Adventure, Drama
        (6, 4), (6, 8), (6, 5), # Your Name: Animation, Romance, Drama
        (7, 1), (7, 5),         # The Dark Knight: Action, Drama
        (8, 2), (8, 3), (8, 5), # Dune: Sci-Fi, Adventure, Drama
        (9, 5),                 # Oppenheimer: Drama
        (10, 4), (10, 7), (10, 3), # Toy Story 4: Animation, Comedy, Adventure
        (11, 1), (11, 2), (11, 6), # Doctor Strange 2: Action, Sci-Fi, Horror
        (12, 4), (12, 1), (12, 3), # Demon Slayer: Animation, Action, Adventure
        (13, 2), (13, 1),       # Matrix: Sci-Fi, Action
        (14, 8), (14, 5),       # Titanic: Romance, Drama
        (15, 6), (15, 2), (15, 5), # A Quiet Place: Horror, Sci-Fi, Drama
        (16, 5), (16, 7),       # Parasite: Drama, Comedy
        (17, 4), (17, 3), (17, 7), # Coco: Animation, Adventure, Comedy
        (18, 1), (18, 7), (18, 2), # Deadpool: Action, Comedy, Sci-Fi
        (19, 1), (19, 5),       # John Wick 4: Action, Drama
        (20, 2), (20, 1), (20, 3)  # Avatar 2: Sci-Fi, Action, Adventure
    ]
    cursor.executemany("INSERT OR IGNORE INTO movie_genres (movie_id, genre_id) VALUES (?, ?);", movie_genres_data)

    # 5. Dữ liệu đánh giá mẫu (Ratings)
    # Alice (id=1): Mê Hành động & Marvel (1, 4, 7, 11, 18, 19)
    # Bob (id=2): Mê Animation & Anime (5, 6, 10, 12, 17)
    # Dave (id=3): Mê Sci-Fi / Vũ trụ (2, 3, 8, 13, 20)
    # Charlie (id=4): User mới tinh -> 0 rating (Để demo Cold-Start)
    ratings_data = [
        # Alice
        (1, 1, 5.0), # Avengers
        (1, 4, 5.0), # Spider-Man
        (1, 7, 4.5), # Dark Knight
        (1, 11, 4.0),# Doctor Strange
        (1, 18, 5.0),# Deadpool
        (1, 19, 4.5),# John Wick
        # Bob
        (2, 5, 5.0), # Spirited Away
        (2, 6, 5.0), # Your Name
        (2, 10, 4.5),# Toy Story 4
        (2, 12, 5.0),# Demon Slayer
        (2, 17, 4.5),# Coco
        # Dave
        (3, 2, 5.0), # Inception
        (3, 3, 5.0), # Interstellar
        (3, 8, 4.5), # Dune
        (3, 13, 5.0),# Matrix
        (3, 20, 4.5),# Avatar
        (3, 9, 4.5)  # Oppenheimer
    ]
    cursor.executemany("INSERT OR REPLACE INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?);", ratings_data)

    # 6. Cập nhật lại rating_avg và vote_count cho bảng movies
    cursor.execute("""
        UPDATE movies
        SET rating_avg = COALESCE((SELECT AVG(rating) FROM ratings WHERE ratings.movie_id = movies.id), 4.2),
            vote_count = COALESCE((SELECT COUNT(rating) FROM ratings WHERE ratings.movie_id = movies.id), 1) + 120
    """)

    conn.commit()
    conn.close()
    print("[+] Hoàn tất nạp dữ liệu mẫu thành công!")

if __name__ == "__main__":
    init_database()
