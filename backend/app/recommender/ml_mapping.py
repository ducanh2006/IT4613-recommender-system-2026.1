"""
Bảng ánh xạ thủ công giữa 20 phim demo (id trong app.db) và movieId của MovieLens-32m.

MovieLens-32m không có poster/overview/trailer nên UI giữ nguyên catalog 20 phim demo.
Bảng này cho phép tầng gợi ý dùng tín hiệu thật (rating, co-rating) từ ML-32m để chấm điểm
cho 20 phim demo đó.

Lưu ý một số tên trong movies.csv của ML-32m khác tên hiển thị:
  - "The Dark Knight" -> "Dark Knight, The (2008)"
  - "The Matrix"      -> "Matrix, The (1999)"
  - "Your Name"       -> "Your Name. (2016)"
  - "Avengers: Endgame" bị ML-32m lưu nhầm thành "Avengers: Infinity War - Part II (2019)"
"""

# demo_movie_id -> ml_movie_id
DEMO_TO_ML = {
    1: 122914,  # Avengers: Endgame
    2: 79132,   # Inception
    3: 109487,  # Interstellar
    4: 263007,  # Spider-Man: No Way Home
    5: 5618,    # Spirited Away
    6: 163134,  # Your Name
    7: 58559,   # The Dark Knight
    8: 254726,  # Dune
    9: 287699,  # Oppenheimer
    10: 201588, # Toy Story 4
    11: 272525, # Doctor Strange in the Multiverse of Madness
    12: 231413, # Demon Slayer: Mugen Train
    13: 2571,   # The Matrix
    14: 1721,   # Titanic
    15: 185029, # A Quiet Place
    16: 202439, # Parasite
    17: 177765, # Coco
    18: 122904, # Deadpool
    19: 283873, # John Wick: Chapter 4
    20: 280218, # Avatar: The Way of Water
}

# ml_movie_id -> demo_movie_id
ML_TO_DEMO = {ml_id: demo_id for demo_id, ml_id in DEMO_TO_ML.items()}

# Danh sách id demo (theo thứ tự id tăng dần)
ALL_DEMO_IDS = sorted(DEMO_TO_ML.keys())


def demo_to_ml(demo_id: int) -> int:
    """Trả ml_movie_id tương ứng, hoặc None nếu không map được."""
    return DEMO_TO_ML.get(demo_id)


def ml_to_demo(ml_id: int) -> int:
    """Trả demo_movie_id tương ứng, hoặc None nếu không map được."""
    return ML_TO_DEMO.get(ml_id)
