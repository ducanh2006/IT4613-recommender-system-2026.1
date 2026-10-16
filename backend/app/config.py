"""
Cấu hình chung cho tầng recommender (IT4613).

Chứa đường dẫn dataset MovieLens-32m, thư mục artifact precompute,
và các hằng số điều chỉnh thuật toán. Tách riêng để dễ tinh chỉnh
và để ETL / API dùng chung một nguồn tham số duy nhất.
"""
import os

# Thư mục gốc của project (backend/app/config.py -> 3 cấp lên là gốc project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn tới dataset MovieLens-32m (đã giải nén trong repo)
ML_32M_DIR = os.path.join(BASE_DIR, "ml-32m")
RATINGS_CSV = os.path.join(ML_32M_DIR, "ratings.csv")
MOVIES_CSV = os.path.join(ML_32M_DIR, "movies.csv")

# Thư mục chứa artifact đã precompute (tái tạo được bằng ETL, đã gitignore)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "backend", "artifacts")
SIMILARITY_NPZ = os.path.join(ARTIFACTS_DIR, "demo_similarity.npz")
POPULARITY_JSON = os.path.join(ARTIFACTS_DIR, "demo_popularity.json")
MAPPING_JSON = os.path.join(ARTIFACTS_DIR, "ml_mapping.json")

# --- Tham số thuật toán ---
RATING_SCALE = 5.0            # Thang điểm 1-5
LIKE_THRESHOLD = 3.5          # Ngưỡng "thích" khi xét lịch sử rating của user

# Item-item Collaborative Filtering
SHRINKAGE_LAMBDA = 50         # co-rating nhỏ sẽ bị "co" về 0 để tránh nhiễu

# Popularity (IMDB Weighted Rating)
POPULARITY_M = 500            # Ngưỡng số vote tối thiểu cho công thức WR

# Trọng số fusion trong "for-you" (tổng = 1.0)
W_CF = 0.6                    # Item-item collaborative
W_CONTENT = 0.2               # Content (genre cosine)
W_POP = 0.2                   # Popularity
