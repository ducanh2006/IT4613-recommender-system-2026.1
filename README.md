# Hướng Dẫn Chạy Dự Án - Hệ Gợi Ý Phim (IT4613)

## Kiến trúc gợi ý (tóm tắt)

Hệ thống dùng **MovieLens-32m** (32.000.204 rating, 200.948 user, 87.585 phim) làm
"bộ não" gợi ý, nhưng UI vẫn hiển thị **20 phim demo** có poster/trailer.

| Endpoint | Thuật toán | Nguồn tín hiệu |
| :--- | :--- | :--- |
| `/api/recommendations/trending` | Popularity - IMDB Weighted Rating | ML-32m (vote_count/rating_avg thật) |
| `/api/recommendations/for-you` | Item-based CF (cosine + shrinkage) lai content + popularity | ML-32m + rating của user |
| `/api/recommendations/similar/{id}` | Item-item similarity; fallback genre cosine | ML-32m |

- 20 phim demo được ánh xạ sang `movieId` của ML-32m trong
  `backend/app/recommender/ml_mapping.py`.
- ETL chạy **offline một lần**, sinh artifact nhỏ trong `backend/artifacts/`
  (popularity + ma trận similarity 20×20). API nạp artifact vào RAM lúc khởi động.
- Nếu chưa chạy ETL, app **vẫn chạy** (tự fallback về logic tính từ SQLite).

---

### 1. Khởi tạo Cơ sở dữ liệu SQLite
Mở terminal tại thư mục gốc dự án:
```bash
python database/seed_data.py
```

---

### 2. Precompute artifact từ MovieLens-32m (chạy 1 lần)
Đảm bảo thư mục `ml-32m/` (chứa `ratings.csv`, `movies.csv`) nằm ở gốc dự án, sau đó:
```bash
python -m backend.etl.build_artifacts --backfill-db
```
- Đọc `ml-32m/ratings.csv` theo chunk (~1-2 phút), sinh `backend/artifacts/`.
- `--backfill-db` ghi `vote_count`/`rating_avg` **thật** từ ML-32m vào `database/app.db`
  để UI hiển thị số liệu thật.
- **Thứ tự bắt buộc:** chạy bước 1 trước, rồi bước 2 (vì `seed_data.py` reset bảng movies).

---

### 3. Khởi chạy Backend (FastAPI)
Cài đặt thư viện (nếu chưa cài):
```bash
pip install -r backend/requirements.txt
```

Chạy máy chủ backend:
```bash
uvicorn backend.app.main:app --reload --port 8000
```
- **Backend API**: http://127.0.0.1:8000

---

### 4. Khởi chạy Frontend (ReactJS)
Mở một cửa sổ terminal mới:
```bash
cd frontend
npm install
npm run dev
```
- **Giao diện Web**: http://localhost:5173

---

### 5. Kiểm chứng tầng gợi ý (tùy chọn)
```bash
python scripts/check_recommenders.py
```
Kiểm tra artifact, popularity, tính cá nhân hóa, cold-start, determinism và fallback.

---

### 6. Tài khoản Demo có sẵn
| Tài khoản (Username) | Mật khẩu | Đặc điểm sở thích | Mục đích Demo |
| :--- | :--- | :--- | :--- |
| `alice` | `123456` | Thích phim Hành Động, Marvel | Demo Gợi ý Cá nhân hóa |
| `bob` | `123456` | Thích phim Hoạt hình, Anime | Demo Gợi ý Cá nhân hóa |
| `dave` | `123456` | Thích phim Khoa học Viễn tưởng | Demo Gợi ý Cá nhân hóa |
| `charlie` | `123456` | Chưa có đánh giá nào (0 ratings) | **Demo Bài toán Cold-Start** |
