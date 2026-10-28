# Hệ Gợi Ý Phim (Movie Recommender System) — IT4613

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
git clone https://github.com/ducanh2006/IT4613-recommender-system-2026.1.git
cd IT4613-recommender-system-2026.1
```

Cài đặt thư viện Python cho toàn bộ dự án:
```bash
pip install -r requirements.txt
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

### 3.2. Bốn Interface / Contract quan trọng

Hệ thống phân định ranh giới rõ ràng thông qua 4 Hợp đồng (Contracts) giúp việc mở rộng thuật toán độc lập hoàn toàn với cơ sở dữ liệu:

| STT | Tên Hợp Đồng | File nguồn & Lớp chính | Vai trò & Quy định bắt buộc |
| :---: | :--- | :--- | :--- |
| **1** | **Data Contract** | [data.py](backend/app/recommender/contracts/data.py#L132-L165)<br>`Dataset`, `IdMap` | **Chuẩn hóa dữ liệu đầu vào:** Mọi nguồn dữ liệu (CSV MovieLens hay SQLite) đều phải đóng gói thành `Dataset` (gồm 4 cột `[user_id, item_id, rating, timestamp]`, bộ ánh xạ ID 2 chiều `IdMap`). Thuật toán không được tự ý đọc file hay query DB. |
| **2** | **Behavior Contract** | [behavior.py](backend/app/recommender/contracts/behavior.py#L26-L53)<br>`Recommender`, `RecommenderCapabilities` | **Cam kết năng lực thuật toán:** Bắt buộc có đủ 3 hàm: `fit(data)`, `recommend(request)`, `predict(user, item)` và thuộc tính `capabilities` để runner tự động biết cách kiểm thử. |
| **3** | **Request & Result** | [recommendation.py](backend/app/recommender/contracts/recommendation.py#L15-L65)<br>`RecommendationRequest`, `RecommendationResult` | **Giao thức hỏi - đáp dự đoán:** Chuẩn hóa đầu vào (`mode`, `user_id`, `top_k`, `exclude_item_ids`) và đầu ra (`RecommendationResult`). Object tự động validate: điểm giảm dần, rank liên tục từ 1..K, không vượt quá `top_k`. |
| **4** | **Online Contract** | [base.py](backend/app/recommender/base.py#L5-L11)<br>`BaseRecommender` | **Giao tiếp phục vụ Web:** Nhận kết nối `sqlite3.Connection`, trả về `List[Dict]` gồm đầy đủ thông tin phim (`id`, `title`, `poster_url`, `genres`) để FastAPI trả JSON cho React. |

### 3.3. Hướng dẫn mở rộng & Thêm thuật toán mới (Hỗ trợ cả AI và Non-AI)

Cơ chế thẻ căn cước **`capabilities: RecommenderCapabilities`** cho phép hệ thống vận hành linh hoạt với mọi dạng thuật toán:

| Loại thuật toán | Ví dụ điển hình | Khai báo `capabilities` | Quy tắc cài đặt trong code |
| :--- | :--- | :--- | :--- |
| **Không cần Train (Non-AI)** | Random, Most Popular, Global Mean | `requires_training=False`<br>`supports_rating_prediction=False` | • `fit()`: Chỉ cần lưu lại danh sách ID phim hoặc đếm tần suất đơn giản, return `self`.<br>• `predict()`: Ném lỗi `UnsupportedOperation`. Runner sẽ tự động bỏ qua đo RMSE/MAE. |
| **Cần Train / Học máy (AI / ML)** | FunkSVD, NCF, Item-KNN, User-KNN | `requires_training=True`<br>`supports_rating_prediction=True` | • `fit()`: Tối ưu ma trận / gradient descent qua các epoch.<br>• `predict()`: Tính điểm rating dự đoán $\hat{r}_{u,i}$. Runner sẽ đo đầy đủ RMSE, MAE. |

**Quy trình 3 bước thêm thuật toán mới:**
1. Tạo file trong `backend/app/recommender/models/my_model.py` kế thừa [BaseRecommender](backend/app/recommender/models/_base.py#L180).
2. Đăng ký model trong `evaluation/runner.py` để tham gia Benchmark so sánh tự động.
3. (Nếu muốn đưa lên Web) Viết 1 class Adapter kế thừa [recommender/base.py](backend/app/recommender/base.py) để ánh xạ ID phim ra thông tin poster từ SQLite.

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
| **Hàng 1: Phim thịnh hành**<br>*(Trending Movies)* | **IMDB Weighted Rating** ([popularity.py](backend/app/recommender/popularity.py)) | $WR = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$<br>($v$: số lượt vote, $m$: ngưỡng tối thiểu, $R$: điểm TB phim, $C$: điểm TB toàn sàn) | Gợi ý các bộ phim chất lượng được đông đảo khán giả đón nhận; không bị thiên lệch bởi phim chỉ có 1-2 đánh giá 5 sao. |
| **Hàng 2: Gợi ý cho bạn**<br>*(Personalized Movies)* | **Collaborative Filtering** ([collaborative.py](backend/app/recommender/collaborative.py)) | Phân tích lịch sử đánh giá của user; tìm các phim tương tự với phim user từng thích điểm cao ($\ge 4.0$). | Cá nhân hóa theo gu riêng của từng tài khoản (`alice`: Marvel, `bob`: Anime...). |
| **Hàng 3: Phim tương tự**<br>*(Similar Movies)* | **Content-Based Filtering** ([content_based.py](backend/app/recommender/content_based.py)) | Biểu diễn thể loại phim thành One-hot vector; tính khoảng cách góc **Cosine Similarity** giữa các phim. | Hiển thị các phim tương tự khi người dùng click vào xem chi tiết một bộ phim cụ thể. |
| **Xử lý Người dùng mới**<br>*(Cold-Start Handler)* | **Fallback Mechanism** | Nếu `user_id` chưa từng đánh giá phim nào (như tài khoản `charlie`): tự động chuyển hướng sang danh sách Trending. | Tránh lỗi crash hệ thống hoặc trang trắng; giúp người dùng mới luôn có phim để trải nghiệm. |

### 4.2. Bảng thuật toán trong Khung đánh giá (Track Offline Benchmark)

| Thuật toán | Phân loại | Đặc điểm & Ưu thế kỹ thuật |
| :--- | :--- | :--- |
| **FunkSVD** | AI / Matrix Factorization | Phân rã ma trận thưa thành các nhân tử ẩn (latent factors) bằng SGD, tối ưu dự đoán điểm rating (RMSE/MAE thấp). |
| **Item-KNN** | Collaborative Filtering | Tính độ tương đồng giữa các cặp phim dựa trên hành vi chấm điểm của toàn bộ người dùng. |
| **User-KNN** | Collaborative Filtering | Tìm nhóm người dùng có sở thích tương đồng (k-láng giềng) để đề xuất phim chéo cho nhau. |
| **Hybrid** | Mô hình Lai | Kết hợp có trọng số điểm số từ Content-Based và Collaborative Filtering nhằm tận dụng ưu điểm của cả hai. |
| **Most Popular / Random** | Baseline Đối chứng | Làm mốc chuẩn tối thiểu để so sánh và chứng minh tính hiệu quả của các mô hình AI/ML. |
