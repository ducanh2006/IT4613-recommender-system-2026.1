# Hệ Gợi Ý Phim (Movie Recommender System) — IT4613

Dự án nghiên cứu và triển khai Hệ thống Gợi ý Phim theo kiến trúc **Dual-track** (Song song 2 luồng):
- **Track Online (Ứng dụng Web):** Web app hoàn chỉnh (FastAPI + ReactJS + SQLite) phục vụ người dùng thực tế với 3 luồng gợi ý trực quan và xử lý mượt mà bài toán Cold-Start.
- **Track Offline (Đánh giá Khoa học):** Khung benchmark chuẩn mực trên tập MovieLens-100k, đo đạc toàn diện các chỉ số Xếp hạng (Ranking) và Dự đoán điểm (Rating) kèm kiểm định thống kê.

---

## 1. Giới thiệu về dự án

### Bảng công nghệ chính

| Thành phần | Công nghệ | Vai trò trong hệ thống |
| :--- | :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, SQLite | Cung cấp RESTful API, quản lý dữ liệu người dùng & phim |
| **Frontend** | ReactJS, Vite, Tailwind/CSS | Giao diện Web tương tác trực quan (xem phim, đánh giá, nhận gợi ý) |
| **Core ML** | NumPy, Pandas, Scikit-learn, SciPy | Xử lý ma trận dữ liệu và các thuật toán gợi ý |
| **Benchmark** | Pytest, Custom Evaluation Runner | Đo đạc chỉ số (NDCG, Precision, RMSE...), kiểm định Wilcoxon, Bootstrap |

---

## 2. Cách chạy và cài đặt dự án

### 2.1. Clone dự án & Cài đặt môi trường
Mở terminal và clone mã nguồn về máy:
```bash
git clone https://github.com/ducanh2006/IT4613-recommender-system-2026.1.git
cd IT4613-recommender-system-2026.1
```

Cài đặt thư viện Python cơ bản (bắt buộc trước khi khởi tạo database):
```bash
pip install -r backend/requirements.txt
```

---

### 2.2. Khởi chạy Ứng dụng Web (Track Online)

| Bước | Thao tác | Lệnh thực thi (Terminal) | Ghi chú |
| :--- | :--- | :--- | :--- |
| **1. Khởi tạo DB** | Nạp dữ liệu mẫu SQLite | `python database/seed_data.py` | Tạo `app.db` và các tài khoản demo (cần `bcrypt`) |
| **2. Chạy Backend** | Chạy máy chủ FastAPI | `uvicorn backend.app.main:app --reload --port 8000` | API tài liệu: `http://127.0.0.1:8000/docs` |
| **3. Chạy Frontend** | Cài thư viện & chạy Web | `cd frontend`<br>`npm install`<br>`npm run dev` | Giao diện Web: `http://localhost:5173` |

#### Tài khoản Demo có sẵn trên Web:

| Tài khoản | Mật khẩu | Đặc điểm sở thích | Mục đích Demo trên Web |
| :--- | :--- | :--- | :--- |
| `alice` | `123456` | Thích phim Hành Động, Marvel | Demo Gợi ý Cá nhân hóa (Action/Sci-Fi) |
| `bob` | `123456` | Thích phim Hoạt hình, Anime | Demo Gợi ý Cá nhân hóa (Animation) |
| `dave` | `123456` | Thích phim Khoa học Viễn tưởng | Demo Gợi ý Cá nhân hóa (Sci-Fi) |
| `charlie` | `123456` | Chưa có đánh giá nào (0 ratings) | **Demo Xử lý Người dùng mới (Cold-Start)** |

---

### 2.3. Đánh giá Mô hình & Benchmark (Track Offline)

Vì dữ liệu lớn không commit lên Git, để chạy Benchmark và Unit Test bạn cần cài đặt thư viện dev và chuẩn bị tập MovieLens-100k:

1. **Cài đặt thư viện phát triển (Dev dependencies):**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Tải tập dữ liệu MovieLens-100k vào `data/ml-100k`:**
   * Tải trực tiếp file zip chính thức từ [GroupLens ml-100k.zip](https://files.grouplens.org/datasets/movielens/ml-100k.zip) và giải nén vào thư mục `data/ml-100k`.
   * *Hoặc chạy nhanh 1 dòng lệnh sau trong terminal để tự động tải & giải nén:*
     ```bash
     python -c "import urllib.request, zipfile, os; os.makedirs('data', exist_ok=True); urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-100k.zip', 'data/ml-100k.zip'); zipfile.ZipFile('data/ml-100k.zip').extractall('data/')"
     ```

3. **Chạy Kiểm thử & Benchmark:**

   | Thao tác | Lệnh thực thi (Terminal) | Mục đích |
   | :--- | :--- | :--- |
   | **Kiểm thử tự động** | `pytest` | Chạy toàn bộ test suite kiểm tra contracts & thuật toán |
   | **Chạy Benchmark** | `python -m evaluation.runner --data-dir data/ml-100k --out results/latest` | Chạy đánh giá toàn diện các model trên MovieLens |

#### Tham số chính của bộ Benchmark (`evaluation.runner`):

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--protocol` | `temporal` | Giao thức chia tập: `temporal` (theo thời gian) hoặc `leave_one_out` |
| `--models` | `all` | Danh sách model cần chạy (ví dụ: `funk_svd,item_knn,hybrid`) |
| `--k` | `10` | Số lượng phim gợi ý trong Top-K |
| `--cold-user-ratio` | `0.1` | Tỉ lệ người dùng giữ lại để mô phỏng đánh giá Cold-Start |
| `--seed` | `42` | Random seed đảm bảo kết quả tất định (deterministic) |

---

## 3. Kiến trúc dự án

### 3.1. Cấu trúc thư mục (File Tree)

```text
IT4613-recommender-system-2026.1/
├── backend/app/
│   ├── api/                   # Router FastAPI (các endpoint cho Web UI gọi)
│   ├── database/              # Khởi tạo kết nối SQLite (app.db)
│   └── recommender/           # LÕI HỆ THỐNG GỢI Ý
│       ├── base.py            # [Online Contract] Base class cho Web UI
│       ├── popularity.py      # [Online] Xếp hạng thịnh hành (IMDB Weighted Rating)
│       ├── collaborative.py   # [Online] Lọc cộng tác cá nhân hóa
│       ├── content_based.py   # [Online] Lọc theo đặc trưng thể loại phim
│       ├── contracts/         # [Offline Contracts] Hợp đồng dữ liệu & hành vi
│       │   ├── data.py        # Contract Dataset, IdMap, ItemFeatures
│       │   ├── behavior.py    # Contract Recommender, Capabilities
│       │   └── recommendation.py # Contract Request & Result
│       └── models/            # [Offline Models] Các thuật toán benchmark
│           ├── _base.py       # Lớp cha BaseRecommender & toolkit dùng chung
│           ├── funk_svd.py    # Phân rã ma trận FunkSVD (AI / Matrix Factorization)
│           ├── item_knn.py    # Item-based Collaborative Filtering
│           ├── user_knn.py    # User-based Collaborative Filtering
│           ├── hybrid.py      # Mô hình lai kết hợp Content + Collaborative
│           └── random.py      # Thuật toán ngẫu nhiên (Baseline non-AI)
├── frontend/                  # Mã nguồn giao diện Web ReactJS
├── evaluation/                # Bộ công cụ Benchmark (runner, metrics, splits, stats)
├── database/                  # Schema SQL (database.sql) và script nạp dữ liệu (seed_data.py)
├── tests/                     # Unit test tự động kiểm tra contracts & thuật toán
└── data/                      # Dữ liệu MovieLens thô (được cấu hình .gitignore)
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

## 4. Giải thích thuật toán đã cài đặt và triển khai thực tế

### 4.1. Bảng tổng hợp thuật toán trên Giao diện Web (Track Online)

| Hàng hiển thị trên Web | Thuật toán cốt lõi | Công thức / Nguyên lý | Mục đích & Trải nghiệm người dùng |
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
