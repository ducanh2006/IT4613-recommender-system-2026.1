# Hướng Dẫn Chạy Dự Án - Hệ Gợi Ý Phim (IT4613)

### 1. Khởi tạo Cơ sở dữ liệu SQLite
Mở terminal tại thư mục gốc dự án:
```bash
python database/seed_data.py
```

---

### 2. Khởi chạy Backend (FastAPI)
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

### 3. Khởi chạy Frontend (ReactJS)
Mở một cửa sổ terminal mới:
```bash
cd frontend
npm install
npm run dev
```
- **Giao diện Web**: http://localhost:5173

---

### 4. Tài khoản Demo có sẵn
| Tài khoản (Username) | Mật khẩu | Đặc điểm sở thích | Mục đích Demo |
| :--- | :--- | :--- | :--- |
| `alice` | `123456` | Thích phim Hành Động, Marvel | Demo Gợi ý Cá nhân hóa |
| `bob` | `123456` | Thích phim Hoạt hình, Anime | Demo Gợi ý Cá nhân hóa |
| `dave` | `123456` | Thích phim Khoa học Viễn tưởng | Demo Gợi ý Cá nhân hóa |
| `charlie` | `123456` | Chưa có đánh giá nào (0 ratings) | **Demo Bài toán Cold-Start** |

---

### 5. Offline Benchmark (đánh giá mô hình)

Cài thêm thư viện cho môi trường dev:
```bash
pip install -r requirements-dev.txt
```

Chạy benchmark (chạy từ thư mục gốc dự án):
```bash
python -m evaluation.runner --help
python -m evaluation.runner --data-dir data/ml-100k --out results/latest --protocol temporal
```

> Có thể gọi trực tiếp `python evaluation/runner.py ...` — entry point tự thêm thư mục gốc vào `sys.path`.

Các tham số chính:

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--protocol` | `temporal` | `temporal` (chia theo thời gian trên từng user) hoặc `leave_one_out` |
| `--models` | `all` | Danh sách model cách nhau bằng dấu phẩy, ví dụ `funk_svd,hybrid` |
| `--k` | `10` | Số phim gợi ý (K) |
| `--cold-user-ratio` | `0.1` | Tỉ lệ user bị giữ lại khỏi tập train để mô phỏng cold-start |
| `--seed` | `42` | Random seed cho toàn bộ pipeline |

Kết quả được ghi vào `--out` (mặc định `results/latest`):

| File | Nội dung |
| :--- | :--- |
| `report.md` | Báo cáo Markdown (bảng ranking, rating, so sánh baseline, cold-start, hiệu năng) |
| `metrics_summary.csv` | Metric theo model/split/task kèm `scope` (`user_macro` hoặc `global`) |
| `global_metrics.csv` | Metric gộp toàn bộ tập (ví dụ RMSE/MAE pooled) |
| `per_user_metrics.csv` | Metric chi tiết từng user |
| `recommendations_sample.csv` | Danh sách gợi ý mẫu |
| `comparisons.csv` | So sánh với baseline (bootstrap CI, Wilcoxon, Holm–Bonferroni) |
| `efficiency.csv` | Thời gian fit và độ trễ `recommend` |
| `run_metadata.json` | Metadata của lần chạy (seed, protocol, phiên bản thư viện, commit) |

---

### 6. Chạy unit test
```bash
pytest
```
