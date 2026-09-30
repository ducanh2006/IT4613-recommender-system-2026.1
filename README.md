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
