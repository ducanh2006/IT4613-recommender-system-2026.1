import React, { useState } from "react";
import { api } from "../services/api";

// Tài khoản mẫu trong seed_data.py, dùng để demo nhanh
const DEMO_ACCOUNTS = [
  { username: "alice", password: "123456", label: "Alice (Hành Động)" },
  { username: "bob", password: "123456", label: "Bob (Hoạt Hình)" },
  { username: "dave", password: "123456", label: "Dave (Viễn Tưởng)" },
  { username: "charlie", password: "123456", label: "Charlie (Cold Start)" },
];

export default function LoginPage({ onLoginSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Đổi giữa chế độ đăng nhập và đăng ký
  const switchMode = (loginMode) => {
    setIsLogin(loginMode);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const user = isLogin
        ? await api.login(username.trim(), password)
        : await api.register(username.trim(), password, fullName.trim());
      onLoginSuccess(user);
    } catch (err) {
      setError(err.message || "Không thể xác thực, vui lòng thử lại!");
    } finally {
      setLoading(false);
    }
  };

  // Điền sẵn tài khoản demo, vẫn cần bấm đăng nhập
  const fillDemoAccount = (account) => {
    setIsLogin(true);
    setError("");
    setUsername(account.username);
    setPassword(account.password);
  };

  return (
    <div className="login-page">
      <div className="login-hero">
        <div className="login-brand">
          <span className="brand-title">RECSYS MOVIE</span>
          <span className="brand-tag">IT4613 - Hệ Gợi Ý</span>
        </div>
        <p className="login-tagline">
          Đăng nhập để hệ gợi ý học sở thích của bạn từ những phim đã chấm điểm.
        </p>
        <p className="login-tagline">
          Mỗi tài khoản có lịch sử đánh giá riêng, nên hàng "Gợi ý cho bạn" sẽ khác nhau.
        </p>
      </div>

      <div className="login-card">
        <div className="login-tabs">
          <button
            type="button"
            className={`login-tab ${isLogin ? "active" : ""}`}
            onClick={() => switchMode(true)}
          >
            Đăng nhập
          </button>
          <button
            type="button"
            className={`login-tab ${!isLogin ? "active" : ""}`}
            onClick={() => switchMode(false)}
          >
            Đăng ký
          </button>
        </div>

        {error && <div className="login-error">{error}</div>}

        <form className="login-form" onSubmit={handleSubmit}>
          {!isLogin && (
            <label className="login-field">
              <span>Họ tên</span>
              <input
                type="text"
                className="search-input"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
                required
              />
            </label>
          )}

          <label className="login-field">
            <span>Tên đăng nhập</span>
            <input
              type="text"
              className="search-input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </label>

          <label className="login-field">
            <span>Mật khẩu</span>
            <input
              type="password"
              className="search-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={isLogin ? "current-password" : "new-password"}
              required
            />
          </label>

          <button type="submit" className="btn-auth login-submit" disabled={loading}>
            {loading ? "Đang xử lý..." : isLogin ? "Đăng Nhập" : "Tạo Tài Khoản"}
          </button>
        </form>

        <div className="demo-hint">
          <div className="demo-hint-title">Tài khoản demo (mật khẩu 123456)</div>
          <div className="demo-hint-list">
            {DEMO_ACCOUNTS.map((account) => (
              <button
                key={account.username}
                type="button"
                className="demo-chip"
                onClick={() => fillDemoAccount(account)}
              >
                {account.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
