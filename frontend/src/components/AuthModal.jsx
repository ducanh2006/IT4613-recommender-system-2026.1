import React, { useState } from "react";
import { api } from "../services/api";

export default function AuthModal({ onClose, onAuthSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      let user;
      if (isLogin) {
        user = await api.login(username, password);
      } else {
        user = await api.register(username, password, fullName);
      }
      onAuthSuccess(user);
      onClose();
    } catch (err) {
      setError(err.message || "Lỗi thao tác!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" style={{ maxWidth: "380px" }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 style={{ fontSize: "1.1rem" }}>{isLogin ? "Đăng Nhập" : "Đăng Ký Tài Khoản Mới"}</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        {error && (
          <div style={{ color: "#ff5252", fontSize: "0.8rem", marginBottom: "12px" }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {!isLogin && (
            <div>
              <label style={{ fontSize: "0.75rem", color: "#aaa" }}>Họ tên</label>
              <input
                type="text"
                className="search-input"
                style={{ width: "100%", marginTop: "4px" }}
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>
          )}

          <div>
            <label style={{ fontSize: "0.75rem", color: "#aaa" }}>Tên đăng nhập (Username)</label>
            <input
              type="text"
              className="search-input"
              style={{ width: "100%", marginTop: "4px" }}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div>
            <label style={{ fontSize: "0.75rem", color: "#aaa" }}>Mật khẩu (Plain text)</label>
            <input
              type="password"
              className="search-input"
              style={{ width: "100%", marginTop: "4px" }}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="btn-auth" style={{ marginTop: "10px", padding: "10px" }} disabled={loading}>
            {loading ? "Đang xử lý..." : isLogin ? "Đăng Nhập" : "Tạo Tài Khoản (Test Cold-Start)"}
          </button>
        </form>

        <div style={{ textAlign: "center", marginTop: "14px", fontSize: "0.8rem", color: "#888" }}>
          {isLogin ? (
            <span style={{ cursor: "pointer", color: "#38bdf8" }} onClick={() => setIsLogin(false)}>
              Chưa có tài khoản? Đăng ký tại đây
            </span>
          ) : (
            <span style={{ cursor: "pointer", color: "#38bdf8" }} onClick={() => setIsLogin(true)}>
              Đã có tài khoản? Đăng nhập tại đây
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
