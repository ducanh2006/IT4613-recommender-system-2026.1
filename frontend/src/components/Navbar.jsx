import React from "react";

export default function Navbar({
  currentUser,
  usersList,
  onSelectUser,
  onOpenAuth,
  searchQuery,
  onSearch,
}) {
  return (
    <header className="navbar">
      <div className="nav-brand">
        <span className="brand-title">RECSYS MOVIE</span>
        <span className="course-tag">IT4613 - Hệ Gợi Ý</span>
      </div>

      <div className="nav-controls">
        {/* Tìm kiếm phim */}
        <input
          type="text"
          className="search-input"
          placeholder="Tìm phim, thể loại..."
          value={searchQuery}
          onChange={(e) => onSearch(e.target.value)}
        />

        {/* Chuyển nhanh tài khoản trực tiếp ngay trên Navbar */}
        <select
          className="user-select"
          value={currentUser ? currentUser.id : ""}
          onChange={(e) => {
            const val = e.target.value;
            if (val === "") {
              onSelectUser(null);
            } else {
              const u = usersList.find((item) => item.id === parseInt(val));
              if (u) onSelectUser(u);
            }
          }}
          title="Chọn người dùng để kiểm thử gợi ý"
        >
          <option value="">👤 Khách (Chưa đăng nhập)</option>
          {usersList.map((u) => (
            <option key={u.id} value={u.id}>
              👤 {u.full_name || u.username}
            </option>
          ))}
        </select>

        {/* Nút Đăng nhập / Đăng ký thủ công */}
        <button className="btn-auth" onClick={onOpenAuth}>
          {currentUser ? "Tài Khoản" : "Đăng Nhập"}
        </button>
      </div>
    </header>
  );
}
