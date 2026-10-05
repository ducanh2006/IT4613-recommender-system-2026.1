import React from "react";

export default function Navbar({
  currentUser,
  onLogout,
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

        {/* Thông tin tài khoản đang đăng nhập */}
        <div className="user-chip" title={currentUser.full_name || currentUser.username}>
          <img
            className="user-avatar"
            src={currentUser.avatar_url}
            alt={currentUser.username}
          />
          <span className="user-name">{currentUser.full_name || currentUser.username}</span>
        </div>

        <button className="btn-auth btn-logout" onClick={onLogout}>
          Đăng Xuất
        </button>
      </div>
    </header>
  );
}
