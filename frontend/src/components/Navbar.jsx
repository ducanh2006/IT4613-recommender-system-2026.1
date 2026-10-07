import React from "react";

export default function Navbar({
  currentUser,
  onLogout,
  onOpenLogin,
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

        {/* Thông tin tài khoản hoặc Khách */}
        {currentUser ? (
          <>
            <div className="user-chip" title={currentUser.full_name || currentUser.username}>
              <img
                className="user-avatar"
                src={currentUser.avatar_url || "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100&h=100&fit=crop&crop=faces"}
                alt={currentUser.username}
              />
              <span className="user-name">{currentUser.full_name || currentUser.username}</span>
            </div>

            <button className="btn-auth btn-logout" onClick={onLogout}>
              Đăng Xuất
            </button>
          </>
        ) : (
          <>
            <div className="guest-badge" title="Bạn đang duyệt với tư cách Khách vãng lai">
              <span className="guest-dot"></span>
              <span>Khách vãng lai</span>
            </div>
            <button className="btn-auth" onClick={onOpenLogin}>
              Đăng Nhập
            </button>
          </>
        )}
      </div>
    </header>
  );
}
