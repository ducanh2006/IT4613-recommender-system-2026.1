import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import MovieRow from "./components/MovieRow";
import VideoModal from "./components/VideoModal";
import LoginPage from "./components/LoginPage";
import { api } from "./services/api";

export default function App() {
  // Phiên đăng nhập được khôi phục từ localStorage
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      const saved = localStorage.getItem("recsys_user");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  // Modal đăng nhập (hiển thị khi khách bấm Đăng nhập hoặc khi chưa chọn chế độ)
  const [showLoginModal, setShowLoginModal] = useState(false);

  const [trendingMovies, setTrendingMovies] = useState([]);
  const [personalizedMovies, setPersonalizedMovies] = useState([]);
  const [actionMovies, setActionMovies] = useState([]);
  const [animationMovies, setAnimationMovies] = useState([]);
  const [scifiMovies, setScifiMovies] = useState([]);

  const [selectedMovie, setSelectedMovie] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  // Đăng nhập thành công: lưu phiên để lần sau không phải nhập lại
  const handleLoginSuccess = (user) => {
    setCurrentUser(user);
    setShowLoginModal(false);
    localStorage.setItem("recsys_user", JSON.stringify(user));
  };

  // Đăng xuất: chuyển về trạng thái khách vãng lai
  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem("recsys_user");
    setSearchQuery("");
    setSearchResults([]);
    setSelectedMovie(null);
  };

  // Tải các hàng phim
  const loadMovies = async () => {
    try {
      const [trending, action, anim, scifi] = await Promise.all([
        api.getTrending(10),
        api.getMoviesByGenre(1),
        api.getMoviesByGenre(4),
        api.getMoviesByGenre(2),
      ]);
      setTrendingMovies(trending);
      setActionMovies(action);
      setAnimationMovies(anim);
      setScifiMovies(scifi);
    } catch (err) {
      console.error("Lỗi tải phim:", err);
    }
  };

  // Tải riêng hàng gợi ý cá nhân hóa (hoặc fallback Popularity nếu không có user_id)
  const loadPersonalized = async (userId) => {
    try {
      const recs = await api.getPersonalized(userId, 10);
      setPersonalizedMovies(recs);
    } catch (err) {
      console.error("Lỗi tải gợi ý:", err);
    }
  };

  // Luôn tải danh mục phim ngay khi vào ứng dụng (cả khách và user đã login)
  useEffect(() => {
    loadMovies();
  }, []);

  // Tải gợi ý khi user thay đổi (nếu khách vãng lai, truyền undefined để backend fallback Popularity)
  useEffect(() => {
    loadPersonalized(currentUser?.id);
  }, [currentUser]);

  // Tìm kiếm phim (hỗ trợ cả khách vãng lai)
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    const timer = setTimeout(() => {
      api.searchMovies(searchQuery, currentUser?.id)
        .then((res) => setSearchResults(res))
        .catch((err) => console.error(err));
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery, currentUser]);

  // Xử lý sau khi người dùng chấm sao
  const handleRateSuccess = () => {
    loadPersonalized(currentUser?.id);
  };

  return (
    <div>
      {/* 1. Thanh điều hướng */}
      <Navbar
        currentUser={currentUser}
        onLogout={handleLogout}
        onOpenLogin={() => setShowLoginModal(true)}
        searchQuery={searchQuery}
        onSearch={setSearchQuery}
      />

      {/* 2. Vùng danh sách phim */}
      <div className="main-container">
        {searchQuery.trim() ? (
          <div>
            <h2 style={{ fontSize: "1.1rem", marginBottom: "16px" }}>
              Kết quả tìm kiếm cho: "{searchQuery}" ({searchResults.length} phim)
            </h2>
            <div className="movie-grid">
              {searchResults.map((m) => (
                <div key={m.id} className="movie-card" onClick={() => setSelectedMovie(m)}>
                  <img src={m.poster_url} alt={m.title} className="card-poster" />
                  <div className="card-details">
                    <div className="card-title">{m.title}</div>
                    <div className="card-meta">
                      <span>⭐ {m.rating_avg}</span>
                      <span>{m.release_year}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            {/* HÀNG 1: GỢI Ý (Cá nhân hóa cho User hoặc Gợi ý phổ biến cho Khách) */}
            <MovieRow
              title={
                currentUser
                  ? `✨ Gợi ý dành riêng cho bạn (${currentUser.username})`
                  : "🔥 Gợi ý cho bạn (Phổ biến nhất)"
              }
              movies={personalizedMovies}
              onSelectMovie={(m) => setSelectedMovie(m)}
            />

            {/* HÀNG 2: XU HƯỚNG THỊNH HÀNH */}
            <MovieRow
              title="📈 Phim thịnh hành"
              movies={trendingMovies}
              onSelectMovie={(m) => setSelectedMovie(m)}
            />

            {/* HÀNG 3: HÀNH ĐỘNG */}
            <MovieRow
              title="💥 Phim Hành Động"
              movies={actionMovies}
              onSelectMovie={(m) => setSelectedMovie(m)}
            />

            {/* HÀNG 4: HOẠT HÌNH & ANIME */}
            <MovieRow
              title="🎨 Phim Hoạt Hình & Anime"
              movies={animationMovies}
              onSelectMovie={(m) => setSelectedMovie(m)}
            />

            {/* HÀNG 5: KHOA HỌC VIỄN TƯỞNG */}
            <MovieRow
              title="🚀 Phim Khoa Học Viễn Tưởng"
              movies={scifiMovies}
              onSelectMovie={(m) => setSelectedMovie(m)}
            />
          </>
        )}
      </div>

      {/* 3. Modal phát trailer & xem phim tương tự */}
      {selectedMovie && (
        <VideoModal
          movie={selectedMovie}
          currentUser={currentUser}
          onClose={() => setSelectedMovie(null)}
          onRateSuccess={handleRateSuccess}
          onSelectMovie={(m) => setSelectedMovie(m)}
          onPromptLogin={() => setShowLoginModal(true)}
        />
      )}

      {/* 4. Modal đăng nhập khi khách click Đăng nhập hoặc muốn Chấm sao */}
      {showLoginModal && (
        <div className="login-modal-overlay" onClick={() => setShowLoginModal(false)}>
          <div className="login-modal-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="login-modal-close"
              onClick={() => setShowLoginModal(false)}
              title="Đóng"
            >
              ✕
            </button>
            <LoginPage
              onLoginSuccess={handleLoginSuccess}
              onGuestAccess={() => setShowLoginModal(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
