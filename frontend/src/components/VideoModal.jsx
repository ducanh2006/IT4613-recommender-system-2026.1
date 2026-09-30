import React, { useState, useEffect } from "react";
import { api } from "../services/api";

export default function VideoModal({ movie, currentUser, onClose, onRateSuccess, onSelectMovie }) {
  const [userRating, setUserRating] = useState(movie?.user_rating || 0);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!movie) return;
    setUserRating(movie.user_rating || 0);
    setMessage("");

    // Lấy phim tương tự
    api.getSimilar(movie.id, 5)
      .then((data) => setSimilarMovies(data))
      .catch((err) => console.error("Lỗi:", err));
  }, [movie]);

  if (!movie) return null;

  const handleRate = async (star) => {
    if (!currentUser) {
      setMessage("Vui lòng chọn hoặc đăng nhập người dùng để chấm sao!");
      return;
    }
    try {
      await api.rateMovie(currentUser.id, movie.id, star);
      setUserRating(star);
      setMessage(`Đã lưu ${star} sao!`);
      if (onRateSuccess) onRateSuccess(movie.id, star);
    } catch {
      setMessage("Lỗi lưu đánh giá!");
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 style={{ fontSize: "1.2rem", fontWeight: "bold" }}>{movie.title}</h2>
          <button className="close-btn" onClick={onClose}>✕ Đóng</button>
        </div>

        {/* Video Trailer YouTube */}
        <iframe
          className="video-frame"
          src={`https://www.youtube.com/embed/${movie.trailer_youtube_id}?autoplay=1&rel=0`}
          title={movie.title}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>

        {/* Thông tin phim */}
        <div style={{ fontSize: "0.85rem", color: "#aaa", marginBottom: "8px" }}>
          Năm: {movie.release_year} • Thời lượng: {movie.duration_minutes} phút • Thể loại: {movie.genres?.join(", ")}
        </div>
        <p style={{ fontSize: "0.9rem", color: "#ddd", marginBottom: "16px" }}>{movie.overview}</p>

        {/* Bảng chấm điểm sao */}
        <div className="rating-bar">
          <div>
            <div style={{ fontSize: "0.85rem", fontWeight: "600" }}>
              {currentUser ? `Đánh giá của bạn (${currentUser.username}):` : "Đăng nhập để đánh giá:"}
            </div>
            {message && <div style={{ fontSize: "0.8rem", color: "#00e676" }}>{message}</div>}
          </div>
          <div>
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                className={`btn-star ${userRating >= star ? "active" : ""}`}
                onClick={() => handleRate(star)}
              >
                ★
              </button>
            ))}
          </div>
        </div>

        {/* Phim tương tự */}
        {similarMovies.length > 0 && (
          <div style={{ marginTop: "16px" }}>
            <div style={{ fontSize: "0.9rem", fontWeight: "700", color: "#fff", marginBottom: "8px" }}>
              Phim tương tự
            </div>
            <div className="similar-list">
              {similarMovies.map((sim) => (
                <div key={sim.id} className="similar-item" onClick={() => onSelectMovie(sim)}>
                  <img src={sim.poster_url} alt={sim.title} />
                  <div className="sim-title">{sim.title}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
