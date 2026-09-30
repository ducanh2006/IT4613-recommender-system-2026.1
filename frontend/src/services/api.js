const API_BASE = "http://127.0.0.1:8000";

async function request(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Lỗi HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

export const api = {
  // Movies & Metadata
  getBannerMovie: () => request("/api/movies/banner"),
  getGenres: () => request("/api/movies/genres"),
  getMoviesByGenre: (genreId, userId) =>
    request(`/api/movies/by-genre/${genreId}${userId ? `?user_id=${userId}` : ""}`),
  getMovieDetail: (movieId, userId) =>
    request(`/api/movies/${movieId}${userId ? `?user_id=${userId}` : ""}`),
  searchMovies: (query, userId) =>
    request(`/api/movies/search?q=${encodeURIComponent(query)}${userId ? `&user_id=${userId}` : ""}`),

  // Recommendations (IT4613 Core)
  getTrending: (topK = 10) => request(`/api/recommendations/trending?top_k=${topK}`),
  getPersonalized: (userId, topK = 10) =>
    request(`/api/recommendations/for-you?${userId ? `user_id=${userId}&` : ""}top_k=${topK}`),
  getSimilar: (movieId, topK = 10) => request(`/api/recommendations/similar/${movieId}?top_k=${topK}`),

  // Interactions (Rating & Watchlist)
  rateMovie: (userId, movieId, rating) =>
    request("/api/interactions/rate", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, movie_id: movieId, rating }),
    }),
  getRatings: (userId) => request(`/api/interactions/ratings?user_id=${userId}`),
  toggleWatchlist: (userId, movieId) =>
    request(`/api/interactions/watchlist/toggle?user_id=${userId}&movie_id=${movieId}`, {
      method: "POST",
    }),
  getWatchlist: (userId) => request(`/api/interactions/watchlist?user_id=${userId}`),

  // Auth & Users
  getUsers: () => request("/api/users"),
  login: (username, password) =>
    request("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  register: (username, password, fullName) =>
    request("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password, full_name: fullName }),
    }),
};
