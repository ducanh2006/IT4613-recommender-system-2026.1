import React from "react";

export default function MovieCard({ movie, onSelectMovie }) {
  if (!movie) return null;

  return (
    <div className="movie-card" onClick={() => onSelectMovie(movie)}>
      <img src={movie.poster_url} alt={movie.title} className="card-poster" loading="lazy" />
      <div className="card-details">
        <div className="card-title" title={movie.title}>{movie.title}</div>
        <div className="card-meta">
          <span>⭐ {movie.rating_avg}</span>
          <span>{movie.release_year}</span>
        </div>
      </div>
    </div>
  );
}
