import React from "react";
import MovieCard from "./MovieCard";

export default function MovieRow({ title, movies, onSelectMovie }) {
  if (!movies || movies.length === 0) return null;

  return (
    <div className="movie-section">
      <div className="section-header">
        <h2 className="section-title">{title}</h2>
      </div>
      <div className="movie-grid">
        {movies.map((movie) => (
          <MovieCard
            key={movie.id}
            movie={movie}
            onSelectMovie={onSelectMovie}
          />
        ))}
      </div>
    </div>
  );
}
