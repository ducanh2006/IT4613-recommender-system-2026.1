# Development Guidelines

## Project Purpose

This IT4613 movie recommender project is a Netflix-style educational demo showing:

- A FastAPI backend with SQLite persistence.
- A React 19 frontend powered by Vite.
- Popularity-based, content-based, and heuristic collaborative recommendations.
- Demo authentication, quick user switching, movie search, trailers, ratings, and watchlists.
- Cold-start handling: users without ratings receive popularity-based recommendations.

This is a classroom and demonstration system, not a production-ready deployment.

## Main Structure

- `backend/app/main.py`: creates the FastAPI app, configures CORS, and registers routers.
- `backend/app/api/`: HTTP routes for auth, users, movies, interactions, and recommendations.
- `backend/app/database.py`: manages SQLite connections to `database/app.db`.
- `backend/app/models/db_models.py`: Pydantic request and response models.
- `backend/app/recommender/`: recommender interface and algorithm implementations.
- `database/database.sql`: SQLite schema.
- `database/seed_data.py`: creates the schema and loads demo data.
- `frontend/src/App.jsx`: owns the main application state and flows.
- `frontend/src/components/`: Navbar, movie cards/rows, authentication, and video modal components.
- `frontend/src/services/api.js`: the central frontend HTTP client.
- `frontend/src/index.css`: global application styling.

## Running the Project

Run these commands from the repository root unless `cd frontend` is shown.

```bash
python database/seed_data.py
python -m pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Default URLs:

- Frontend: `http://localhost:5173`
- Backend: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

Frontend scripts:

```bash
npm run dev
npm run build
npm run lint
npm run preview
```

There are no backend tests, linting, or type-check scripts, and no frontend test script. After frontend changes, at minimum run `npm run lint` and `npm run build`. For backend changes, start the API and check affected endpoints through Swagger or manual requests.

## Data Flow and API

All routers use the `/api` prefix:

- Auth: `POST /api/auth/register`, `POST /api/auth/login`.
- Users: `GET /api/users`, `GET /api/users/{user_id}`.
- Movies: `GET /api/movies/banner`, `/genres`, `/by-genre/{genre_id}`, `/search`, `/{movie_id}`.
- Recommendations: `/trending`, `/for-you`, `/similar/{movie_id}`.
- Interactions: `/rate`, `/ratings`, `/watchlist/toggle`, `/watchlist`.

The frontend calls the API through `frontend/src/services/api.js`. The API base URL is hardcoded to `http://127.0.0.1:8000`; update it when changing the backend port or deployment environment, and consider adding Vite environment configuration.

`App.jsx` manages the current user, the `recsys_user` localStorage key, movie rows, debounced search, modals, and recommendation refreshes after ratings. Components should receive data and callbacks through props; keep application-level state out of small cards.

## Database

The schema contains `users`, `genres`, `movies`, `movie_genres`, `ratings`, and `watchlist`. Movie genres use a many-to-many relationship; ratings and watchlists are unique per user/movie pair.

When adding or changing a data field:

1. Update `database/database.sql`.
2. Update the matching inserts in `database/seed_data.py`.
3. Update the related Pydantic models, queries, and response formatting.
4. Update frontend API assumptions when the response contract changes.
5. Reseed the demo database and check affected endpoints.

`seed_data.py` uses `CREATE IF NOT EXISTS`, `INSERT OR IGNORE`, and `INSERT OR REPLACE`; rerunning it is not a complete migration or reset. Existing ratings and watchlist entries may remain. Do not commit `database/app.db`; it is ignored.

All user-controlled values must use SQL parameter binding (`?`), never string interpolation. Preserve relationship integrity when adding foreign keys or delete operations.

## Recommendation Algorithms

- `PopularityRecommender`: IMDb-style weighted rating using the global average and `m=5`; used for trending and cold-start fallback.
- `ContentBasedRecommender`: one-hot genre vectors with cosine similarity; excludes the target movie and sorts by similarity, then rating.
- `CollaborativeRecommender`: heuristic based on high ratings, favorite genres, and ratings from other users; excludes rated movies and falls back to popularity when needed.
- `BaseRecommender`: preserve `recommend(conn, user_id=None, movie_id=None, top_k=10)` when adding an algorithm.

When changing an algorithm, document its inputs, formula/weights, cold-start behavior, rated-movie filtering, and fallback. Check users with many ratings, only low ratings, no ratings, and invalid movie IDs. Endpoint `top_k` is currently limited to 1 through 50.

## Code Conventions

- Keep backend modules within their current boundaries; routers handle HTTP/query orchestration, while recommendation logic stays in `recommender/`.
- Use functional React components and hooks, following the existing style.
- Use Pydantic models for backend payloads and responses when an endpoint has a contract.
- Keep movie responses consistent with `_format_movie()` and `MovieResponse`.
- User-facing UI text and code comments must use Vietnamese where the existing code does.
- Write code comments in Vietnamese, keep each comment brief at approximately 5-15 words, and avoid comments that merely restate the code.
- Keep Python docstrings short and focused; `"""..."""` blocks must not become long explanations.
- Avoid dependencies when the current solution is sufficient. If one is needed, update `backend/requirements.txt` or `frontend/package.json` and its lockfile.
- Avoid formatting unrelated files. `frontend/package-lock.json` may contain pre-existing changes; do not revert them.

## Demo Limitations and Safety

- Passwords are currently stored and compared as plain text; there are no tokens, sessions, or authorization. Do not treat this as real authentication or deploy publicly before adding password hashing, identity validation, and ownership checks.
- Clients can submit arbitrary `user_id` values; rating and watchlist ownership is not validated.
- CORS currently allows every origin with credentials; restrict it before production.
- Posters, backdrops, trailers, and avatars come from TMDB, YouTube, and Unsplash; the UI should handle failed URLs and offline conditions.
- Ratings update `rating_avg` and add a `+120` vote-count offset for demo data only.
- The collaborative recommender runs an additional query for each candidate; optimize queries or batch data before scaling.
- Content-based recommendations use genres only; they are not learned from full movie content.

## Task Workflow

1. Identify the owning module and read the nearest call sites and contracts.
2. For API or schema changes, update backend, seed/schema, and frontend service together.
3. Keep changes focused and preserve input validation and existing fallbacks.
4. Run suitable validation: `npm run lint`, `npm run build`, database checks, or Swagger endpoint checks.
5. When recommendation behavior changes, test both cold-start and rated users.
6. Never commit or reset existing user changes unless explicitly requested.
