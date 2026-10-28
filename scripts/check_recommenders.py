"""
Kiểm chứng tầng recommender sau khi sửa nền tảng + tích hợp ML-32m.

Chạy (từ thư mục gốc project):
    python scripts/check_recommenders.py

Kiểm tra: artifact, popularity, item-item similarity, for-you cá nhân hóa,
cold-start fallback, determinism, và fallback khi thiếu artifact.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from backend.app.database import get_db_connection
from backend.app.recommender import ml_mapping
from backend.app.recommender import popularity as popularity_mod
from backend.app.recommender import content_based as content_mod
from backend.app.recommender import collaborative as collab_mod
from backend.app.recommender.popularity import PopularityRecommender
from backend.app.recommender.content_based import ContentBasedRecommender
from backend.app.recommender.collaborative import CollaborativeRecommender

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"[{status}] {name}" + (f" -> {detail}" if detail else ""))


def main():
    conn = get_db_connection()
    cur = conn.cursor()

    # ---------- A. Artifact & mapping ----------
    check("Mapping phủ đủ 20 phim demo", len(ml_mapping.DEMO_TO_ML) == 20,
          f"{len(ml_mapping.DEMO_TO_ML)} entries")
    check("mlId là duy nhất", len(set(ml_mapping.DEMO_TO_ML.values())) == 20)

    from backend.app.recommender.loader import get_similarity, get_popularity
    sim_mat = get_similarity()
    pop = get_popularity()
    check("Artifact similarity tồn tại", sim_mat is not None)
    check("Artifact popularity tồn tại", pop is not None and len(pop) == 20,
          f"{len(pop) if pop else 0} entries")

    if sim_mat is not None:
        check("Ma trận sim là 20x20", np.array(sim_mat.sim).shape == (20, 20))
        check("Ma trận sim đối xứng", np.allclose(sim_mat.sim, sim_mat.sim.T, atol=1e-5))
        check("Đường chéo sim = 0", np.allclose(np.diag(sim_mat.sim), 0.0, atol=1e-6))
        check("Support có co-rating thật", int(np.array(sim_mat.support).max()) > 1000,
              f"max support = {int(np.array(sim_mat.support).max())}")

    # ---------- B. vote_count thật, không còn dấu vết +120 ----------
    if pop:
        db_counts = {r["id"]: r["vote_count"] for r in cur.execute("SELECT id, vote_count FROM movies;")}
        mism = [d for d in pop if db_counts.get(d) != pop[d]["vote_count"]]
        check("DB vote_count khớp số liệu ML thật (đã backfill)", not mism, f"lệch: {mism}")
        check("Không còn vote_count kiểu +120 (mọi giá trị > 120)",
              all(pop[d]["vote_count"] > 120 for d in pop),
              f"min = {min(pop[d]['vote_count'] for d in pop)}")
        check("mean_rating trong khoảng [0.5, 5]", all(0.5 <= pop[d]["mean_rating"] <= 5.0 for d in pop))

    # ---------- C. Popularity ----------
    pop_engine = PopularityRecommender()
    trending = pop_engine.recommend(conn, top_k=10)
    check("Trending trả về kết quả", len(trending) > 0, f"{len(trending)} phim")
    check("Trending đúng thứ tự giảm dần",
          all(trending[i]["score"] >= trending[i + 1]["score"] for i in range(len(trending) - 1)))
    trending_again = pop_engine.recommend(conn, top_k=10)
    check("Trending deterministic", [m["id"] for m in trending] == [m["id"] for m in trending_again])

    # ---------- D. For-you cá nhân hóa ----------
    content_engine = ContentBasedRecommender()
    hybrid_engine = CollaborativeRecommender()

    for uid, uname in [(1, "alice"), (2, "bob"), (3, "dave")]:
        rated = {r["movie_id"] for r in cur.execute("SELECT movie_id FROM ratings WHERE user_id = ?;", (uid,))}
        recs = hybrid_engine.recommend(conn, user_id=uid, top_k=10)
        ids = [m["id"] for m in recs]
        check(f"for-you({uname}) có kết quả", len(recs) > 0, f"{len(recs)} phim")
        check(f"for-you({uname}) không gợi ý lại phim đã xem",
              not (set(ids) & rated), f"giao = {set(ids) & rated}")
        check(f"for-you({uname}) predicted_score trong [1, 5]",
              all(1.0 <= m["predicted_score"] <= 5.0 for m in recs),
              f"range = [{min(m['predicted_score'] for m in recs)}, {max(m['predicted_score'] for m in recs)}]")
        check(f"for-you({uname}) predicted_score giảm dần",
              all(recs[i]["predicted_score"] >= recs[i + 1]["predicted_score"] for i in range(len(recs) - 1)))

    # ---------- E. Cold-start (charlie, 0 rating) ----------
    charlie_rated = cur.execute("SELECT COUNT(*) FROM ratings WHERE user_id = 4;").fetchone()[0]
    charlie = hybrid_engine.recommend(conn, user_id=4, top_k=10)
    check("charlie có 0 rating (cold-start)", charlie_rated == 0)
    check("for-you(charlie) == trending (fallback popularity)",
          [m["id"] for m in charlie] == [m["id"] for m in trending])
    check("for-you(charlie) trả field predicted_score nhất quán",
          all("predicted_score" in m for m in charlie)
          and all(1.0 <= m["predicted_score"] <= 5.0 for m in charlie))

    # ---------- F. Similar ----------
    for mid in (2, 13):
        sim = content_engine.recommend(conn, movie_id=mid, top_k=5)
        ids = [m["id"] for m in sim]
        check(f"similar({mid}) không chứa chính nó", mid not in ids)
        check(f"similar({mid}) similarity_score giảm dần",
              all(sim[i]["similarity_score"] >= sim[i + 1]["similarity_score"] for i in range(len(sim) - 1)))
        sim_again = content_engine.recommend(conn, movie_id=mid, top_k=5)
        check(f"similar({mid}) deterministic", ids == [m["id"] for m in sim_again])

    # ---------- G. Fallback khi thiếu artifact (không crash) ----------
    orig_pop = popularity_mod.get_popularity
    orig_collab_pop = collab_mod.get_popularity
    orig_sim = collab_mod.get_similarity
    orig_content_sim = content_mod.get_similarity
    try:
        popularity_mod.get_popularity = lambda: None
        collab_mod.get_popularity = lambda: None
        collab_mod.get_similarity = lambda: None
        content_mod.get_similarity = lambda: None

        fb_trending = PopularityRecommender().recommend(conn, top_k=10)
        fb_for_you = CollaborativeRecommender().recommend(conn, user_id=1, top_k=10)
        fb_similar = ContentBasedRecommender().recommend(conn, movie_id=2, top_k=5)

        check("Fallback trending chạy được", len(fb_trending) > 0, f"{len(fb_trending)} phim")
        check("Fallback for-you chạy được", len(fb_for_you) > 0, f"{len(fb_for_you)} phim")
        check("Fallback similar chạy được", len(fb_similar) > 0, f"{len(fb_similar)} phim")
        check("Fallback for-you vẫn trong thang [1,5]",
              all(1.0 <= m["predicted_score"] <= 5.0 for m in fb_for_you))
    finally:
        popularity_mod.get_popularity = orig_pop
        collab_mod.get_popularity = orig_collab_pop
        collab_mod.get_similarity = orig_sim
        content_mod.get_similarity = orig_content_sim

    conn.close()

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} kiểm tra thất bại: {FAILURES}")
        sys.exit(1)
    print("✅ Tất cả kiểm tra đều đạt.")


if __name__ == "__main__":
    main()
