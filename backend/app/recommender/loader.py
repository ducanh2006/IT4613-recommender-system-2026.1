"""
Nạp artifact precompute (tạo bởi backend/etl/build_artifacts.py) vào RAM một lần.

Nếu artifact chưa tồn tại (chưa chạy ETL), các hàm trả về None để recommender
tự fallback về logic cũ -> app luôn khởi động được.
"""
import os
import json
from functools import lru_cache

import numpy as np

from ..config import SIMILARITY_NPZ, POPULARITY_JSON, MAPPING_JSON
from .ml_mapping import DEMO_TO_ML, ML_TO_DEMO, ALL_DEMO_IDS


class SimilarityMatrix:
    """Ma trận tương đồng item-item giữa các phim demo, kèm support (n_ij)."""

    def __init__(self, demo_ids, sim, support):
        self.demo_ids = list(demo_ids)
        self._index = {demo_id: i for i, demo_id in enumerate(self.demo_ids)}
        self.sim = sim
        self.support = support

    def get(self, demo_i: int, demo_j: int) -> float:
        if demo_i not in self._index or demo_j not in self._index:
            return 0.0
        val = self.sim[self._index[demo_i], self._index[demo_j]]
        return float(val) if np.isfinite(val) else 0.0

    def support_of(self, demo_i: int, demo_j: int) -> int:
        if demo_i not in self._index or demo_j not in self._index:
            return 0
        return int(self.support[self._index[demo_i], self._index[demo_j]])


@lru_cache(maxsize=1)
def get_mapping() -> dict:
    """Ánh xạ demo_id -> ml_id. Luôn khả dụng (từ source), không phụ thuộc ETL."""
    return dict(DEMO_TO_ML)


@lru_cache(maxsize=1)
def get_similarity():
    """Trả SimilarityMatrix hoặc None nếu artifact chưa được build."""
    if not os.path.exists(SIMILARITY_NPZ):
        return None
    try:
        data = np.load(SIMILARITY_NPZ)
        demo_ids = data["demo_ids"].tolist()
        sim = data["sim"]
        support = data["support"]
        return SimilarityMatrix(demo_ids, sim, support)
    except (KeyError, ValueError, OSError):
        return None


@lru_cache(maxsize=1)
def get_popularity() -> dict:
    """Trả {demo_id: {"vote_count", "mean_rating", "weighted_rating"}} hoặc None."""
    if not os.path.exists(POPULARITY_JSON):
        return None
    try:
        with open(POPULARITY_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except (ValueError, OSError):
        return None
