from abc import ABC, abstractmethod
from typing import List, Dict, Any
import sqlite3

class BaseRecommender(ABC):
    """
    Interface cơ sở cho các thuật toán gợi ý trong hệ thống (IT4613)
    """
    @abstractmethod
    def recommend(self, conn: sqlite3.Connection, user_id: int = None, movie_id: int = None, top_k: int = 10) -> List[Dict[str, Any]]:
        pass
