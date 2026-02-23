"""
attention_store.py
------------------
Thread-safe in-memory store for per-student attention scores.
Each student gets a rolling window of up to MAX_SCORES entries.
"""

import threading
from collections import deque
from typing import Dict, List, Optional


MAX_SCORES = 100  # rolling window per student


class AttentionStore:
    def __init__(self):
        self._lock = threading.Lock()
        # student_id → deque of {"score": float, "timestamp": str}
        self._data: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    def record(self, student_id: str, score: float, timestamp: str) -> None:
        """Store one score entry for a student."""
        with self._lock:
            if student_id not in self._data:
                self._data[student_id] = deque(maxlen=MAX_SCORES)
            self._data[student_id].append({"score": score, "timestamp": timestamp})

    # ------------------------------------------------------------------
    def get_latest(self, student_id: str, n: int = 10) -> List[dict]:
        """Return the last *n* scores for a given student (newest last)."""
        with self._lock:
            if student_id not in self._data:
                return []
            return list(self._data[student_id])[-n:]

    # ------------------------------------------------------------------
    def get_all_latest(self) -> Dict[str, Optional[dict]]:
        """Return the single most-recent score entry per student."""
        with self._lock:
            return {
                sid: scores[-1] if scores else None
                for sid, scores in self._data.items()
            }

    # ------------------------------------------------------------------
    def get_student_ids(self) -> List[str]:
        """Return all currently tracked student IDs."""
        with self._lock:
            return list(self._data.keys())


# Singleton – imported by main.py
store = AttentionStore()
