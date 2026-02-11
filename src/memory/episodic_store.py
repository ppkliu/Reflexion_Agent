"""Episodic memory store — persists trial data (trajectory, score, reflection) across sessions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_config


class EpisodicStore:
    """SQLite-backed episodic memory for Reflexion trials."""

    def __init__(self, db_path: str | Path | None = None):
        cfg = get_config()
        db_path = db_path or cfg.db.path
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_table(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflexion_trials (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_category   TEXT NOT NULL,
                    task_key        TEXT NOT NULL,
                    trial_id        TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    query           TEXT NOT NULL,
                    trajectory_digest TEXT,
                    final_answer    TEXT,
                    score           REAL DEFAULT 0.0,
                    reflection      TEXT,
                    used_reflections INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_category_key
                ON reflexion_trials (task_category, task_key)
            """)

    def save_trial(
        self,
        category: str,
        task_key: str,
        trial_id: str,
        query: str,
        trajectory_digest: str = "",
        final_answer: str = "",
        score: float = 0.0,
        reflection: str | None = None,
        used_reflections: int = 0,
    ) -> None:
        """Store a single trial record."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reflexion_trials
                   (task_category, task_key, trial_id, timestamp, query,
                    trajectory_digest, final_answer, score, reflection, used_reflections)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (category, task_key, trial_id, now, query,
                 trajectory_digest, final_answer, score, reflection, used_reflections),
            )

    def get_relevant_reflections(
        self,
        task_key: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve past reflections, most recent first.

        Returns list of {trial_id, score, reflection, query}.
        """
        cfg = get_config()
        top_k = top_k or cfg.reflexion.reflection_top_k

        # keyword-based retrieval: match task_key prefix (first 60 chars)
        key_prefix = task_key[:60] if task_key else ""

        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    """SELECT trial_id, score, reflection, query
                       FROM reflexion_trials
                       WHERE task_category = ? AND task_key LIKE ?
                         AND reflection IS NOT NULL
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (category, f"%{key_prefix}%", top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT trial_id, score, reflection, query
                       FROM reflexion_trials
                       WHERE task_key LIKE ?
                         AND reflection IS NOT NULL
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (f"%{key_prefix}%", top_k),
                ).fetchall()

        return [
            {"trial_id": r[0], "score": r[1], "reflection": r[2], "query": r[3]}
            for r in rows
        ]

    def get_all_trials(self, category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Get all trials for debugging / inspection."""
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    """SELECT trial_id, task_key, score, reflection, timestamp
                       FROM reflexion_trials
                       WHERE task_category = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT trial_id, task_key, score, reflection, timestamp
                       FROM reflexion_trials
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()

        return [
            {"trial_id": r[0], "task_key": r[1], "score": r[2],
             "reflection": r[3], "timestamp": r[4]}
            for r in rows
        ]
