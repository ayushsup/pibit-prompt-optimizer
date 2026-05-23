"""
SQLite-backed persistence layer for the optimizer.

Responsibilities:
  - Log every LLM call (input, output, token usage, latency, cost)
  - Track the optimization trajectory (prompt, val_score, accepted) per iteration
  - Cache stochastic metric results (string_semantic, array_llm) for determinism
  - Enable interrupted runs to resume from the last valid checkpoint

Database: run_state.db (created automatically in the working directory)
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class StateManager:
    def __init__(self, db_path: str = "run_state.db"):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Every LLM interaction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_logs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    role          TEXT    NOT NULL,
                    model         TEXT    NOT NULL,
                    prompt        TEXT    NOT NULL,
                    response      TEXT    NOT NULL,
                    input_tokens  INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cost          REAL    DEFAULT 0.0,
                    latency_ms    REAL    DEFAULT 0.0,
                    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Prompt iteration history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_trajectory (
                    iteration INTEGER PRIMARY KEY,
                    prompt    TEXT    NOT NULL,
                    val_score REAL    NOT NULL,
                    accepted  BOOLEAN NOT NULL,
                    breakdown TEXT    DEFAULT '{}',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Deterministic cache for stochastic metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metric_cache (
                    cache_key TEXT PRIMARY KEY,
                    score     REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    # ------------------------------------------------------------------
    # LLM call logging
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        role: str,
        prompt: str,
        response: str,
        model: str,
        usage: Dict[str, int],
        cost: float,
        latency_ms: float = 0.0,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_logs
                    (role, model, prompt, response, input_tokens, output_tokens, cost, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role,
                    model,
                    prompt,
                    response,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    cost,
                    latency_ms,
                ),
            )
            conn.commit()

    def get_total_cost(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT SUM(cost) FROM llm_logs").fetchone()[0]
        return result if result else 0.0

    # ------------------------------------------------------------------
    # Optimization trajectory
    # ------------------------------------------------------------------

    def log_iteration(
        self,
        iteration: int,
        prompt: str,
        val_score: float,
        accepted: bool,
        breakdown: Optional[Dict] = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO optimization_trajectory
                    (iteration, prompt, val_score, accepted, breakdown)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    iteration,
                    prompt,
                    val_score,
                    accepted,
                    json.dumps(breakdown or {}),
                ),
            )
            conn.commit()

    def get_trajectory(self) -> List[Dict]:
        """Return the full recorded trajectory, ordered by iteration."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT iteration, prompt, val_score, accepted FROM optimization_trajectory ORDER BY iteration"
            ).fetchall()
        return [
            {"iteration": r[0], "prompt": r[1], "val_score": r[2], "accepted": bool(r[3])}
            for r in rows
        ]

    def get_best_state(self) -> Optional[Dict]:
        """
        Return the best accepted prompt and score from previous runs.
        Used for resumability: the optimizer can warm-start from here.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT prompt, val_score FROM optimization_trajectory
                WHERE accepted = 1
                ORDER BY val_score DESC, iteration DESC
                LIMIT 1
                """
            ).fetchone()
        if row:
            return {"prompt": row[0], "val_score": row[1]}
        return None

    def get_last_completed_iteration(self) -> int:
        """Returns the highest recorded iteration index, or -1 if none."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(iteration) FROM optimization_trajectory"
            ).fetchone()
        return row[0] if row[0] is not None else -1

    def get_rejected_prompts(self) -> List[str]:
        """Return all prompts that were evaluated but rejected."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT prompt FROM optimization_trajectory WHERE accepted = 0 ORDER BY iteration"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Stochastic metric cache
    # ------------------------------------------------------------------

    def get_metric_cache(self, cache_key: str) -> Optional[float]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT score FROM metric_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return row[0] if row else None

    def set_metric_cache(self, cache_key: str, score: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metric_cache (cache_key, score) VALUES (?, ?)",
                (cache_key, score),
            )
            conn.commit()