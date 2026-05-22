import sqlite3
import json
from typing import Dict, Any

class StateManager:
    def __init__(self, db_path: str = "run_state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for logging every LLM call
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    prompt TEXT,
                    response TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Table for tracking prompt iterations and scores
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_trajectory (
                    iteration INTEGER PRIMARY KEY,
                    prompt TEXT,
                    val_score REAL,
                    accepted BOOLEAN
                )
            ''')
            conn.commit()

    def log_llm_call(self, role: str, prompt: str, response: str, model: str, usage: Dict[str, int], cost: float):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO llm_logs (role, prompt, response, model, input_tokens, output_tokens, cost) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (role, prompt, response, model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cost)
            )
            conn.commit()

    def get_total_cost(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(cost) FROM llm_logs")
            result = cursor.fetchone()[0]
            return result if result else 0.0

    def log_iteration(self, iteration: int, prompt: str, val_score: float, accepted: bool):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO optimization_trajectory (iteration, prompt, val_score, accepted) VALUES (?, ?, ?, ?)",
                (iteration, prompt, val_score, accepted)
            )
            conn.commit()