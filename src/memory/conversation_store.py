"""SQLite-based persistent conversation storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class ConversationStore:
    """Stores and retrieves conversations in a local SQLite database."""

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or str(DATA_DIR / "conversations.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        messages: list[dict],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Save a conversation and return its row id."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO conversations (messages, metadata, created_at) VALUES (?, ?, ?)",
            (json.dumps(messages), json.dumps(metadata or {}), now),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_recent_conversations(self, n: int = 5) -> list[dict]:
        """Return the *n* most recent conversations."""
        rows = self._conn.execute(
            "SELECT id, messages, metadata, created_at "
            "FROM conversations ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "messages": json.loads(row["messages"]),
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                messages TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
