"""User preference storage and retrieval using ChromaDB."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings


class UserPreferences(BaseModel):
    """Aggregate user preference model."""

    brand_affinities: list[str] = Field(default_factory=list)
    brand_avoidances: list[str] = Field(default_factory=list)
    price_sensitivity: int | None = None  # 1-10
    category_preferences: dict[str, str] = Field(default_factory=dict)
    size_info: dict[str, str] = Field(default_factory=dict)
    preferred_retailers: list[str] = Field(default_factory=list)
    past_searches: list[str] = Field(default_factory=list)


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class PreferenceStore:
    """Persists user preferences in ChromaDB for semantic retrieval."""

    def __init__(self, persist_dir: str | None = None) -> None:
        persist_path = persist_dir or str(DATA_DIR / "chromadb")
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name="user_preferences",
        )
        self._prefs_path = Path(persist_path) / "user_prefs.json"
        self._prefs = self._load_prefs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_preference(self, key: str, value: str, category: str = "general") -> None:
        """Store a single preference."""
        doc_id = f"{category}:{key}"
        doc_text = f"{key}: {value}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{"category": category, "key": key, "value": value}],
        )

    def get_preferences(self, category: str | None = None) -> UserPreferences:
        """Return the current aggregated user preferences."""
        return self._prefs

    def get_relevant_preferences(self, query: str) -> list[str]:
        """Semantic search for preferences relevant to a query."""
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=5,
            )
            docs = results.get("documents", [[]])[0]
            return [d for d in docs if d]
        except Exception:
            return []

    def update_from_conversation(self, messages: list[dict]) -> None:
        """Extract explicit preferences from conversation messages."""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                continue
            self._extract_preferences(content)
        self._save_prefs()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_preferences(self, text: str) -> None:
        """Simple rule-based preference extraction from user text."""
        text_lower = text.lower()

        # Brand preferences: "I prefer X" / "I like X" / "I love X"
        prefer_patterns = [
            r"i (?:prefer|like|love|want)\s+(\w+)\s+(?:over|more than|instead of)\s+(\w+)",
            r"i (?:prefer|like|love)\s+(\w+)",
        ]
        for pattern in prefer_patterns:
            match = re.search(pattern, text_lower)
            if match:
                brand = match.group(1).title()
                if brand not in self._prefs.brand_affinities:
                    self._prefs.brand_affinities.append(brand)
                    self.save_preference("brand_affinity", brand, "brands")
                if match.lastindex and match.lastindex >= 2:
                    avoid = match.group(2).title()
                    if avoid not in self._prefs.brand_avoidances:
                        self._prefs.brand_avoidances.append(avoid)
                        self.save_preference("brand_avoidance", avoid, "brands")
                break

        # Brand avoidance: "I don't like X" / "I hate X" / "avoid X"
        avoid_patterns = [
            r"(?:i (?:don'?t like|hate|dislike)|avoid|no)\s+(\w+)",
        ]
        for pattern in avoid_patterns:
            match = re.search(pattern, text_lower)
            if match:
                brand = match.group(1).title()
                if brand not in self._prefs.brand_avoidances:
                    self._prefs.brand_avoidances.append(brand)
                    self.save_preference("brand_avoidance", brand, "brands")

        # Budget: "budget is $X" / "under $X" / "max $X"
        budget_patterns = [
            r"(?:budget|max|under|less than|up to)\s*(?:is\s*)?\$?\s*(\d+)",
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, text_lower)
            if match:
                budget = int(match.group(1))
                sensitivity = min(10, max(1, budget // 50))
                # Lower budget = higher sensitivity
                if budget < 100:
                    sensitivity = 8
                elif budget < 200:
                    sensitivity = 6
                else:
                    sensitivity = 4
                self._prefs.price_sensitivity = sensitivity
                self.save_preference("budget", str(budget), "budget")
                break

    def _load_prefs(self) -> UserPreferences:
        if self._prefs_path.exists():
            try:
                data = json.loads(self._prefs_path.read_text())
                return UserPreferences(**data)
            except Exception:
                pass
        return UserPreferences()

    def _save_prefs(self) -> None:
        self._prefs_path.parent.mkdir(parents=True, exist_ok=True)
        self._prefs_path.write_text(self._prefs.model_dump_json(indent=2))
