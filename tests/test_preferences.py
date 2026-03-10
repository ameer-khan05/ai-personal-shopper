"""Tests for preference storage and retrieval."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.memory.preferences import PreferenceStore, UserPreferences
from src.memory.conversation_store import ConversationStore


class TestUserPreferences:
    def test_defaults(self):
        prefs = UserPreferences()
        assert prefs.brand_affinities == []
        assert prefs.brand_avoidances == []
        assert prefs.price_sensitivity is None

    def test_custom_values(self):
        prefs = UserPreferences(
            brand_affinities=["Nike"],
            brand_avoidances=["Adidas"],
            price_sensitivity=7,
        )
        assert prefs.brand_affinities == ["Nike"]
        assert prefs.price_sensitivity == 7


class TestPreferenceStore:
    @pytest.fixture
    def store(self, tmp_path):
        return PreferenceStore(persist_dir=str(tmp_path / "chroma"))

    def test_save_and_retrieve_preference(self, store):
        store.save_preference("color", "blue", "style")
        results = store.get_relevant_preferences("what color")
        assert any("blue" in r for r in results)

    def test_extract_brand_preference(self, store):
        messages = [
            {"role": "user", "content": "I prefer Nike over Adidas"},
        ]
        store.update_from_conversation(messages)
        prefs = store.get_preferences()
        assert "Nike" in prefs.brand_affinities
        assert "Adidas" in prefs.brand_avoidances

    def test_extract_budget(self, store):
        messages = [
            {"role": "user", "content": "My budget is under $150"},
        ]
        store.update_from_conversation(messages)
        prefs = store.get_preferences()
        assert prefs.price_sensitivity is not None

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "persist_test")
        store1 = PreferenceStore(persist_dir=path)
        store1.update_from_conversation([
            {"role": "user", "content": "I prefer Nike over Adidas"},
        ])
        # Force save.
        store1._save_prefs()

        store2 = PreferenceStore(persist_dir=path)
        prefs = store2.get_preferences()
        assert "Nike" in prefs.brand_affinities

    def test_ignores_assistant_messages(self, store):
        messages = [
            {"role": "assistant", "content": "I prefer Nike"},
        ]
        store.update_from_conversation(messages)
        prefs = store.get_preferences()
        assert prefs.brand_affinities == []

    def test_relevant_preferences_empty_store(self, store):
        results = store.get_relevant_preferences("anything")
        assert results == []


class TestConversationStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ConversationStore(db_path=str(tmp_path / "test.db"))

    def test_save_and_retrieve(self, store):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        row_id = store.save_conversation(messages)
        assert row_id is not None

        recent = store.get_recent_conversations(1)
        assert len(recent) == 1
        assert recent[0]["messages"] == messages

    def test_recent_ordering(self, store):
        store.save_conversation([{"role": "user", "content": "first"}])
        store.save_conversation([{"role": "user", "content": "second"}])
        store.save_conversation([{"role": "user", "content": "third"}])

        recent = store.get_recent_conversations(2)
        assert len(recent) == 2
        assert recent[0]["messages"][0]["content"] == "third"
        assert recent[1]["messages"][0]["content"] == "second"

    def test_metadata(self, store):
        store.save_conversation(
            [{"role": "user", "content": "x"}],
            metadata={"topic": "shoes"},
        )
        recent = store.get_recent_conversations(1)
        assert recent[0]["metadata"]["topic"] == "shoes"

    def test_empty_store(self, store):
        assert store.get_recent_conversations() == []
