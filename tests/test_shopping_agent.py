"""Tests for ShoppingAgent — all interactions are mocked, no real API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.client import AnthropicClient, ShopperAPIError
from src.agents.shopping_agent import SYSTEM_PROMPT, ShoppingAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str = "mock reply") -> MagicMock:
    """Build a ``MagicMock(spec=AnthropicClient)`` whose ``stream_message``
    returns a context manager that yields *response_text* as a single chunk.
    """
    client = MagicMock(spec=AnthropicClient)

    # The stream context manager exposes a `.text_stream` iterable.
    stream = MagicMock()
    stream.text_stream = iter([response_text])
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)

    client.stream_message.return_value = stream
    return client


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestShoppingAgentInit:
    def test_empty_history(self):
        agent = ShoppingAgent(_make_mock_client())
        assert agent.conversation_history == []

    def test_default_system_prompt(self):
        agent = ShoppingAgent(_make_mock_client())
        assert agent.system_prompt == SYSTEM_PROMPT

    def test_custom_system_prompt(self):
        agent = ShoppingAgent(_make_mock_client(), system_prompt="Be brief.")
        assert agent.system_prompt == "Be brief."

    def test_stores_client_reference(self):
        client = _make_mock_client()
        agent = ShoppingAgent(client)
        assert agent.client is client


# ---------------------------------------------------------------------------
# Chat tests
# ---------------------------------------------------------------------------

class TestShoppingAgentChat:
    def test_returns_response_text(self):
        agent = ShoppingAgent(_make_mock_client("Hello!"))
        result = agent.chat("Hi")
        assert result == "Hello!"

    def test_appends_user_and_assistant_to_history(self):
        agent = ShoppingAgent(_make_mock_client("world"))
        agent.chat("hello")
        assert agent.conversation_history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_multi_turn_history(self):
        client = _make_mock_client()

        agent = ShoppingAgent(client)

        # Turn 1
        client.stream_message.return_value = _stream_ctx("reply1")
        agent.chat("msg1")

        # Turn 2
        client.stream_message.return_value = _stream_ctx("reply2")
        agent.chat("msg2")

        assert len(agent.conversation_history) == 4
        assert agent.conversation_history[0] == {"role": "user", "content": "msg1"}
        assert agent.conversation_history[1] == {"role": "assistant", "content": "reply1"}
        assert agent.conversation_history[2] == {"role": "user", "content": "msg2"}
        assert agent.conversation_history[3] == {"role": "assistant", "content": "reply2"}

    def test_passes_system_prompt_and_history(self):
        client = _make_mock_client("ok")
        agent = ShoppingAgent(client, system_prompt="custom")
        agent.chat("hi")

        client.stream_message.assert_called_once_with(
            messages=[{"role": "user", "content": "hi"}],
            system="custom",
        )

    def test_accepts_tool_registry(self):
        from src.tools.tool_registry import ToolRegistry
        client = _make_mock_client("ok")
        registry = ToolRegistry()
        agent = ShoppingAgent(client, tool_registry=registry)
        assert agent.tool_registry is registry

    def test_on_token_callback(self):
        chunks: list[str] = []
        stream = MagicMock()
        stream.text_stream = iter(["one", "two", "three"])
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)

        client = MagicMock(spec=AnthropicClient)
        client.stream_message.return_value = stream

        agent = ShoppingAgent(client)
        result = agent.chat("go", on_token=chunks.append)

        assert result == "onetwothree"
        assert chunks == ["one", "two", "three"]

    def test_error_rolls_back_user_message(self):
        client = MagicMock(spec=AnthropicClient)
        client.stream_message.side_effect = ShopperAPIError("boom")

        agent = ShoppingAgent(client)
        with pytest.raises(ShopperAPIError):
            agent.chat("oops")

        assert agent.conversation_history == []


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestShoppingAgentReset:
    def test_clears_history(self):
        agent = ShoppingAgent(_make_mock_client("a"))
        agent.chat("b")
        assert len(agent.conversation_history) == 2
        agent.reset()
        assert agent.conversation_history == []


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _stream_ctx(text: str) -> MagicMock:
    """Return a fresh mock stream context manager yielding *text*."""
    stream = MagicMock()
    stream.text_stream = iter([text])
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    return stream
