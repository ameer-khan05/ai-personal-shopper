"""Tests for the multi-agent pipeline — all LLM calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.agents.agent_base import BaseAgent
from src.agents.client import AnthropicClient, ShopperAPIError
from src.agents.orchestrator import OrchestratorAgent
from src.agents.search_agent import SearchAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.pipeline import AgentPipeline
from src.tools.tool_registry import ToolRegistry
from src.models.product import Product


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client() -> MagicMock:
    """Build a MagicMock AnthropicClient."""
    client = MagicMock(spec=AnthropicClient)
    client.model = "claude-sonnet-4-20250514"
    return client


def _mock_response(text: str = "", tool_blocks: list | None = None):
    """Create a mock API response with text and/or tool_use blocks."""
    blocks = []
    if text:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        blocks.append(text_block)
    if tool_blocks:
        for tb in tool_blocks:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tb.get("id", "tool_1")
            block.name = tb.get("name", "search_products")
            block.input = tb.get("input", {"query": "test"})
            blocks.append(block)
    resp = MagicMock()
    resp.content = blocks
    return resp


# ---------------------------------------------------------------------------
# OrchestratorAgent tests
# ---------------------------------------------------------------------------

class TestOrchestratorAgent:
    def test_parse_search_intent(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response(
            '{"intent": "search", "queries": ["wireless earbuds"], "budget_per_query": [100]}'
        )
        orch = OrchestratorAgent(client)
        result = orch.process({"user_message": "find earbuds under $100"})
        assert result["intent"] == "search"
        assert result["search_queries"] == ["wireless earbuds"]

    def test_parse_conversation_intent(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response(
            '{"intent": "conversation"}'
        )
        orch = OrchestratorAgent(client)
        result = orch.process({"user_message": "hello"})
        assert result["intent"] == "conversation"

    def test_parse_malformed_json_falls_back(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response("not json at all")
        orch = OrchestratorAgent(client)
        result = orch.process({"user_message": "hi"})
        assert result["intent"] == "conversation"

    def test_parse_json_in_markdown_fences(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response(
            '```json\n{"intent": "search", "queries": ["laptop"]}\n```'
        )
        orch = OrchestratorAgent(client)
        result = orch.process({"user_message": "find laptop"})
        assert result["intent"] == "search"
        assert result["search_queries"] == ["laptop"]

    def test_multi_query_decomposition(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response(
            '{"intent": "search", "queries": ["desk", "chair", "monitor"], '
            '"budget_per_query": [300, 350, 350]}'
        )
        orch = OrchestratorAgent(client)
        result = orch.process({"user_message": "home office under $1000"})
        assert len(result["search_queries"]) == 3
        assert result["budget_per_query"] == [300, 350, 350]

    def test_api_error_defaults_to_conversation(self):
        client = _mock_client()
        client.create_message.side_effect = Exception("API down")
        orch = OrchestratorAgent(client)
        result = orch.process({"user_message": "anything"})
        assert result["intent"] == "conversation"


# ---------------------------------------------------------------------------
# SearchAgent tests
# ---------------------------------------------------------------------------

class TestSearchAgent:
    def test_search_with_tool_call(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response(
            tool_blocks=[{
                "name": "search_products",
                "input": {"query": "earbuds"},
            }]
        )
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.TOOL_NAME = "search_products"
        mock_tool.get_tool_definition.return_value = {
            "name": "search_products",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        mock_tool.execute.return_value = json.dumps({
            "products": [{"name": "TestBud", "price": 50.0}],
            "query": "earbuds",
        })
        registry.register(mock_tool)

        agent = SearchAgent(client, registry)
        result = agent.process({"search_queries": ["earbuds"]})
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["products"][0]["name"] == "TestBud"

    def test_search_no_tool_available(self):
        client = _mock_client()
        registry = ToolRegistry()
        agent = SearchAgent(client, registry)
        result = agent.process({"search_queries": ["earbuds"]})
        assert result["error"] == "No search tool available"

    def test_search_fallback_on_no_tool_call(self):
        client = _mock_client()
        # LLM returns text instead of tool call.
        client.create_message.return_value = _mock_response("I'll search for you")

        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.TOOL_NAME = "search_products"
        mock_tool.get_tool_definition.return_value = {
            "name": "search_products",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        mock_tool.execute.return_value = json.dumps({
            "products": [], "query": "earbuds",
        })
        registry.register(mock_tool)

        agent = SearchAgent(client, registry)
        result = agent.process({"search_queries": ["earbuds"]})
        # Should have done a direct search fallback.
        assert result["search_results"] is not None


# ---------------------------------------------------------------------------
# RecommendationAgent tests
# ---------------------------------------------------------------------------

class TestRecommendationAgent:
    def test_ranks_and_recommends(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response(
            "I recommend Product A because it has the best value."
        )
        agent = RecommendationAgent(client)
        result = agent.process({
            "search_results": [{
                "products": [
                    {"name": "Product A", "price": 30.0, "rating": 4.5,
                     "review_count": 500},
                    {"name": "Product B", "price": 100.0, "rating": 3.0,
                     "review_count": 10},
                ],
            }],
            "user_message": "find headphones",
        })
        assert "recommend" in result["response_text"].lower()
        assert len(result["recommendations"]) == 2

    def test_empty_results(self):
        client = _mock_client()
        agent = RecommendationAgent(client)
        result = agent.process({
            "search_results": [{"products": []}],
            "user_message": "find unicorns",
        })
        assert "wasn't able to find" in result["response_text"]

    def test_graceful_degradation_on_llm_failure(self):
        client = _mock_client()
        client.create_message.side_effect = Exception("LLM down")
        agent = RecommendationAgent(client)
        result = agent.process({
            "search_results": [{
                "products": [
                    {"name": "Widget", "price": 25.0, "rating": 4.0,
                     "review_count": 100},
                ],
            }],
            "user_message": "find widgets",
        })
        # Should still return formatted results.
        assert "Widget" in result["response_text"]
        assert result["error"] is None


# ---------------------------------------------------------------------------
# AgentPipeline tests
# ---------------------------------------------------------------------------

class TestAgentPipeline:
    def test_search_flow_end_to_end(self):
        client = _mock_client()
        # First call: orchestrator returns search intent.
        # Second call: search agent tool call.
        # Third call: recommendation synthesis.
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Orchestrator.
                return _mock_response(
                    '{"intent": "search", "queries": ["earbuds"]}'
                )
            elif call_count[0] == 2:
                # Search agent — returns tool call.
                return _mock_response(tool_blocks=[{
                    "name": "search_products",
                    "input": {"query": "earbuds"},
                }])
            else:
                # Recommendation agent.
                return _mock_response("Top pick: TestBuds at $50.")

        client.create_message.side_effect = side_effect

        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.TOOL_NAME = "search_products"
        mock_tool.get_tool_definition.return_value = {
            "name": "search_products",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        mock_tool.execute.return_value = json.dumps({
            "products": [{"name": "TestBuds", "price": 50.0, "rating": 4.5,
                          "review_count": 1000}],
        })
        registry.register(mock_tool)

        pipeline = AgentPipeline(client, tool_registry=registry)
        result = pipeline.chat("find earbuds")
        assert "TestBuds" in result
        assert len(pipeline.conversation_history) == 2

    def test_conversation_falls_back(self):
        client = _mock_client()
        # Orchestrator says conversation.
        client.create_message.return_value = _mock_response(
            '{"intent": "conversation"}'
        )
        # Fallback agent needs streaming.
        stream = MagicMock()
        stream.text_stream = iter(["Hello there!"])
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        client.stream_message.return_value = stream

        pipeline = AgentPipeline(client)
        result = pipeline.chat("hello")
        assert "Hello" in result

    def test_reset_clears_all(self):
        client = _mock_client()
        stream = MagicMock()
        stream.text_stream = iter(["hi"])
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        client.stream_message.return_value = stream
        client.create_message.return_value = _mock_response(
            '{"intent": "conversation"}'
        )

        pipeline = AgentPipeline(client)
        pipeline.chat("hi")
        assert len(pipeline.conversation_history) == 2
        pipeline.reset()
        assert pipeline.conversation_history == []

    def test_pipeline_error_uses_fallback(self):
        client = _mock_client()
        # Orchestrator throws.
        client.create_message.side_effect = Exception("boom")
        # Fallback streaming.
        stream = MagicMock()
        stream.text_stream = iter(["Fallback response"])
        stream.__enter__ = MagicMock(return_value=stream)
        stream.__exit__ = MagicMock(return_value=False)
        client.stream_message.return_value = stream

        pipeline = AgentPipeline(client)
        result = pipeline.chat("find shoes")
        assert "Fallback response" in result

    def test_on_status_callback(self):
        client = _mock_client()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(
                    '{"intent": "search", "queries": ["shoes"]}'
                )
            elif call_count[0] == 2:
                return _mock_response(tool_blocks=[{
                    "name": "search_products",
                    "input": {"query": "shoes"},
                }])
            else:
                return _mock_response("Buy these shoes.")

        client.create_message.side_effect = side_effect

        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.TOOL_NAME = "search_products"
        mock_tool.get_tool_definition.return_value = {
            "name": "search_products",
            "description": "Search",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        mock_tool.execute.return_value = json.dumps({
            "products": [{"name": "RunShoe", "price": 80.0, "rating": 4.2,
                          "review_count": 200}],
        })
        registry.register(mock_tool)

        statuses: list[str] = []
        pipeline = AgentPipeline(client, tool_registry=registry)
        pipeline.chat("find shoes", on_status=statuses.append)
        assert any("request" in s.lower() or "search" in s.lower() for s in statuses)


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------

class TestBaseAgent:
    def test_model_override(self):
        client = _mock_client()
        client.create_message.return_value = _mock_response("ok")

        class TestAgent(BaseAgent):
            name = "test"
            system_prompt = "test"

            def process(self, input_data):
                self._create_message([{"role": "user", "content": "hi"}])
                return {}

        agent = TestAgent(client, model="claude-haiku-4-5-20251001")
        agent.process({})
        # Model should be restored after call.
        assert client.model == "claude-sonnet-4-20250514"

    def test_abstract_process_required(self):
        with pytest.raises(TypeError):
            BaseAgent(_mock_client())  # type: ignore
