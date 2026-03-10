"""Tests for ToolRegistry and shopping search with mocked API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.tools.tool_registry import ToolRegistry
from src.tools.shopping_search import ShoppingSearchTool
from src.tools.comparison_tool import ComparisonTool


class TestToolRegistry:
    def test_register_and_get_definitions(self):
        registry = ToolRegistry()
        tool = MagicMock()
        tool.TOOL_NAME = "test_tool"
        tool.get_tool_definition.return_value = {"name": "test_tool"}
        registry.register(tool)
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "test_tool"

    def test_execute_known_tool(self):
        registry = ToolRegistry()
        tool = MagicMock()
        tool.TOOL_NAME = "my_tool"
        tool.execute.return_value = '{"ok": true}'
        registry.register(tool)
        result = registry.execute("my_tool", {"x": 1})
        assert '"ok"' in result
        tool.execute.assert_called_once_with({"x": 1})

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nope", {})
        assert "Unknown tool" in result

    def test_multiple_tools(self):
        registry = ToolRegistry()
        for name in ["a", "b", "c"]:
            tool = MagicMock()
            tool.TOOL_NAME = name
            tool.get_tool_definition.return_value = {"name": name}
            registry.register(tool)
        assert len(registry.get_definitions()) == 3


class TestShoppingSearchTool:
    def test_tool_definition_schema(self):
        defn = ShoppingSearchTool.get_tool_definition()
        assert defn["name"] == "search_products"
        assert "query" in defn["input_schema"]["properties"]
        assert "query" in defn["input_schema"]["required"]

    @patch("src.tools.shopping_search.httpx.get")
    def test_search_products_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "shopping_results": [
                {
                    "title": "Sony WH-1000XM5",
                    "extracted_price": 298.0,
                    "rating": 4.7,
                    "reviews": 5400,
                    "source": "Amazon",
                    "link": "https://example.com/sony",
                    "thumbnail": "https://img.com/sony.jpg",
                    "snippet": "Premium noise-cancelling headphones",
                },
                {
                    "title": "AirPods Pro",
                    "extracted_price": 189.0,
                    "rating": 4.6,
                    "reviews": 12000,
                    "source": "Apple",
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        tool = ShoppingSearchTool(api_key="fake-key")
        results = tool.search_products("headphones")
        assert len(results.products) == 2
        assert results.products[0].name == "Sony WH-1000XM5"
        assert results.products[0].price == 298.0

    @patch("src.tools.shopping_search.httpx.get")
    def test_search_with_min_rating_filter(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "shopping_results": [
                {"title": "Good", "extracted_price": 50.0, "rating": 4.5},
                {"title": "Bad", "extracted_price": 30.0, "rating": 3.0},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        tool = ShoppingSearchTool(api_key="fake-key")
        results = tool.search_products("stuff", min_rating=4.0)
        assert len(results.products) == 1
        assert results.products[0].name == "Good"

    @patch("src.tools.shopping_search.httpx.get")
    def test_search_network_error_returns_empty(self, mock_get):
        import httpx as _httpx
        mock_get.side_effect = _httpx.RequestError("fail")

        tool = ShoppingSearchTool(api_key="fake-key")
        results = tool.search_products("anything")
        assert results.products == []

    @patch("src.tools.shopping_search.httpx.get")
    def test_execute_returns_json(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"shopping_results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        tool = ShoppingSearchTool(api_key="fake-key")
        result_str = tool.execute({"query": "test"})
        data = json.loads(result_str)
        assert "products" in data


class TestComparisonTool:
    def test_tool_definition(self):
        defn = ComparisonTool.get_tool_definition()
        assert defn["name"] == "compare_products"
        assert "products" in defn["input_schema"]["required"]

    def test_execute_ranks_products(self):
        tool = ComparisonTool()
        result_str = tool.execute({
            "products": [
                {"name": "Cheap Good", "price": 30.0, "rating": 4.5, "review_count": 500},
                {"name": "Expensive OK", "price": 200.0, "rating": 3.5, "review_count": 50},
            ]
        })
        data = json.loads(result_str)
        ranked = data["ranked_products"]
        assert len(ranked) == 2
        # Cheap + good rated should rank higher.
        assert ranked[0]["name"] == "Cheap Good"
