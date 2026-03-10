"""Search specialist agent — handles product search operations."""

from __future__ import annotations

import json
from typing import Any

from src.agents.agent_base import BaseAgent
from src.agents.client import AnthropicClient
from src.tools.tool_registry import ToolRegistry


class SearchAgent(BaseAgent):
    """Specialist agent for product search via tool use."""

    name = "search"
    system_prompt = (
        "You are a product search specialist. Your ONLY job is to call the "
        "search_products tool with the best possible query based on the user's "
        "request. Formulate precise search queries that will return relevant "
        "results. If the user specifies a price limit, pass it as max_price. "
        "If they mention quality requirements, set min_rating. "
        "Always call the search_products tool — never make up products."
    )

    def __init__(
        self,
        client: AnthropicClient,
        tool_registry: ToolRegistry,
        model: str | None = None,
    ) -> None:
        super().__init__(client, model)
        self.tool_registry = tool_registry

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Search for products based on the user request.

        Input keys:
            - search_queries: list[str] — queries to search for
            - user_message: str — original user message (fallback)
        Returns:
            - search_results: list[dict] — raw search results per query
            - error: str | None
        """
        queries = input_data.get("search_queries", [])
        if not queries:
            user_msg = input_data.get("user_message", "")
            queries = [user_msg] if user_msg else []

        all_results: list[dict] = []
        tools = self.tool_registry.get_definitions()
        search_tools = [t for t in tools if t["name"] == "search_products"]
        if not search_tools:
            return {"search_results": [], "error": "No search tool available"}

        for query in queries:
            messages = [{"role": "user", "content": query}]
            try:
                response = self._create_message(
                    messages=messages, tools=search_tools
                )
                # Process tool calls.
                tool_use_blocks = [
                    b for b in response.content if b.type == "tool_use"
                ]
                if tool_use_blocks:
                    for block in tool_use_blocks:
                        result_str = self.tool_registry.execute(
                            block.name, block.input
                        )
                        result_data = json.loads(result_str)
                        result_data["original_query"] = query
                        all_results.append(result_data)
                else:
                    # LLM didn't call tool — do a direct search as fallback.
                    result_str = self.tool_registry.execute(
                        "search_products", {"query": query}
                    )
                    result_data = json.loads(result_str)
                    result_data["original_query"] = query
                    all_results.append(result_data)
            except Exception as exc:
                all_results.append({
                    "products": [],
                    "original_query": query,
                    "error": str(exc),
                })

        return {"search_results": all_results, "error": None}
