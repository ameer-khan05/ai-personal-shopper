"""Orchestrator agent — routes user messages to specialist agents."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from src.agents.agent_base import BaseAgent
from src.agents.client import AnthropicClient

if TYPE_CHECKING:
    from src.tools.tool_registry import ToolRegistry
    from src.memory.preferences import PreferenceStore


class OrchestratorAgent(BaseAgent):
    """Decides whether a user message needs product search or is just conversation.

    For shopping requests, it decomposes complex queries into sub-tasks,
    allocates budget, and delegates to specialist agents.
    """

    name = "orchestrator"
    system_prompt = (
        "You are the orchestrator of a shopping assistant. Analyse the user's "
        "message and decide what to do.\n\n"
        "Respond ONLY with a JSON object (no markdown fences, no extra text):\n"
        '{"intent": "search", "queries": ["query1", ...], "budget_per_query": [100, ...]} '
        "for product searches, or\n"
        '{"intent": "conversation"} '
        "for general chat (greetings, follow-up questions, etc.).\n\n"
        "For complex requests like 'home office under $1000', break it into "
        "sub-items with budget allocation:\n"
        '{"intent": "search", "queries": ["office desk", "office chair", '
        '"monitor"], "budget_per_query": [300, 350, 350]}\n\n'
        "Always return valid JSON. Nothing else."
    )

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Analyse user intent and return routing information.

        Input keys:
            - user_message: str
            - conversation_history: list[dict]
        Returns:
            - intent: "search" | "conversation"
            - search_queries: list[str] (if search)
            - budget_per_query: list[float] (if search)
        """
        user_message = input_data.get("user_message", "")
        conversation_history = input_data.get("conversation_history", [])

        messages = list(conversation_history) + [
            {"role": "user", "content": user_message}
        ]

        try:
            response = self._create_message(messages=messages)
            text_blocks = [b for b in response.content if b.type == "text"]
            raw_text = "".join(b.text for b in text_blocks).strip()

            # Parse JSON from the response.
            plan = self._parse_plan(raw_text)
            return plan
        except Exception:
            # Default: treat as conversation if orchestrator fails.
            return {"intent": "conversation"}

    @staticmethod
    def _parse_plan(text: str) -> dict[str, Any]:
        """Extract a JSON plan from the LLM response."""
        # Strip markdown fences if present.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON in the text.
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    plan = json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    return {"intent": "conversation"}
            else:
                return {"intent": "conversation"}

        intent = plan.get("intent", "conversation")
        result: dict[str, Any] = {"intent": intent}

        if intent == "search":
            result["search_queries"] = plan.get("queries", [])
            result["budget_per_query"] = plan.get("budget_per_query", [])

        return result
