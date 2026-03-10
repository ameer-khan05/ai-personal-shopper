"""Shopping agent that wraps conversation state, streaming, and tool use."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.agents.client import AnthropicClient, ShopperAPIError

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.tools.tool_registry import ToolRegistry
    from src.memory.preferences import PreferenceStore

SYSTEM_PROMPT = """\
You are an expert personal shopping assistant. Your goal is to help the user \
find the perfect product by understanding their needs deeply before making \
recommendations.

Guidelines:
- Always ask clarifying questions about budget, intended use-case, and \
personal preferences before recommending products.
- When you have enough context, use the search_products tool to find real \
products. Do NOT make up product names or prices.
- After receiving search results, use the compare_products tool to rank and \
analyse the options before presenting recommendations.
- Present 2-3 concrete product recommendations with brief pros/cons for each.
- Explain WHY you recommend something ("I'm recommending X because it has \
the best balance of price and reviews").
- When two products are close, help the user decide through dialogue \
("X is cheaper but Y has better reviews — what matters more to you?").
- For multi-item requests, break them into sub-items with budget allocation.
- Be concise but thorough — prioritise actionable information over filler.
- If the user's request is vague, narrow it down step by step rather than \
guessing.
- Be honest about trade-offs; never oversell a product.
- When you know user preferences (brands, budgets, styles), factor them in \
and mention that you did so.
"""


class ShoppingAgent:
    """Stateful shopping advisor backed by an :class:`AnthropicClient`."""

    def __init__(
        self,
        client: AnthropicClient,
        system_prompt: str | None = None,
        tool_registry: "ToolRegistry | None" = None,
        preference_store: "PreferenceStore | None" = None,
    ) -> None:
        self.client = client
        self.system_prompt: str = system_prompt or SYSTEM_PROMPT
        self.conversation_history: list[dict] = []
        self.tool_registry = tool_registry
        self.preference_store = preference_store

    def chat(
        self,
        user_message: str,
        on_token: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Send *user_message* and stream back the assistant reply.

        Handles the tool-use loop: if Claude requests a tool, execute it,
        send the result back, and continue until we get a final text response.
        """
        self.conversation_history.append({"role": "user", "content": user_message})

        # Build system prompt, optionally enriched with preferences.
        system = self._build_system_prompt(user_message)

        try:
            response_text = self._run_agent_loop(system, on_token, on_status)
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
            # Extract preferences from conversation asynchronously.
            self._maybe_store_preferences()
            return response_text
        except ShopperAPIError:
            self.conversation_history.pop()
            raise

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_system_prompt(self, user_message: str) -> str:
        """Add preference context to the system prompt if available."""
        system = self.system_prompt
        if self.preference_store is not None:
            try:
                prefs = self.preference_store.get_relevant_preferences(user_message)
                if prefs:
                    pref_text = "\n".join(f"- {p}" for p in prefs)
                    system += (
                        f"\n\nUser preferences (use these to personalise recommendations):\n"
                        f"{pref_text}"
                    )
                user_prefs = self.preference_store.get_preferences()
                if user_prefs and user_prefs.brand_affinities:
                    brands = ", ".join(user_prefs.brand_affinities)
                    system += f"\nPreferred brands: {brands}"
                if user_prefs and user_prefs.brand_avoidances:
                    avoid = ", ".join(user_prefs.brand_avoidances)
                    system += f"\nBrands to avoid: {avoid}"
                if user_prefs and user_prefs.price_sensitivity:
                    system += f"\nPrice sensitivity: {user_prefs.price_sensitivity}/10"
            except Exception:
                pass  # Don't let preference errors break chat.
        return system

    def _run_agent_loop(
        self,
        system: str,
        on_token: Callable[[str], None] | None,
        on_status: Callable[[str], None] | None,
    ) -> str:
        """Run the tool-use loop until we get a final text response."""
        tools = self.tool_registry.get_definitions() if self.tool_registry else None

        # First call: stream text response.
        # If no tools are registered, just stream directly.
        if not tools:
            return self._stream_text(system, on_token)

        # With tools, use create_message (non-streaming) for tool calls,
        # then stream the final text response.
        messages = list(self.conversation_history)
        max_iterations = 10

        for _ in range(max_iterations):
            response = self.client.create_message(
                messages=messages, system=system, tools=tools
            )

            # Check if the response contains tool use.
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_use_blocks:
                # No tool calls — return the text.
                return "".join(b.text for b in text_blocks)

            # Process tool calls.
            # Add assistant message with all content blocks.
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool and build tool_result messages.
            tool_results = []
            for block in tool_use_blocks:
                if on_status:
                    on_status(f"Searching for {block.input.get('query', block.name)}...")
                result = self.tool_registry.execute(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

        # Fallback: stream the final response.
        return "".join(b.text for b in text_blocks) if text_blocks else ""

    def _stream_text(
        self,
        system: str,
        on_token: Callable[[str], None] | None,
    ) -> str:
        """Stream a text-only response (no tools)."""
        full_response: list[str] = []
        with self.client.stream_message(
            messages=list(self.conversation_history),
            system=system,
        ) as stream:
            for text in stream.text_stream:
                full_response.append(text)
                if on_token is not None:
                    on_token(text)
        return "".join(full_response)

    def _maybe_store_preferences(self) -> None:
        """Extract and store preferences from the conversation."""
        if self.preference_store is None:
            return
        try:
            self.preference_store.update_from_conversation(self.conversation_history)
        except Exception:
            pass  # Best-effort.
