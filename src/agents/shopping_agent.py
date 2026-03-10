"""Shopping agent that wraps conversation state and streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.client import AnthropicClient, ShopperAPIError

if TYPE_CHECKING:
    from collections.abc import Callable

SYSTEM_PROMPT = """\
You are an expert personal shopping assistant. Your goal is to help the user \
find the perfect product by understanding their needs deeply before making \
recommendations.

Guidelines:
- Always ask clarifying questions about budget, intended use-case, and \
personal preferences before recommending products.
- When you have enough context, provide 2-3 concrete product recommendations \
with brief pros/cons for each.
- Be concise but thorough — prioritise actionable information over filler.
- If the user's request is vague, narrow it down step by step rather than \
guessing.
- Be honest about trade-offs; never oversell a product.
"""


class ShoppingAgent:
    """Stateful shopping advisor backed by an :class:`AnthropicClient`."""

    def __init__(
        self,
        client: AnthropicClient,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt: str = system_prompt or SYSTEM_PROMPT
        self.conversation_history: list[dict] = []

    def chat(
        self,
        user_message: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Send *user_message* and stream back the assistant reply.

        Args:
            user_message: The next user turn.
            on_token: Optional callback invoked with each text chunk as it
                arrives.  Useful for live-printing in a terminal UI.

        Returns:
            The full assistant response text.
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        try:
            full_response: list[str] = []
            with self.client.stream_message(
                messages=list(self.conversation_history),
                system=self.system_prompt,
            ) as stream:
                for text in stream.text_stream:
                    full_response.append(text)
                    if on_token is not None:
                        on_token(text)

            assistant_text = "".join(full_response)
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )
            return assistant_text
        except ShopperAPIError:
            # Roll back the user message so history stays consistent.
            self.conversation_history.pop()
            raise

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
