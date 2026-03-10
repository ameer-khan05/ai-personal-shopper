"""Base class for all specialist agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.agents.client import AnthropicClient


class BaseAgent(ABC):
    """Common interface for specialist agents in the pipeline."""

    name: str = "base"
    system_prompt: str = ""

    def __init__(
        self,
        client: AnthropicClient,
        model: str | None = None,
    ) -> None:
        self.client = client
        self.model_override = model

    @abstractmethod
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process input and return output for the next stage.

        Args:
            input_data: Dict containing at minimum ``user_message`` and
                optionally ``conversation_history``, ``preferences``, etc.

        Returns:
            Dict with the agent's results (keys vary by agent type).
        """

    def _create_message(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        """Create a message, optionally overriding the client's model."""
        original_model = self.client.model
        if self.model_override:
            self.client.model = self.model_override
        try:
            return self.client.create_message(
                messages=messages,
                system=system or self.system_prompt,
                tools=tools,
            )
        finally:
            self.client.model = original_model
