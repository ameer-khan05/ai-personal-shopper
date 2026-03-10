"""Agent pipeline — coordinates the multi-agent flow."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.agents.client import AnthropicClient, ShopperAPIError
from src.agents.orchestrator import OrchestratorAgent
from src.agents.search_agent import SearchAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.shopping_agent import ShoppingAgent

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.tools.tool_registry import ToolRegistry
    from src.memory.preferences import PreferenceStore


class AgentPipeline:
    """Coordinates user input through Orchestrator -> Search -> Recommendation.

    Falls back to the original ShoppingAgent if anything goes wrong,
    ensuring the user always gets a response.
    """

    def __init__(
        self,
        client: AnthropicClient,
        tool_registry: "ToolRegistry | None" = None,
        preference_store: "PreferenceStore | None" = None,
    ) -> None:
        self.client = client
        self.tool_registry = tool_registry
        self.preference_store = preference_store
        self.conversation_history: list[dict] = []

        # Specialist agents.
        self.orchestrator = OrchestratorAgent(client)
        self.search_agent = (
            SearchAgent(client, tool_registry) if tool_registry else None
        )
        self.recommendation_agent = RecommendationAgent(client)

        # Fallback: original single-agent for conversation-only turns.
        self.fallback_agent = ShoppingAgent(
            client,
            tool_registry=tool_registry,
            preference_store=preference_store,
        )

    def chat(
        self,
        user_message: str,
        on_token: "Callable[[str], None] | None" = None,
        on_status: "Callable[[str], None] | None" = None,
    ) -> str:
        """Process a user message through the multi-agent pipeline."""
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )

        try:
            response_text = self._run_pipeline(user_message, on_status)
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
            self._maybe_store_preferences()
            return response_text
        except ShopperAPIError:
            self.conversation_history.pop()
            raise
        except Exception:
            # Pipeline failure — fall back to single agent.
            self.conversation_history.pop()
            return self._run_fallback(user_message, on_token, on_status)

    def reset(self) -> None:
        """Clear conversation history for all agents."""
        self.conversation_history.clear()
        self.fallback_agent.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        user_message: str,
        on_status: "Callable[[str], None] | None",
    ) -> str:
        """Execute the orchestrator -> search -> recommend pipeline."""
        # Step 1: Orchestrator decides intent.
        if on_status:
            on_status("Understanding your request...")

        plan = self.orchestrator.process({
            "user_message": user_message,
            "conversation_history": self.conversation_history[:-1],
        })

        intent = plan.get("intent", "conversation")

        if intent != "search" or not self.search_agent:
            # Pure conversation — delegate to fallback.
            raise _FallbackSignal()

        # Step 2: Search for products.
        search_queries = plan.get("search_queries", [user_message])
        if not search_queries:
            search_queries = [user_message]

        if on_status:
            if len(search_queries) == 1:
                on_status(f"Searching for {search_queries[0]}...")
            else:
                on_status(
                    f"Searching for {len(search_queries)} categories..."
                )

        search_result = self.search_agent.process({
            "search_queries": search_queries,
            "user_message": user_message,
        })

        if search_result.get("error"):
            raise _FallbackSignal()

        search_results = search_result.get("search_results", [])
        has_products = any(
            r.get("products") for r in search_results
        )
        if not has_products:
            raise _FallbackSignal()

        # Step 3: Generate recommendations.
        if on_status:
            on_status("Analysing and ranking results...")

        preferences = self._get_preference_context(user_message)
        rec_result = self.recommendation_agent.process({
            "search_results": search_results,
            "preferences": preferences,
            "user_message": user_message,
            "conversation_history": self.conversation_history[:-1],
        })

        return rec_result.get("response_text", "")

    def _run_fallback(
        self,
        user_message: str,
        on_token: "Callable[[str], None] | None",
        on_status: "Callable[[str], None] | None",
    ) -> str:
        """Run the original ShoppingAgent as a fallback."""
        # Sync history into fallback agent.
        self.fallback_agent.conversation_history = list(
            self.conversation_history
        )
        result = self.fallback_agent.chat(
            user_message, on_token=on_token, on_status=on_status
        )
        # Sync history back.
        self.conversation_history = list(
            self.fallback_agent.conversation_history
        )
        return result

    def _get_preference_context(self, user_message: str) -> dict | None:
        """Build a preferences dict for the recommendation agent."""
        if self.preference_store is None:
            return None
        try:
            prefs = self.preference_store.get_preferences()
            context: dict[str, Any] = {}
            if prefs.brand_affinities:
                context["preferred_brands"] = prefs.brand_affinities
            if prefs.brand_avoidances:
                context["avoided_brands"] = prefs.brand_avoidances
            if prefs.price_sensitivity:
                context["price_sensitivity"] = prefs.price_sensitivity
            return context if context else None
        except Exception:
            return None

    def _maybe_store_preferences(self) -> None:
        if self.preference_store is None:
            return
        try:
            self.preference_store.update_from_conversation(
                self.conversation_history
            )
        except Exception:
            pass


class _FallbackSignal(Exception):
    """Internal signal to trigger fallback to the single agent."""
