"""Recommendation specialist — ranks products and generates explanations."""

from __future__ import annotations

import json
from typing import Any

from src.agents.agent_base import BaseAgent
from src.agents.client import AnthropicClient
from src.agents.comparison import rank_products, RankedProduct
from src.models.product import Product


class RecommendationAgent(BaseAgent):
    """Takes search results + preferences and generates ranked recommendations."""

    name = "recommendation"
    system_prompt = (
        "You are a product recommendation specialist. You receive ranked "
        "product data with scores and must write a clear, helpful recommendation "
        "for the user. Explain WHY you recommend each product. Highlight "
        "trade-offs honestly. If two products are close, help the user decide "
        "by asking what matters more to them. When user preferences are "
        "provided, factor them in and mention that you did so."
    )

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Rank products and generate recommendation text.

        Input keys:
            - search_results: list[dict] — from SearchAgent
            - preferences: dict | None — user preference context
            - user_message: str — original user request
            - conversation_history: list[dict] — prior conversation
        Returns:
            - recommendations: list[dict] — ranked product dicts
            - response_text: str — final recommendation text
            - error: str | None
        """
        search_results = input_data.get("search_results", [])
        preferences = input_data.get("preferences")
        user_message = input_data.get("user_message", "")
        conversation_history = input_data.get("conversation_history", [])

        # Collect all products from search results.
        all_products: list[Product] = []
        for result_set in search_results:
            for p in result_set.get("products", []):
                try:
                    all_products.append(Product(**p))
                except Exception:
                    continue

        if not all_products:
            return {
                "recommendations": [],
                "response_text": (
                    "I wasn't able to find any products matching your request. "
                    "Could you try rephrasing or broadening your search?"
                ),
                "error": None,
            }

        # Rank using the comparison engine.
        ranked = rank_products(all_products, preferences)

        # Build recommendation data for the LLM.
        rec_data = []
        for r in ranked[:10]:
            rec_data.append({
                "name": r.product.name,
                "price": r.product.price,
                "rating": r.product.rating,
                "review_count": r.product.review_count,
                "source": r.product.source,
                "url": r.product.url,
                "score": r.total_score,
                "explanation": r.explanation,
            })

        # Ask the LLM to synthesise a recommendation.
        synthesis_prompt = (
            f"User request: {user_message}\n\n"
            f"Here are the ranked products (best first):\n"
            f"{json.dumps(rec_data, indent=2)}\n\n"
        )
        if preferences:
            synthesis_prompt += f"User preferences: {json.dumps(preferences)}\n\n"
        synthesis_prompt += (
            "Write a clear recommendation highlighting the top 2-3 products. "
            "Explain WHY each is recommended based on the scores and user needs. "
            "Include prices and ratings. If products are close in score, "
            "ask the user what matters more. Be concise."
        )

        messages = list(conversation_history) + [
            {"role": "user", "content": synthesis_prompt}
        ]

        try:
            response = self._create_message(messages=messages)
            text_blocks = [b for b in response.content if b.type == "text"]
            response_text = "".join(b.text for b in text_blocks)
        except Exception:
            # Graceful degradation: return formatted results without LLM.
            lines = ["Here are the top products I found:\n"]
            for i, r in enumerate(rec_data[:3], 1):
                lines.append(
                    f"{i}. **{r['name']}** — ${r['price']:.2f} "
                    f"({r.get('rating', 'N/A')} stars, "
                    f"{r.get('review_count', 0):,} reviews)"
                )
                lines.append(f"   {r['explanation']}")
            response_text = "\n".join(lines)

        return {
            "recommendations": rec_data,
            "response_text": response_text,
            "error": None,
        }
