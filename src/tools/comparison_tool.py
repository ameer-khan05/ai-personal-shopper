"""Comparison tool that Claude can call to rank and analyse products."""

from __future__ import annotations

import json

from src.agents.comparison import rank_products
from src.models.product import Product


class ComparisonTool:
    """Tool for comparing and ranking a set of products."""

    TOOL_NAME = "compare_products"

    @staticmethod
    def get_tool_definition() -> dict:
        return {
            "name": "compare_products",
            "description": (
                "Compare and rank a list of products based on price, ratings, "
                "review count, and value. Returns products sorted by a composite "
                "score with explanations. Use this after search_products to "
                "analyse and rank results before presenting recommendations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "products": {
                        "type": "array",
                        "description": "List of product objects to compare.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "price": {"type": "number"},
                                "rating": {"type": "number"},
                                "review_count": {"type": "integer"},
                                "source": {"type": "string"},
                                "url": {"type": "string"},
                            },
                            "required": ["name", "price"],
                        },
                    },
                    "preferred_brands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Brands the user prefers (optional).",
                    },
                    "avoided_brands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Brands the user wants to avoid (optional).",
                    },
                    "max_budget": {
                        "type": "number",
                        "description": "Maximum budget in USD (optional).",
                    },
                },
                "required": ["products"],
            },
        }

    def execute(self, tool_input: dict) -> str:
        raw_products = tool_input.get("products", [])
        products = [Product(name=p["name"], price=p["price"],
                            rating=p.get("rating"), review_count=p.get("review_count"),
                            source=p.get("source"), url=p.get("url"))
                    for p in raw_products]

        preferences = {}
        if tool_input.get("preferred_brands"):
            preferences["preferred_brands"] = tool_input["preferred_brands"]
        if tool_input.get("avoided_brands"):
            preferences["avoided_brands"] = tool_input["avoided_brands"]
        if tool_input.get("max_budget"):
            preferences["max_budget"] = tool_input["max_budget"]

        ranked = rank_products(products, preferences or None)

        results = []
        for r in ranked:
            results.append({
                "name": r.product.name,
                "price": r.product.price,
                "rating": r.product.rating,
                "review_count": r.product.review_count,
                "source": r.product.source,
                "url": r.product.url,
                "total_score": r.total_score,
                "score_breakdown": r.score_breakdown,
                "explanation": r.explanation,
            })

        return json.dumps({"ranked_products": results}, indent=2)
