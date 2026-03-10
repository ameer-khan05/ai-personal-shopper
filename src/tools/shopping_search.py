"""SerpAPI Google Shopping search tool for the agent."""

from __future__ import annotations

import json
from typing import Any

import httpx

import config
from src.models.product import Product, SearchResults

SERPAPI_BASE_URL = "https://serpapi.com/search"


class ShoppingSearchTool:
    """Searches Google Shopping via SerpAPI and returns normalised results."""

    TOOL_NAME = "search_products"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or config.get_serpapi_key()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_products(
        self,
        query: str,
        max_price: float | None = None,
        min_rating: float | None = None,
        num_results: int = 10,
    ) -> SearchResults:
        """Run a Google Shopping search and return normalised results."""
        params: dict[str, Any] = {
            "engine": "google_shopping",
            "q": query,
            "api_key": self.api_key,
            "num": str(num_results),
        }
        if max_price is not None:
            params["tbs"] = f"mr:1,price:1,ppr_max:{int(max_price)}"

        try:
            resp = httpx.get(SERPAPI_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return SearchResults(products=[], query=query, total_results=0)
        except httpx.HTTPStatusError:
            return SearchResults(products=[], query=query, total_results=0)
        except httpx.RequestError:
            return SearchResults(products=[], query=query, total_results=0)

        raw_results = data.get("shopping_results", [])
        products = self._normalize_results(raw_results, min_rating)
        return SearchResults(
            products=products[:num_results],
            query=query,
            total_results=len(products),
        )

    @staticmethod
    def get_tool_definition() -> dict:
        """Return the Claude tool-use schema for this tool."""
        return {
            "name": "search_products",
            "description": (
                "Search for products on Google Shopping. Returns a list of "
                "products with names, prices, ratings, and links. Use this "
                "whenever the user wants to find, compare, or buy products."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The product search query.",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price filter in USD (optional).",
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum star rating filter 1-5 (optional).",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, tool_input: dict) -> str:
        """Execute the tool with the given input and return JSON string."""
        results = self.search_products(
            query=tool_input["query"],
            max_price=tool_input.get("max_price"),
            min_rating=tool_input.get("min_rating"),
            num_results=tool_input.get("num_results", 10),
        )
        return results.model_dump_json()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_results(
        raw: list[dict], min_rating: float | None = None
    ) -> list[Product]:
        """Convert raw SerpAPI shopping results into Product models."""
        products: list[Product] = []
        for item in raw:
            price = item.get("extracted_price")
            if price is None:
                continue
            rating = item.get("rating")
            if min_rating is not None and (rating is None or rating < min_rating):
                continue
            products.append(
                Product(
                    name=item.get("title", "Unknown"),
                    price=float(price),
                    currency="USD",
                    rating=float(rating) if rating else None,
                    review_count=item.get("reviews", None),
                    source=item.get("source"),
                    url=item.get("link"),
                    image_url=item.get("thumbnail"),
                    description=item.get("snippet"),
                )
            )
        return products
