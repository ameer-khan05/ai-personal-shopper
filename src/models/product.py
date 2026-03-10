"""Product data models for shopping search results."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Product(BaseModel):
    """A single product from a shopping search."""

    name: str
    price: float
    currency: str = "USD"
    rating: float | None = None
    review_count: int | None = None
    source: str | None = None
    url: str | None = None
    image_url: str | None = None
    description: str | None = None
    features: list[str] = Field(default_factory=list)


class SearchResults(BaseModel):
    """Container for a batch of product search results."""

    products: list[Product]
    query: str
    total_results: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
