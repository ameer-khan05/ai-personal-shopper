"""Scoring and ranking engine for product comparison."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.models.product import Product


@dataclass
class RankedProduct:
    """A product with its ranking score and breakdown."""

    product: Product
    total_score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def rank_products(
    products: list[Product],
    preferences: dict | None = None,
) -> list[RankedProduct]:
    """Score and rank products by value, returning best first.

    Scoring factors:
    - Price score: lower price relative to category average = higher score
    - Rating score: rating weighted by review count (Bayesian-ish)
    - Value score: rating-to-price ratio
    """
    if not products:
        return []

    prefs = preferences or {}

    prices = [p.price for p in products]
    avg_price = statistics.mean(prices) if prices else 1.0
    max_price = max(prices) if prices else 1.0

    # Preferred / avoided brands from preferences.
    preferred_brands = {b.lower() for b in prefs.get("preferred_brands", [])}
    avoided_brands = {b.lower() for b in prefs.get("avoided_brands", [])}
    max_budget = prefs.get("max_budget")

    ranked: list[RankedProduct] = []
    for product in products:
        breakdown: dict[str, float] = {}

        # --- Price score (0-30 points) ---
        if max_price > 0:
            price_ratio = product.price / max_price
            breakdown["price"] = round((1 - price_ratio) * 30, 2)
        else:
            breakdown["price"] = 15.0

        # Budget bonus: extra points if under stated budget.
        if max_budget and product.price <= max_budget:
            breakdown["price"] = min(30.0, breakdown["price"] + 5.0)

        # --- Rating score (0-35 points) ---
        raw_rating = product.rating or 0.0
        review_count = product.review_count or 0

        # Bayesian adjustment: pull toward 3.5 with low review counts.
        prior_weight = 10
        adjusted_rating = (
            (raw_rating * review_count + 3.5 * prior_weight)
            / (review_count + prior_weight)
        )

        # Review volume bonus (log scale, up to 5 extra points).
        import math
        volume_bonus = min(5.0, math.log1p(review_count) / math.log1p(10000) * 5)

        breakdown["rating"] = round((adjusted_rating / 5.0) * 30 + volume_bonus, 2)

        # --- Value score (0-20 points) ---
        if product.price > 0:
            value_ratio = adjusted_rating / product.price
            # Normalise against category average value.
            avg_value = (3.5 / avg_price) if avg_price > 0 else 1.0
            breakdown["value"] = round(min(20.0, (value_ratio / avg_value) * 10), 2)
        else:
            breakdown["value"] = 10.0

        # --- Brand preference (0-15 points) ---
        name_lower = product.name.lower()
        source_lower = (product.source or "").lower()
        brand_text = f"{name_lower} {source_lower}"

        if any(b in brand_text for b in preferred_brands):
            breakdown["brand_match"] = 15.0
        elif any(b in brand_text for b in avoided_brands):
            breakdown["brand_match"] = -10.0
        else:
            breakdown["brand_match"] = 0.0

        total = sum(breakdown.values())
        explanation = _build_explanation(product, breakdown, total)

        ranked.append(RankedProduct(
            product=product,
            total_score=round(total, 2),
            score_breakdown=breakdown,
            explanation=explanation,
        ))

    ranked.sort(key=lambda r: r.total_score, reverse=True)
    return ranked


def _build_explanation(product: Product, breakdown: dict[str, float], total: float) -> str:
    """Build a human-readable explanation for the ranking."""
    parts: list[str] = []

    if breakdown.get("price", 0) > 20:
        parts.append("competitively priced")
    elif breakdown.get("price", 0) > 10:
        parts.append("reasonably priced")

    rating_score = breakdown.get("rating", 0)
    if rating_score > 25:
        rc = product.review_count or 0
        parts.append(f"highly rated ({product.rating} stars, {rc:,} reviews)")
    elif rating_score > 15:
        parts.append(f"well reviewed ({product.rating} stars)")

    if breakdown.get("value", 0) > 15:
        parts.append("excellent value for money")

    if breakdown.get("brand_match", 0) > 0:
        parts.append("matches your brand preferences")
    elif breakdown.get("brand_match", 0) < 0:
        parts.append("note: this is a brand you've asked to avoid")

    if not parts:
        parts.append("a solid option")

    return f"{product.name}: {', '.join(parts)} (score: {total:.1f}/100)."
