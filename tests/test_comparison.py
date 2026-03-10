"""Tests for the scoring and ranking logic."""

from __future__ import annotations

import pytest

from src.agents.comparison import rank_products, RankedProduct
from src.models.product import Product


class TestRankProducts:
    def test_empty_list(self):
        assert rank_products([]) == []

    def test_single_product(self):
        products = [Product(name="Solo", price=50.0, rating=4.0, review_count=100)]
        ranked = rank_products(products)
        assert len(ranked) == 1
        assert ranked[0].product.name == "Solo"
        assert ranked[0].total_score > 0

    def test_cheaper_product_scores_higher_on_price(self):
        products = [
            Product(name="Expensive", price=200.0, rating=4.0, review_count=100),
            Product(name="Cheap", price=50.0, rating=4.0, review_count=100),
        ]
        ranked = rank_products(products)
        cheap = next(r for r in ranked if r.product.name == "Cheap")
        expensive = next(r for r in ranked if r.product.name == "Expensive")
        assert cheap.score_breakdown["price"] > expensive.score_breakdown["price"]

    def test_higher_rated_scores_higher_on_rating(self):
        products = [
            Product(name="Low", price=50.0, rating=2.0, review_count=1000),
            Product(name="High", price=50.0, rating=4.8, review_count=1000),
        ]
        ranked = rank_products(products)
        high = next(r for r in ranked if r.product.name == "High")
        low = next(r for r in ranked if r.product.name == "Low")
        assert high.score_breakdown["rating"] > low.score_breakdown["rating"]

    def test_review_count_matters(self):
        """4.5 stars with 10K reviews should beat 5.0 with 3 reviews."""
        products = [
            Product(name="Popular", price=50.0, rating=4.5, review_count=10000),
            Product(name="Sparse", price=50.0, rating=5.0, review_count=3),
        ]
        ranked = rank_products(products)
        popular = next(r for r in ranked if r.product.name == "Popular")
        sparse = next(r for r in ranked if r.product.name == "Sparse")
        assert popular.score_breakdown["rating"] > sparse.score_breakdown["rating"]

    def test_brand_preference_boost(self):
        products = [
            Product(name="Nike Air Max", price=100.0, rating=4.0, review_count=500),
            Product(name="Adidas Ultra", price=100.0, rating=4.0, review_count=500),
        ]
        ranked = rank_products(products, preferences={"preferred_brands": ["Nike"]})
        nike = next(r for r in ranked if "Nike" in r.product.name)
        assert nike.score_breakdown["brand_match"] == 15.0

    def test_brand_avoidance_penalty(self):
        products = [
            Product(name="Nike Air Max", price=100.0, rating=4.0, review_count=500),
            Product(name="Adidas Ultra", price=100.0, rating=4.0, review_count=500),
        ]
        ranked = rank_products(products, preferences={"avoided_brands": ["Adidas"]})
        adidas = next(r for r in ranked if "Adidas" in r.product.name)
        assert adidas.score_breakdown["brand_match"] == -10.0

    def test_budget_bonus(self):
        products = [
            Product(name="Under Budget", price=80.0, rating=4.0, review_count=100),
            Product(name="Over Budget", price=180.0, rating=4.0, review_count=100),
        ]
        ranked = rank_products(products, preferences={"max_budget": 100})
        under = next(r for r in ranked if r.product.name == "Under Budget")
        over = next(r for r in ranked if r.product.name == "Over Budget")
        assert under.total_score > over.total_score

    def test_explanation_contains_product_name(self):
        products = [Product(name="Test Widget", price=25.0, rating=4.2, review_count=300)]
        ranked = rank_products(products)
        assert "Test Widget" in ranked[0].explanation

    def test_ordering_best_first(self):
        products = [
            Product(name="Worst", price=999.0, rating=1.0, review_count=1),
            Product(name="Best", price=20.0, rating=4.9, review_count=5000),
            Product(name="Mid", price=50.0, rating=3.5, review_count=200),
        ]
        ranked = rank_products(products)
        assert ranked[0].product.name == "Best"
        assert ranked[-1].product.name == "Worst"
