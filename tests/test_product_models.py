"""Tests for Product and SearchResults models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.product import Product, SearchResults


class TestProduct:
    def test_minimal_product(self):
        p = Product(name="Widget", price=9.99)
        assert p.name == "Widget"
        assert p.price == 9.99
        assert p.currency == "USD"
        assert p.features == []

    def test_full_product(self):
        p = Product(
            name="Headphones",
            price=79.99,
            currency="USD",
            rating=4.5,
            review_count=1200,
            source="Amazon",
            url="https://example.com",
            image_url="https://example.com/img.jpg",
            description="Great headphones",
            features=["Noise cancelling", "Bluetooth 5.0"],
        )
        assert p.rating == 4.5
        assert p.review_count == 1200
        assert len(p.features) == 2

    def test_optional_fields_default_none(self):
        p = Product(name="X", price=1.0)
        assert p.rating is None
        assert p.review_count is None
        assert p.source is None

    def test_serialization_round_trip(self):
        p = Product(name="Test", price=10.0, rating=4.0)
        data = p.model_dump()
        p2 = Product(**data)
        assert p2 == p


class TestSearchResults:
    def test_basic_search_results(self):
        products = [Product(name="A", price=10.0), Product(name="B", price=20.0)]
        sr = SearchResults(products=products, query="test", total_results=2)
        assert len(sr.products) == 2
        assert sr.query == "test"
        assert sr.total_results == 2

    def test_timestamp_auto_set(self):
        sr = SearchResults(products=[], query="q")
        assert isinstance(sr.timestamp, datetime)

    def test_empty_results(self):
        sr = SearchResults(products=[], query="nothing")
        assert sr.products == []
        assert sr.total_results == 0
