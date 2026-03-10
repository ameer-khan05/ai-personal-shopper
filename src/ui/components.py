"""Reusable Streamlit UI components for product display."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.models.product import Product
from src.agents.comparison import RankedProduct
from src.memory.preferences import UserPreferences


def product_card(product: Product, score: float | None = None, explanation: str | None = None) -> None:
    """Render a styled product card."""
    with st.container():
        cols = st.columns([1, 3])

        with cols[0]:
            if product.image_url:
                st.image(product.image_url, width=120)
            else:
                st.markdown("&nbsp;")

        with cols[1]:
            st.markdown(f"**{product.name}**")

            price_rating = f"**${product.price:.2f}**"
            if product.rating:
                stars = star_display(product.rating)
                reviews = f"({product.review_count:,} reviews)" if product.review_count else ""
                price_rating += f" &nbsp; {stars} {reviews}"
            st.markdown(price_rating)

            if product.source:
                st.caption(f"From {product.source}")

            if score is not None:
                st.markdown(
                    f'<span class="product-score">Score: {score:.1f}/100</span>',
                    unsafe_allow_html=True,
                )

            if explanation:
                st.caption(explanation)

            if product.url:
                st.markdown(f"[View Product]({product.url})")

        st.divider()


def comparison_table(ranked_products: list[dict]) -> None:
    """Render a side-by-side comparison of ranked products."""
    if not ranked_products:
        st.info("No products to compare.")
        return

    # Show top 3 side by side.
    top = ranked_products[:3]
    cols = st.columns(len(top))

    for i, (col, rp) in enumerate(zip(cols, top)):
        with col:
            medal = ["1st", "2nd", "3rd"][i]
            st.markdown(
                f'<div class="comparison-header"><b>#{i+1} {medal}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{rp['name']}**")
            st.metric("Price", f"${rp['price']:.2f}")

            if rp.get("rating"):
                st.markdown(f"{star_display(rp['rating'])}")
                if rp.get("review_count"):
                    st.caption(f"{rp['review_count']:,} reviews")

            if rp.get("total_score"):
                st.progress(min(rp["total_score"] / 100, 1.0))
                st.caption(f"Score: {rp['total_score']:.1f}/100")

            if rp.get("explanation"):
                st.caption(rp["explanation"])


def preference_display(preferences: UserPreferences) -> None:
    """Show user preferences in the sidebar."""
    has_prefs = False

    if preferences.brand_affinities:
        has_prefs = True
        st.markdown("**Preferred Brands**")
        chips = " ".join(
            f'<span class="pref-chip">{b}</span>'
            for b in preferences.brand_affinities
        )
        st.markdown(chips, unsafe_allow_html=True)

    if preferences.brand_avoidances:
        has_prefs = True
        st.markdown("**Brands to Avoid**")
        chips = " ".join(
            f'<span class="pref-chip avoid">{b}</span>'
            for b in preferences.brand_avoidances
        )
        st.markdown(chips, unsafe_allow_html=True)

    if preferences.price_sensitivity:
        has_prefs = True
        st.markdown(f"**Price Sensitivity:** {preferences.price_sensitivity}/10")

    if not has_prefs:
        st.caption("No preferences learned yet. Start chatting to build your profile!")


def star_display(rating: float) -> str:
    """Return a star string for a rating value."""
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty + f" {rating:.1f}"
