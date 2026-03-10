"""Streamlit web UI for the AI Personal Shopper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

import config
from src.agents.client import AnthropicClient, ShopperAPIError
from src.agents.pipeline import AgentPipeline
from src.tools.shopping_search import ShoppingSearchTool
from src.tools.comparison_tool import ComparisonTool
from src.tools.tool_registry import ToolRegistry
from src.memory.preferences import PreferenceStore
from src.memory.conversation_store import ConversationStore
from src.ui.styles import CUSTOM_CSS
from src.ui.components import product_card, comparison_table, preference_display
from src.models.product import Product


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Personal Shopper",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session() -> None:
    """Initialise session state on first run."""
    if "initialised" in st.session_state:
        return

    st.session_state.initialised = True
    st.session_state.messages = []
    st.session_state.pipeline = _build_pipeline()
    st.session_state.pref_store = _build_pref_store()
    st.session_state.conv_store = _build_conv_store()


def _build_pipeline() -> AgentPipeline:
    """Create the multi-agent pipeline."""
    client = AnthropicClient()

    registry = ToolRegistry()
    try:
        search_tool = ShoppingSearchTool()
        registry.register(search_tool)
    except SystemExit:
        pass  # SerpAPI not configured.

    comparison_tool = ComparisonTool()
    registry.register(comparison_tool)

    pref_store = _build_pref_store()

    return AgentPipeline(
        client,
        tool_registry=registry,
        preference_store=pref_store,
    )


def _build_pref_store() -> PreferenceStore | None:
    try:
        return PreferenceStore()
    except Exception:
        return None


def _build_conv_store() -> ConversationStore | None:
    try:
        return ConversationStore()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    with st.sidebar:
        st.title("AI Personal Shopper")
        st.caption(f"v{config.APP_VERSION} | {config.DEFAULT_MODEL}")

        st.divider()

        # New conversation button.
        if st.button("New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pipeline.reset()
            st.rerun()

        st.divider()

        # Preferences section.
        st.subheader("Your Preferences")
        pref_store = st.session_state.get("pref_store")
        if pref_store:
            prefs = pref_store.get_preferences()
            preference_display(prefs)
        else:
            st.caption("Preference store unavailable.")

        st.divider()

        # Conversation history.
        st.subheader("Past Conversations")
        conv_store = st.session_state.get("conv_store")
        if conv_store:
            recent = conv_store.get_recent_conversations(5)
            if recent:
                for conv in recent:
                    msgs = conv["messages"]
                    if msgs:
                        first_msg = msgs[0].get("content", "")
                        if isinstance(first_msg, str):
                            preview = first_msg[:50] + ("..." if len(first_msg) > 50 else "")
                            st.caption(f"• {preview}")
            else:
                st.caption("No past conversations yet.")
        else:
            st.caption("Conversation store unavailable.")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
def _render_chat() -> None:
    st.title("AI Personal Shopper")

    # Display existing messages.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input.
    if prompt := st.chat_input("What are you shopping for?"):
        # Add user message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response.
        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            def on_status(msg: str) -> None:
                status_placeholder.markdown(f"*{msg}*")

            try:
                pipeline: AgentPipeline = st.session_state.pipeline
                response = pipeline.chat(prompt, on_status=on_status)
                status_placeholder.empty()
                st.markdown(response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Save conversation.
                conv_store = st.session_state.get("conv_store")
                if conv_store:
                    conv_store.save_conversation(st.session_state.messages)

            except ShopperAPIError as exc:
                status_placeholder.empty()
                st.error(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session()
    _render_sidebar()
    _render_chat()


main()
