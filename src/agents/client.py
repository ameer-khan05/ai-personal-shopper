"""Anthropic SDK wrapper with unified error handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic

import config

if TYPE_CHECKING:
    from collections.abc import Callable


class ShopperAPIError(Exception):
    """User-friendly wrapper around Anthropic SDK errors."""


class AnthropicClient:
    """Thin wrapper around :class:`anthropic.Anthropic`.

    Provides streaming and non-streaming message creation with consistent
    error handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or config.DEFAULT_MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS
        self._client = anthropic.Anthropic(
            api_key=api_key or config.get_anthropic_api_key(),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def stream_message(
        self,
        messages: list[dict],
        system: str | None = None,
    ):
        """Return a ``MessageStreamManager`` context manager for streaming.

        Usage::

            with client.stream_message(messages, system) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
        """
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        try:
            return self._client.messages.stream(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise ShopperAPIError(
                "Authentication failed — check your ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ShopperAPIError(
                "Rate limit reached — please wait a moment and try again."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ShopperAPIError(
                "Cannot connect to the Anthropic API — check your network."
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ShopperAPIError(
                "Request timed out — the API may be overloaded, try again."
            ) from exc

    def create_message(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> str:
        """Non-streaming variant that returns the full assistant text."""
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        try:
            response = self._client.messages.create(**kwargs)
            return response.content[0].text
        except anthropic.AuthenticationError as exc:
            raise ShopperAPIError(
                "Authentication failed — check your ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ShopperAPIError(
                "Rate limit reached — please wait a moment and try again."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ShopperAPIError(
                "Cannot connect to the Anthropic API — check your network."
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ShopperAPIError(
                "Request timed out — the API may be overloaded, try again."
            ) from exc
