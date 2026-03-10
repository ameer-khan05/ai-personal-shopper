"""Application configuration loaded from environment variables.

Reads a `.env` file at import time and exposes typed settings used
throughout the application.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

import os  # noqa: E402 — must import after load_dotenv so vars are present

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "AI Personal Shopper"
APP_VERSION: str = "0.1.0"
DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
MAX_TOKENS: int = 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_env(key: str) -> str:
    """Return the value of an environment variable or exit with a clear error.

    Args:
        key: Name of the required environment variable.

    Returns:
        The variable's value if set and non-empty.
    """
    value = os.environ.get(key, "").strip()
    if not value or value.startswith("your-"):
        print(
            f"\n[ERROR] Missing required environment variable: {key}\n"
            f"  1. Copy .env.example to .env (if you haven't already)\n"
            f"  2. Replace the placeholder with your real key\n"
        )
        sys.exit(1)
    return value


# ---------------------------------------------------------------------------
# API keys (validated lazily — call these when you actually need the key)
# ---------------------------------------------------------------------------
def get_anthropic_api_key() -> str:
    """Return the Anthropic API key or exit with an error."""
    return require_env("ANTHROPIC_API_KEY")


def get_serpapi_key() -> str:
    """Return the SerpAPI key or exit with an error."""
    return require_env("SERPAPI_KEY")
