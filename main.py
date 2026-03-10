"""AI Personal Shopper — entry point.

Verifies that the environment is configured correctly and prints a
Rich-formatted welcome banner.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config

console = Console()


def main() -> None:
    """Verify environment setup and display the welcome message."""
    # Validate that the Anthropic API key is present before proceeding.
    config.get_anthropic_api_key()

    title = Text(f"{config.APP_NAME} v{config.APP_VERSION}", style="bold cyan")
    body = Text.assemble(
        ("Status: ", "bold"),
        ("initialized", "bold green"),
        "\n",
        ("Model:  ", "bold"),
        (config.DEFAULT_MODEL, "dim"),
    )

    console.print()
    console.print(
        Panel(
            body,
            title=title,
            subtitle="Ready to shop",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


if __name__ == "__main__":
    main()
