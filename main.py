"""AI Personal Shopper — entry point.

Launches an interactive terminal chat loop powered by Claude.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config
from src.agents.client import AnthropicClient, ShopperAPIError
from src.agents.shopping_agent import ShoppingAgent

console = Console()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_welcome() -> None:
    """Display a Rich welcome banner with app info and usage hints."""
    title = Text(f"{config.APP_NAME} v{config.APP_VERSION}", style="bold cyan")
    body = Text.assemble(
        ("Model:  ", "bold"),
        (config.DEFAULT_MODEL, "dim"),
        "\n",
        ("Type ", ""),
        ("quit", "bold"),
        (" or ", ""),
        ("exit", "bold"),
        (" to leave.  ", ""),
        ("Ctrl+C", "bold"),
        (" also works.", ""),
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


def _stream_with_spinner(agent: ShoppingAgent, user_message: str) -> str:
    """Stream the agent response, showing a spinner until the first token."""
    status = console.status("[cyan]Thinking...", spinner="dots")
    status.start()
    first_token = True

    def on_token(text: str) -> None:
        nonlocal first_token
        if first_token:
            status.stop()
            console.print("[bold cyan]Assistant[/bold cyan]")
            first_token = False
        console.file.write(text)
        console.file.flush()

    response = agent.chat(user_message, on_token=on_token)

    if first_token:
        # No tokens arrived (empty response edge case) — stop spinner.
        status.stop()
        console.print("[bold cyan]Assistant[/bold cyan]")

    # Newline after streamed output.
    console.print()
    return response


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_loop(agent: ShoppingAgent) -> None:
    """Run the interactive input loop until the user quits."""
    while True:
        try:
            console.print()
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            console.print("[dim]Goodbye![/dim]")
            break

        try:
            _stream_with_spinner(agent, user_input)
        except ShopperAPIError as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Boot the application."""
    config.get_anthropic_api_key()
    print_welcome()

    client = AnthropicClient()
    agent = ShoppingAgent(client)

    chat_loop(agent)


if __name__ == "__main__":
    main()
