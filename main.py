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
from src.tools.shopping_search import ShoppingSearchTool
from src.tools.comparison_tool import ComparisonTool
from src.tools.tool_registry import ToolRegistry
from src.memory.preferences import PreferenceStore
from src.memory.conversation_store import ConversationStore

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

    def on_status(msg: str) -> None:
        status.update(f"[cyan]{msg}")

    response = agent.chat(user_message, on_token=on_token, on_status=on_status)

    if first_token:
        # No tokens arrived via streaming (tool-use path) — stop spinner.
        status.stop()
        console.print("[bold cyan]Assistant[/bold cyan]")
        console.print(response)

    # Newline after streamed output.
    console.print()
    return response


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_loop(agent: ShoppingAgent, conversation_store: ConversationStore | None = None) -> None:
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

    # Save conversation on exit.
    if conversation_store and agent.conversation_history:
        conversation_store.save_conversation(agent.conversation_history)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Boot the application."""
    config.get_anthropic_api_key()
    print_welcome()

    client = AnthropicClient()

    # Set up tool registry.
    registry = ToolRegistry()
    try:
        search_tool = ShoppingSearchTool()
        registry.register(search_tool)
    except SystemExit:
        console.print("[yellow]SerpAPI key not set — product search disabled.[/yellow]")

    comparison_tool = ComparisonTool()
    registry.register(comparison_tool)

    # Set up preference store.
    pref_store: PreferenceStore | None = None
    try:
        pref_store = PreferenceStore()
    except Exception:
        console.print("[yellow]Preference store unavailable.[/yellow]")

    # Set up conversation store.
    conv_store: ConversationStore | None = None
    try:
        conv_store = ConversationStore()
    except Exception:
        console.print("[yellow]Conversation store unavailable.[/yellow]")

    agent = ShoppingAgent(
        client,
        tool_registry=registry,
        preference_store=pref_store,
    )

    chat_loop(agent, conversation_store=conv_store)


if __name__ == "__main__":
    main()
