"""Central registry that maps tool names to tool instances."""

from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    """Minimal interface a tool must satisfy."""

    TOOL_NAME: str

    def get_tool_definition(self) -> dict: ...
    def execute(self, tool_input: dict) -> str: ...


class ToolRegistry:
    """Manages available tools and dispatches execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.TOOL_NAME] = tool

    def get_definitions(self) -> list[dict]:
        return [t.get_tool_definition() for t in self._tools.values()]

    def execute(self, tool_name: str, tool_input: dict) -> str:
        tool = self._tools.get(tool_name)
        if tool is None:
            return f'{{"error": "Unknown tool: {tool_name}"}}'
        return tool.execute(tool_input)
