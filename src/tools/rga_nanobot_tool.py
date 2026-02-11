"""nanobot Tool(ABC) wrapper for rga_search."""

from __future__ import annotations

from typing import Any

from src.tools.rga_search import rga_search

try:
    from nanobot.agent.tools.base import Tool as NanobotTool

    class RgaSearchTool(NanobotTool):
        """Nanobot-compatible tool that searches documents via ripgrep-all."""

        @property
        def name(self) -> str:
            return "rga_search"

        @property
        def description(self) -> str:
            return (
                "Search local documents (pdf, docx, md, code, etc.) using ripgrep-all. "
                "Returns matching passages with file name, line number, and context."
            )

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Regex or keyword to search for in documents.",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob filter, e.g. '*.pdf'. Omit to search all types.",
                    },
                },
                "required": ["query"],
            }

        async def execute(self, query: str, file_pattern: str | None = None, **kwargs: Any) -> str:
            return rga_search(query=query, file_pattern=file_pattern)

except ImportError:
    # nanobot not installed — provide a no-op placeholder
    class RgaSearchTool:  # type: ignore[no-redef]
        """Placeholder when nanobot is not installed."""

        def __init__(self) -> None:
            raise ImportError("nanobot-ai is required for RgaSearchTool")
