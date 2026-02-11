"""Actor agent — solves tasks using rga_search and past reflections."""

from __future__ import annotations

from agno.agent import Agent
from agno.tools import tool

from src.config import get_config
from src.tools.rga_search import rga_search as _rga_search


@tool
def search_knowledge(query: str, file_pattern: str = "") -> str:
    """Search the local document knowledge base (pdf, docx, md, code, etc.).

    Args:
        query: Regex or keyword to search for.
        file_pattern: Optional glob filter, e.g. '*.pdf'.

    Returns:
        Matching passages with file name and context.
    """
    return _rga_search(query=query, file_pattern=file_pattern or None)


def create_actor(model_id: str | None = None) -> Agent:
    """Create the Actor agent with rga search tool."""
    cfg = get_config()
    mid = model_id or cfg.model.id

    return Agent(
        name="Actor",
        model=mid,
        instructions=[
            "You are a reasoning agent that solves tasks step by step.",
            "Use the search_knowledge tool to find relevant information from local documents.",
            "At the start of each attempt, you will receive PAST REFLECTIONS from previous trials.",
            "Use those lessons to avoid repeating past mistakes.",
            "Think carefully, then provide a clear final answer.",
        ],
        tools=[search_knowledge],
        parse_response=True,
    )
