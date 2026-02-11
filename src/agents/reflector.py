"""Reflector agent — generates verbal self-reflection on failed trials."""

from __future__ import annotations

from agno.agent import Agent

from src.config import get_config

REFLECTOR_INSTRUCTIONS = """\
You analyze a reasoning trajectory and its evaluation outcome, then produce
a concise, specific, actionable self-reflection.

Focus on:
- What went wrong? Why?
- What key insight or information was missing?
- What concrete step should be taken differently next time?

Format: 3-5 bullet points. Be brutally honest but constructive.
"""


def create_reflector(model_id: str | None = None) -> Agent:
    cfg = get_config()
    mid = model_id or cfg.model.reflector_model or cfg.model.id

    return Agent(
        name="Reflector",
        model=mid,
        instructions=[REFLECTOR_INSTRUCTIONS],
        parse_response=True,
    )
