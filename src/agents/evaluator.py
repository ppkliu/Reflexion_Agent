"""Evaluator agent — judges whether an answer is correct / good enough."""

from __future__ import annotations

import json
from typing import Any

from agno.agent import Agent

from src.config import get_config

EVALUATOR_INSTRUCTIONS = """\
You are a strict evaluator. Given a task and a candidate answer, judge quality.

Output ONLY valid JSON (no markdown fences):
{"success": true/false, "score": 0.0-1.0, "reason": "brief explanation"}

Scoring guide:
- 0.9-1.0: Excellent, fully correct and complete
- 0.7-0.89: Mostly correct, minor issues
- 0.5-0.69: Partially correct, significant gaps
- 0.0-0.49: Incorrect or irrelevant
"""


def create_evaluator(model_id: str | None = None) -> Agent:
    cfg = get_config()
    mid = model_id or cfg.model.evaluator_model or cfg.model.id

    return Agent(
        name="Evaluator",
        model=mid,
        instructions=[EVALUATOR_INSTRUCTIONS],
        parse_response=True,
    )


def parse_evaluation(raw: str) -> dict[str, Any]:
    """Parse evaluator JSON output, with fallback."""
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    try:
        result = json.loads(text)
        return {
            "success": bool(result.get("success", False)),
            "score": float(result.get("score", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return {"success": False, "score": 0.0, "reason": f"Parse error: {text[:200]}"}
