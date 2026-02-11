"""Format retrieved reflections into prompt context blocks."""

from __future__ import annotations

from typing import Any


def format_reflections(reflections: list[dict[str, Any]]) -> str:
    """Format a list of past reflections into a prompt-ready block.

    Args:
        reflections: List of dicts with keys {trial_id, score, reflection, query}.

    Returns:
        Formatted multi-line string for insertion into actor system prompt.
    """
    if not reflections:
        return "No previous reflections available."

    lines = []
    for i, r in enumerate(reflections, 1):
        score = r.get("score", "?")
        reflection = r.get("reflection", "").strip()
        trial_id = r.get("trial_id", f"#{i}")
        lines.append(
            f"[Trial {trial_id} | score={score}]\n{reflection}"
        )

    header = f"=== Past Reflections ({len(reflections)} lessons) ===\n"
    return header + "\n---\n".join(lines) + "\n=== End Reflections ==="


def build_actor_context(
    task_instruction: str,
    task_query: str,
    reflections: list[dict[str, Any]],
) -> str:
    """Build the full actor prompt context including reflections."""
    reflection_block = format_reflections(reflections)

    return f"""{task_instruction}

{reflection_block}

Avoid repeating mistakes from previous trials.
Use the search_knowledge tool to find relevant information when needed.

Current task:
{task_query}

Think step by step, then provide your final answer."""
