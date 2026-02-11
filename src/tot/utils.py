"""Utility functions for Tree of Thoughts parsing."""

from __future__ import annotations

import re


def parse_numbered_thoughts(raw: str, expect_k: int = 4) -> list[str]:
    """Parse LLM output containing numbered thoughts like '1. ...', '2. ...'."""
    # Match lines starting with a number followed by . or )
    pattern = r"(?:^|\n)\s*(\d+)[.\)]\s*(.+?)(?=\n\s*\d+[.\)]|\Z)"
    matches = re.findall(pattern, raw, re.DOTALL)

    thoughts = [m[1].strip() for m in matches if m[1].strip()]

    # Fallback: if regex found fewer than expected, split by double newlines
    if len(thoughts) < max(1, expect_k // 2):
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(paragraphs) >= len(thoughts):
            thoughts = paragraphs[:expect_k]

    return thoughts[:expect_k]


def extract_float_score(raw: str) -> float:
    """Extract a float score (0.0 to 1.0) from LLM evaluator output."""
    # Look for a decimal number between 0 and 1 (must not be part of a larger number)
    matches = re.findall(r"(?<!\d)(0\.\d+|1\.0)(?!\d)", raw)
    for m in matches:
        val = float(m)
        if 0.0 <= val <= 1.0:
            return val

    # Fallback: look for "X out of 10" or "X/10" patterns
    out_of_matches = re.findall(r"(\d{1,2})\s*(?:out of|/)\s*10", raw)
    for m in out_of_matches:
        val = int(m)
        if 0 <= val <= 10:
            return val / 10.0

    # Last fallback: standalone integers 0-10
    int_matches = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", raw)
    for m in int_matches:
        val = int(m)
        if 0 <= val <= 10:
            return val / 10.0

    return 0.0


def extract_final_answer(state: str) -> str:
    """Extract the final answer from a ToT leaf state.

    Looks for patterns like 'Final answer: ...' or 'Answer: ...',
    otherwise returns the last paragraph.
    """
    # Try explicit markers
    for marker in ["Final answer:", "Answer:", "ANSWER:", "**Answer**:"]:
        idx = state.lower().rfind(marker.lower())
        if idx != -1:
            return state[idx + len(marker):].strip()

    # Fallback: last non-empty paragraph
    paragraphs = [p.strip() for p in state.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else state.strip()


def summarize_tree(nodes_evaluated: int, depth_reached: int, best_value: float) -> str:
    """Create a brief summary of a ToT run for reflection."""
    return (
        f"Tree search: {nodes_evaluated} nodes evaluated, "
        f"depth {depth_reached}, best value {best_value:.2f}"
    )
