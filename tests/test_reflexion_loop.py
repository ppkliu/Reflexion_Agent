"""Tests for reflection retriever formatting (no LLM calls)."""

from src.memory.reflection_retriever import build_actor_context, format_reflections


def test_format_reflections_empty():
    result = format_reflections([])
    assert "No previous reflections" in result


def test_format_reflections_with_data():
    reflections = [
        {"trial_id": "t1", "score": 0.3, "reflection": "Missed edge case", "query": "q"},
        {"trial_id": "t2", "score": 0.6, "reflection": "Better approach needed", "query": "q"},
    ]
    result = format_reflections(reflections)
    assert "Past Reflections (2 lessons)" in result
    assert "Missed edge case" in result
    assert "Better approach needed" in result


def test_build_actor_context():
    reflections = [
        {"trial_id": "t1", "score": 0.4, "reflection": "Check null input", "query": "q"},
    ]
    context = build_actor_context(
        task_instruction="Solve the coding problem.",
        task_query="Implement binary search",
        reflections=reflections,
    )
    assert "Solve the coding problem" in context
    assert "Check null input" in context
    assert "Implement binary search" in context
    assert "Avoid repeating mistakes" in context
