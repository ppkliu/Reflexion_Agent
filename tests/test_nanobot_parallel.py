"""Tests for parallel execution utilities (no LLM calls — uses mocks)."""

from unittest.mock import MagicMock, patch

from src.tools.nanobot_parallel import (
    parallel_expand_node,
    parallel_llm_calls,
    parallel_rga_search,
)
from src.tot.evaluator import evaluate_states_parallel
from src.tot.node import ToTNode


def test_parallel_rga_search_calls_rga(monkeypatch):
    """Verify parallel_rga_search dispatches to rga_search for each query."""
    call_log = []

    def mock_rga(query, root_dir=None):
        call_log.append(query)
        return f"result for {query}"

    monkeypatch.setattr("src.tools.nanobot_parallel.rga_search", mock_rga)

    results = parallel_rga_search(["query1", "query2", "query3"], max_workers=3)

    assert len(results) == 3
    assert len(call_log) == 3
    assert "result for query1" in results
    assert "result for query2" in results
    assert "result for query3" in results


@patch("src.tools.nanobot_parallel.litellm")
def test_parallel_llm_calls(mock_litellm):
    """Verify parallel_llm_calls runs all prompts and returns in order."""
    # Mock litellm.completion
    def fake_completion(model, messages, max_tokens, temperature):
        content = f"response to: {messages[0]['content'][:20]}"
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        return resp

    mock_litellm.completion = fake_completion

    prompts = ["prompt A", "prompt B", "prompt C"]
    results = parallel_llm_calls(prompts, model_id="test-model", max_workers=3)

    assert len(results) == 3
    assert "response to: prompt A" in results[0]
    assert "response to: prompt B" in results[1]
    assert "response to: prompt C" in results[2]


@patch("src.tools.nanobot_parallel.litellm")
def test_parallel_expand_node(mock_litellm):
    """Verify parallel_expand_node generates children with values."""
    call_count = {"gen": 0, "eval": 0}

    def fake_completion(model, messages, max_tokens, temperature):
        content_in = messages[0]["content"]
        resp = MagicMock()
        resp.choices = [MagicMock()]

        if "Generate" in content_in:
            # Generation call
            call_count["gen"] += 1
            resp.choices[0].message.content = "1. First approach\n2. Second approach"
        else:
            # Evaluation call
            call_count["eval"] += 1
            resp.choices[0].message.content = "0.75 Looks promising"
        return resp

    mock_litellm.completion = fake_completion

    root = ToTNode(state="Test state", depth=0)
    children = parallel_expand_node(
        node=root,
        system_prompt="System",
        task_query="Test task",
        k=2,
        model_id="test-model",
    )

    assert len(children) >= 1
    assert call_count["gen"] == 1  # one generation call
    assert call_count["eval"] >= 1  # at least one eval call
    for child, value in children:
        assert isinstance(child, ToTNode)
        assert child.depth == 1
        assert child.parent is root


@patch("src.tools.nanobot_parallel.litellm")
def test_evaluate_states_parallel(mock_litellm):
    """Verify evaluate_states_parallel returns scores for all states."""
    def fake_completion(model, messages, max_tokens, temperature):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "0.65 Moderate confidence"
        return resp

    mock_litellm.completion = fake_completion

    states = ["state A", "state B", "state C"]
    results = evaluate_states_parallel(states, "test task", model_id="test")

    assert len(results) == 3
    for score, reason in results:
        assert score == 0.65
        assert "Moderate" in reason
