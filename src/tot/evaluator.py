"""State value evaluator for Tree of Thoughts — uses LLM to score intermediate states."""

from __future__ import annotations

import litellm

from src.config import get_config
from src.tot.utils import extract_float_score

VALUE_PROMPT_TEMPLATE = """\
Task: {task_description}

Current reasoning path:
{state}

How promising is this reasoning path for correctly solving the task?
0.0 = hopeless / wrong direction
0.5 = unclear, could go either way
1.0 = very likely correct and on track

Output ONLY a number between 0.0 and 1.0, followed by one sentence explanation.
"""


def evaluate_state(
    state: str,
    task_description: str,
    model_id: str | None = None,
) -> tuple[float, str]:
    """Evaluate how promising an intermediate ToT state is.

    Returns:
        (score, reason) tuple.
    """
    cfg = get_config()
    model = model_id or cfg.model.id

    prompt = VALUE_PROMPT_TEMPLATE.format(
        task_description=task_description,
        state=state[:1500],  # truncate to control token usage
    )

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.1,
    )

    raw = response.choices[0].message.content or ""
    score = extract_float_score(raw)
    reason = raw.strip()

    return score, reason


def evaluate_states_parallel(
    states: list[str],
    task_description: str,
    model_id: str | None = None,
    max_workers: int = 6,
) -> list[tuple[float, str]]:
    """Evaluate multiple states in parallel using ThreadPoolExecutor.

    Returns:
        List of (score, reason) tuples in same order as input states.
    """
    from src.tools.nanobot_parallel import parallel_llm_calls

    prompts = [
        VALUE_PROMPT_TEMPLATE.format(
            task_description=task_description,
            state=state[:1500],
        )
        for state in states
    ]

    raw_results = parallel_llm_calls(
        prompts=prompts,
        model_id=model_id,
        max_tokens=150,
        temperature=0.1,
        max_workers=max_workers,
    )

    return [
        (extract_float_score(raw), raw.strip())
        for raw in raw_results
    ]
