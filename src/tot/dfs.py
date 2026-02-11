"""Tree of Thoughts — DFS (Depth-First Search) variant with backtracking.

Supports parallel evaluation of children at each node via ThreadPoolExecutor
(nanobot subagent-style concurrency).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import litellm

from src.config import get_config
from src.tot.evaluator import evaluate_state, evaluate_states_parallel
from src.tot.node import ToTNode
from src.tot.utils import extract_final_answer, parse_numbered_thoughts, summarize_tree

GENERATE_PROMPT = """\
{system_prompt}

Current reasoning chain:
{state}

Generate {k} distinct next reasoning steps. Each should be a short paragraph (2-5 sentences).
Number them 1 to {k}. Explore different angles.
"""

# Prune threshold — states scoring below this are abandoned (backtrack)
DEFAULT_PRUNE_THRESHOLD = 0.3


def run_tot_dfs(
    query: str,
    system_prompt: str,
    k: int | None = None,
    max_depth: int | None = None,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    model_id: str | None = None,
    tool_executor: Optional[Callable[[str], str]] = None,
    parallel_eval: bool = True,
) -> dict[str, Any]:
    """Run Tree of Thoughts with DFS + backtracking.

    DFS explores the most promising branch first (depth-first), and backtracks
    when a state scores below prune_threshold.

    When parallel_eval=True (default), the k children at each node are
    evaluated concurrently via ThreadPoolExecutor — similar to nanobot's
    SubagentManager spawning parallel tasks for evaluation.

    Args:
        query: The task/question to solve.
        system_prompt: System context (includes past reflections if any).
        k: Branch factor — number of thoughts generated per node.
        max_depth: Maximum tree depth.
        prune_threshold: States below this value trigger backtracking.
        model_id: LLM model to use (litellm format).
        tool_executor: Optional callable for tool use (e.g. rga_search).
        parallel_eval: If True, evaluate children in parallel at each node.

    Returns:
        {final_answer, confidence, path, depth_reached, nodes_evaluated, tree_digest}
    """
    cfg = get_config().tot
    k = k or cfg.branch_factor
    max_depth = max_depth or cfg.max_depth
    model = model_id or get_config().model.id

    root = ToTNode(
        state=f"{system_prompt}\n\nQuestion: {query}",
        depth=0,
    )

    best_node = root
    best_value = -1.0
    nodes_evaluated = 0

    def _dfs(node: ToTNode) -> None:
        nonlocal best_node, best_value, nodes_evaluated

        if node.depth >= max_depth:
            return

        # Generate k candidate thoughts
        gen_prompt = GENERATE_PROMPT.format(
            system_prompt=system_prompt,
            state=node.state[-1500:],
            k=k,
        )

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": gen_prompt}],
            max_tokens=1500,
            temperature=0.7,
        )
        raw = response.choices[0].message.content or ""
        thoughts = parse_numbered_thoughts(raw, expect_k=k)

        # Build next states
        depth = node.depth + 1
        next_states: list[str] = []
        for thought in thoughts:
            state = f"{node.state}\n\nStep {depth}: {thought}"
            if tool_executor and ("search" in thought.lower() or "rga" in thought.lower()):
                tool_result = tool_executor(thought)
                state += f"\n[Tool result]: {tool_result[:500]}"
            next_states.append(state)

        # Evaluate — parallel or sequential
        if parallel_eval and len(next_states) > 1:
            # ── Parallel evaluation (nanobot subagent-style) ──
            eval_results = evaluate_states_parallel(
                states=next_states,
                task_description=query,
                model_id=model,
            )
            nodes_evaluated += len(eval_results)
        else:
            eval_results = []
            for st in next_states:
                value, reason = evaluate_state(st, query, model_id=model)
                eval_results.append((value, reason))
                nodes_evaluated += 1

        # Create child nodes
        children_with_values: list[tuple[ToTNode, float]] = []
        for thought, st, (value, reason) in zip(thoughts, next_states, eval_results):
            child = ToTNode(
                thought=thought,
                state=st,
                parent=node,
                value=value,
                depth=depth,
                eval_reason=reason,
            )
            node.add_child(child)
            children_with_values.append((child, value))

        # Sort children by value descending — explore best first
        children_with_values.sort(key=lambda x: x[1], reverse=True)

        for child, value in children_with_values:
            # Track global best
            if value > best_value:
                best_value = value
                best_node = child

            # Prune: skip low-value branches (backtrack)
            if value < prune_threshold:
                child.pruned = True
                continue

            # Recurse deeper
            _dfs(child)

    _dfs(root)

    path = best_node.get_path()
    final_answer = extract_final_answer(best_node.state)

    return {
        "final_answer": final_answer,
        "confidence": best_node.value,
        "path": path,
        "depth_reached": best_node.depth,
        "nodes_evaluated": nodes_evaluated,
        "tree_digest": summarize_tree(nodes_evaluated, best_node.depth, best_value),
    }
