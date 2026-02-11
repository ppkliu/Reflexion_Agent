"""Tree of Thoughts — BFS (Beam Search) variant with parallel node expansion."""

from __future__ import annotations

from typing import Any, Callable, Optional

import litellm

from src.config import get_config
from src.tot.evaluator import evaluate_state
from src.tot.node import ToTNode
from src.tot.utils import extract_final_answer, parse_numbered_thoughts, summarize_tree

GENERATE_PROMPT = """\
{system_prompt}

Current reasoning chain:
{state}

Generate {k} distinct, creative next reasoning steps. Each should be a short paragraph (2-5 sentences).
Number them 1 to {k}. Explore different angles.
"""


def run_tot_bfs(
    query: str,
    system_prompt: str,
    k: int | None = None,
    max_depth: int | None = None,
    beam_width: int | None = None,
    model_id: str | None = None,
    tool_executor: Optional[Callable[[str], str]] = None,
    parallel: bool = True,
) -> dict[str, Any]:
    """Run Tree of Thoughts with BFS (beam search).

    When parallel=True (default), each layer's frontier nodes are expanded
    concurrently via ThreadPoolExecutor — each node's generate + evaluate
    runs in its own thread, similar to nanobot's subagent spawn pattern.

    Args:
        query: The task/question to solve.
        system_prompt: System context (includes past reflections if any).
        k: Branch factor — number of thoughts generated per node.
        max_depth: Maximum tree depth.
        beam_width: Number of nodes kept per level (beam).
        model_id: LLM model to use (litellm format).
        tool_executor: Optional callable for tool use (e.g. rga_search).
        parallel: If True, expand frontier nodes in parallel (nanobot-style).

    Returns:
        {final_answer, confidence, path, depth_reached, nodes_evaluated, tree_digest}
    """
    cfg = get_config().tot
    k = k or cfg.branch_factor
    max_depth = max_depth or cfg.max_depth
    beam_width = beam_width or cfg.beam_width
    model = model_id or get_config().model.id

    root = ToTNode(
        state=f"{system_prompt}\n\nQuestion: {query}",
        depth=0,
    )
    frontier = [root]
    best_node = root
    best_value = -1.0
    nodes_evaluated = 0

    for depth in range(1, max_depth + 1):
        if parallel:
            # ── Parallel expansion (nanobot subagent-style) ──
            # Each frontier node expanded concurrently in its own thread
            from src.tools.nanobot_parallel import parallel_expand_frontier

            children_with_values = parallel_expand_frontier(
                frontier=frontier,
                system_prompt=system_prompt,
                task_query=query,
                k=k,
                model_id=model,
                tool_executor=tool_executor,
            )
            nodes_evaluated += len(children_with_values)

            candidates: list[ToTNode] = []
            for child, value in children_with_values:
                candidates.append(child)
                if value > best_value:
                    best_value = value
                    best_node = child
        else:
            # ── Sequential expansion (original) ──
            candidates = []
            for node in frontier:
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

                for thought in thoughts:
                    next_state = f"{node.state}\n\nStep {depth}: {thought}"
                    if tool_executor and ("search" in thought.lower() or "rga" in thought.lower()):
                        tool_result = tool_executor(thought)
                        next_state += f"\n[Tool result]: {tool_result[:500]}"

                    value, reason = evaluate_state(next_state, query, model_id=model)
                    nodes_evaluated += 1

                    child = ToTNode(
                        thought=thought, state=next_state, parent=node,
                        value=value, depth=depth, eval_reason=reason,
                    )
                    node.add_child(child)
                    candidates.append(child)

                    if value > best_value:
                        best_value = value
                        best_node = child

        # Beam pruning — keep top beam_width nodes
        candidates.sort(key=lambda n: n.value, reverse=True)
        frontier = candidates[:beam_width]

        if not frontier:
            break

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
