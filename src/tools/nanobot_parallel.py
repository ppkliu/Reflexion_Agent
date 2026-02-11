"""Parallel execution for ToT node expansion and knowledge retrieval.

Two modes:
  - "threadpool" (default): Uses concurrent.futures.ThreadPoolExecutor.
    Zero config, works everywhere, parallelizes litellm + rga calls.
  - "nanobot": Uses nanobot SubagentManager.spawn() for full agent-based
    parallel tasks. Requires nanobot config and running MessageBus.

The threadpool mode is recommended for most use cases since each "task"
(generate thoughts + evaluate state) is a simple LLM call, not a full
agent loop. The nanobot mode is for when you need tool-using subagents.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

import litellm

from src.config import get_config
from src.tools.rga_search import rga_search
from src.tot.node import ToTNode
from src.tot.utils import extract_float_score, parse_numbered_thoughts

# ── Threadpool-based parallel execution (default) ──

_DEFAULT_MAX_WORKERS = 6


def parallel_llm_calls(
    prompts: list[str],
    model_id: str | None = None,
    max_tokens: int = 1500,
    temperature: float = 0.7,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[str]:
    """Execute multiple litellm completion calls in parallel.

    Args:
        prompts: List of user-message prompts.
        model_id: litellm model identifier.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.
        max_workers: Thread pool size.

    Returns:
        List of response content strings (same order as prompts).
    """
    cfg = get_config()
    model = model_id or cfg.model.id
    results: list[str | None] = [None] * len(prompts)

    def _call(idx: int, prompt: str) -> tuple[int, str]:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return idx, resp.choices[0].message.content or ""

    with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as pool:
        futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx, content = future.result()
            results[idx] = content

    return [r or "" for r in results]


def parallel_rga_search(
    queries: list[str],
    root_dir: str | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[str]:
    """Execute multiple rga searches in parallel.

    Args:
        queries: List of search queries.
        root_dir: Knowledge directory (uses config default if None).
        max_workers: Thread pool size.

    Returns:
        List of search results (same order as queries).
    """
    results: list[str | None] = [None] * len(queries)

    def _search(idx: int, query: str) -> tuple[int, str]:
        return idx, rga_search(query=query, root_dir=root_dir)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(queries))) as pool:
        futures = {pool.submit(_search, i, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            idx, content = future.result()
            results[idx] = content

    return [r or "" for r in results]


def parallel_expand_node(
    node: ToTNode,
    system_prompt: str,
    task_query: str,
    k: int,
    model_id: str | None = None,
    tool_executor: Callable[[str], str] | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[tuple[ToTNode, float]]:
    """Expand a single node: generate k thoughts, then evaluate all in parallel.

    Args:
        node: Parent node to expand.
        system_prompt: System prompt for generation.
        task_query: Original task description.
        k: Number of thoughts to generate.
        model_id: litellm model.
        tool_executor: Optional tool for search augmentation.
        max_workers: Thread pool size.

    Returns:
        List of (child_node, value) tuples.
    """
    cfg = get_config()
    model = model_id or cfg.model.id

    # Step 1: Generate k thoughts (single LLM call)
    gen_prompt = f"""{system_prompt}

Current reasoning chain:
{node.state[-1500:]}

Generate {k} distinct, creative next reasoning steps. Each should be a short paragraph (2-5 sentences).
Number them 1 to {k}. Explore different angles."""

    gen_response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": gen_prompt}],
        max_tokens=1500,
        temperature=0.7,
    )
    raw = gen_response.choices[0].message.content or ""
    thoughts = parse_numbered_thoughts(raw, expect_k=k)

    if not thoughts:
        return []

    # Build next states
    depth = node.depth + 1
    next_states: list[str] = []
    for thought in thoughts:
        state = f"{node.state}\n\nStep {depth}: {thought}"
        if tool_executor and ("search" in thought.lower() or "rga" in thought.lower()):
            tool_result = tool_executor(thought)
            state += f"\n[Tool result]: {tool_result[:500]}"
        next_states.append(state)

    # Step 2: Evaluate all states in parallel
    eval_prompts = []
    for state in next_states:
        eval_prompts.append(
            f"Task: {task_query}\n\nCurrent reasoning path:\n{state[:1500]}\n\n"
            "How promising is this reasoning path? "
            "Output ONLY a number 0.0-1.0, then one sentence."
        )

    eval_results = parallel_llm_calls(
        prompts=eval_prompts,
        model_id=model,
        max_tokens=150,
        temperature=0.1,
        max_workers=max_workers,
    )

    # Build child nodes
    children: list[tuple[ToTNode, float]] = []
    for thought, state, eval_raw in zip(thoughts, next_states, eval_results):
        score = extract_float_score(eval_raw)
        child = ToTNode(
            thought=thought,
            state=state,
            parent=node,
            value=score,
            depth=depth,
            eval_reason=eval_raw.strip(),
        )
        node.add_child(child)
        children.append((child, score))

    return children


def parallel_expand_frontier(
    frontier: list[ToTNode],
    system_prompt: str,
    task_query: str,
    k: int,
    model_id: str | None = None,
    tool_executor: Callable[[str], str] | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[tuple[ToTNode, float]]:
    """Expand all frontier nodes in parallel (each node generates + evaluates).

    This is the main entry point for parallel BFS expansion.
    Each frontier node is expanded in its own thread.

    Returns:
        Flat list of all (child_node, value) tuples across all frontier nodes.
    """
    if len(frontier) == 1:
        # Single node — no need for outer parallelism, just parallelize eval
        return parallel_expand_node(
            frontier[0], system_prompt, task_query, k,
            model_id, tool_executor, max_workers,
        )

    all_children: list[tuple[ToTNode, float]] = []

    def _expand_one(node: ToTNode) -> list[tuple[ToTNode, float]]:
        return parallel_expand_node(
            node, system_prompt, task_query, k,
            model_id, tool_executor,
            max_workers=max(2, max_workers // len(frontier)),
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(frontier))) as pool:
        futures = {pool.submit(_expand_one, n): n for n in frontier}
        for future in as_completed(futures):
            children = future.result()
            all_children.extend(children)

    return all_children


# ── SubagentManager-based parallel execution ──


def subagent_expand_frontier(
    frontier: list[ToTNode],
    system_prompt: str,
    task_query: str,
    k: int,
    model_id: str | None = None,
    tool_executor: Callable[[str], str] | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> list[tuple[ToTNode, float]]:
    """Expand frontier using SubagentManager (spawn one task per node).

    This wraps the SubagentManager interface over the existing
    parallel_expand_node logic, providing standardized task tracking.

    Returns:
        Flat list of all (child_node, value) tuples.
    """
    from src.agents.subagent_manager import SubagentManager

    cfg = get_config()
    model = model_id or cfg.model.id

    # For each frontier node, spawn a subagent that generates + evaluates
    manager = SubagentManager(model_id=model, max_workers=max_workers)
    tasks = manager.spawn_batch([
        (
            f"{system_prompt}\n\nCurrent reasoning chain:\n{node.state[-1500:]}\n\n"
            f"Generate {k} distinct next reasoning steps for the task: {task_query}\n"
            f"Number them 1 to {k}.",
            f"expand-d{node.depth}-{i}",
        )
        for i, node in enumerate(frontier)
    ])

    manager.wait_all(tasks)

    # Parse generated thoughts and evaluate
    all_children: list[tuple[ToTNode, float]] = []
    for node, task in zip(frontier, tasks):
        if task.status != "completed" or not task.result:
            continue

        thoughts = parse_numbered_thoughts(task.result, expect_k=k)
        if not thoughts:
            continue

        depth = node.depth + 1
        next_states: list[str] = []
        for thought in thoughts:
            state = f"{node.state}\n\nStep {depth}: {thought}"
            if tool_executor and ("search" in thought.lower() or "rga" in thought.lower()):
                tool_result = tool_executor(thought)
                state += f"\n[Tool result]: {tool_result[:500]}"
            next_states.append(state)

        # Evaluate all states in parallel via SubagentManager
        eval_manager = SubagentManager(
            model_id=model, max_workers=max_workers, max_tokens=150, temperature=0.1,
        )
        eval_tasks = eval_manager.spawn_batch([
            (
                f"Task: {task_query}\n\nCurrent reasoning path:\n{state[:1500]}\n\n"
                "How promising is this reasoning path? "
                "Output ONLY a number 0.0-1.0, then one sentence.",
                f"eval-d{depth}-{j}",
            )
            for j, state in enumerate(next_states)
        ])
        eval_manager.wait_all(eval_tasks)

        for thought, state, eval_task in zip(thoughts, next_states, eval_tasks):
            eval_raw = eval_task.result or ""
            score = extract_float_score(eval_raw)
            child = ToTNode(
                thought=thought,
                state=state,
                parent=node,
                value=score,
                depth=depth,
                eval_reason=eval_raw.strip(),
            )
            node.add_child(child)
            all_children.append((child, score))

    return all_children


# ── nanobot SubagentManager-based parallel execution (optional) ──

async def nanobot_parallel_search(
    queries: list[str],
    workspace_path: str | None = None,
) -> list[str]:
    """Use nanobot SubagentManager to run parallel rga searches.

    Each query is dispatched as a background subagent task.
    Requires nanobot to be properly configured with a running MessageBus.
    """
    try:
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.litellm_provider import LiteLLMProvider
    except ImportError:
        raise ImportError(
            "nanobot-ai is required for nanobot_parallel_search. "
            "Falling back to parallel_rga_search() is recommended."
        )

    from pathlib import Path

    cfg = get_config()
    workspace = Path(workspace_path or ".").resolve()

    bus = MessageBus()
    provider = LiteLLMProvider()

    manager = SubagentManager(
        provider=provider,
        workspace=workspace,
        bus=bus,
        model=cfg.model.id,
        restrict_to_workspace=False,
    )

    # Spawn one subagent per query
    for query in queries:
        task_prompt = (
            f"Search the knowledge base for: {query}\n"
            f"Use the rga_search tool with query='{query}'. "
            f"Return the raw search results."
        )
        await manager.spawn(task=task_prompt, label=f"rga:{query[:30]}")

    # Collect results from message bus
    results: list[str] = []
    collected = 0
    while collected < len(queries):
        try:
            msg = await asyncio.wait_for(bus.consume_outbound(), timeout=30.0)
            results.append(msg.content or "")
            collected += 1
        except asyncio.TimeoutError:
            results.append(f"Timeout waiting for subagent result (collected {collected}/{len(queries)})")
            break

    # Pad if some results missing
    while len(results) < len(queries):
        results.append("No result (subagent timeout)")

    return results
