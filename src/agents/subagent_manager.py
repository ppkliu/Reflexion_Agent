"""SubagentManager — standardized spawn interface for parallel agent tasks.

Mirrors nanobot's SubagentManager.spawn() pattern with two execution modes:
  - "threadpool" (default): Uses concurrent.futures.ThreadPoolExecutor + litellm.
    Zero config, works everywhere, good for simple LLM call parallelism.
  - "nanobot": Uses nanobot SubagentManager.spawn() + MessageBus for full
    agent-based parallel tasks. Requires nanobot config.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import litellm

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SubagentTask:
    """Represents a spawned subagent task."""

    task_id: str
    label: str
    prompt: str
    status: str = "pending"  # pending | running | completed | failed
    result: Any | None = None
    error: str | None = None

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed")


class SubagentManager:
    """Manages spawning and tracking of parallel subagent tasks.

    Usage:
        manager = SubagentManager(model_id="openai/gpt-4o")

        # Spawn a single task
        task = manager.spawn("Analyze this text...", label="analyzer")

        # Spawn a batch
        tasks = manager.spawn_batch([
            ("Summarize document A", "sum-A"),
            ("Summarize document B", "sum-B"),
        ])

        # Wait for all to complete
        results = manager.wait_all(tasks)
    """

    def __init__(
        self,
        model_id: str | None = None,
        max_workers: int = 6,
        mode: str = "threadpool",
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> None:
        cfg = get_config()
        self._model = model_id or cfg.model.id
        self._max_workers = max_workers
        self._mode = mode
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tasks: dict[str, SubagentTask] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def spawn(self, task: str, label: str = "") -> SubagentTask:
        """Spawn a single subagent task.

        Args:
            task: The prompt/instruction for the subagent.
            label: Human-readable label for tracking.

        Returns:
            SubagentTask with status="pending" (call wait_all to execute).
        """
        task_id = f"sa-{uuid.uuid4().hex[:8]}"
        sa_task = SubagentTask(
            task_id=task_id,
            label=label or task[:30],
            prompt=task,
        )
        self._tasks[task_id] = sa_task
        logger.debug("Spawned task %s: %s", task_id, label)
        return sa_task

    def spawn_batch(self, tasks: list[tuple[str, str]]) -> list[SubagentTask]:
        """Spawn multiple tasks at once.

        Args:
            tasks: List of (prompt, label) tuples.

        Returns:
            List of SubagentTask objects.
        """
        return [self.spawn(prompt, label) for prompt, label in tasks]

    def wait_all(self, tasks: list[SubagentTask] | None = None) -> list[SubagentTask]:
        """Execute and wait for all pending tasks to complete.

        Args:
            tasks: Specific tasks to wait for. If None, waits for all pending tasks.

        Returns:
            List of completed SubagentTask objects.
        """
        if tasks is None:
            tasks = [t for t in self._tasks.values() if not t.is_done]

        pending = [t for t in tasks if not t.is_done]
        if not pending:
            return tasks

        if self._mode == "nanobot":
            return self._execute_nanobot(pending)
        return self._execute_threadpool(pending)

    def get_status(self, task_id: str) -> SubagentTask | None:
        """Get a task by its ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[SubagentTask]:
        """Get all tracked tasks."""
        return list(self._tasks.values())

    def clear(self) -> None:
        """Clear all completed tasks from tracking."""
        self._tasks = {
            tid: t for tid, t in self._tasks.items() if not t.is_done
        }

    def _execute_threadpool(self, tasks: list[SubagentTask]) -> list[SubagentTask]:
        """Execute tasks using ThreadPoolExecutor + litellm."""

        def _run_one(task: SubagentTask) -> SubagentTask:
            task.status = "running"
            try:
                resp = litellm.completion(
                    model=self._model,
                    messages=[{"role": "user", "content": task.prompt}],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                task.result = resp.choices[0].message.content or ""
                task.status = "completed"
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                logger.error("Task %s failed: %s", task.task_id, e)
            return task

        workers = min(self._max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, t): t for t in tasks}
            for future in as_completed(futures):
                future.result()  # propagate exceptions, task already updated

        return tasks

    def _execute_nanobot(self, tasks: list[SubagentTask]) -> list[SubagentTask]:
        """Execute tasks using nanobot SubagentManager."""
        try:
            from nanobot.agent.subagent import SubagentManager as NbSubagentManager
            from nanobot.bus.queue import MessageBus
            from nanobot.providers.litellm_provider import LiteLLMProvider
        except ImportError:
            logger.warning(
                "nanobot-ai not installed, falling back to threadpool mode."
            )
            return self._execute_threadpool(tasks)

        async def _run_nanobot() -> None:
            bus = MessageBus()
            provider = LiteLLMProvider()
            manager = NbSubagentManager(
                provider=provider,
                workspace=None,
                bus=bus,
                model=self._model,
                restrict_to_workspace=False,
            )

            for task in tasks:
                task.status = "running"
                await manager.spawn(task=task.prompt, label=task.label)

            collected = 0
            for task in tasks:
                try:
                    msg = await asyncio.wait_for(
                        bus.consume_outbound(), timeout=30.0
                    )
                    task.result = msg.content or ""
                    task.status = "completed"
                    collected += 1
                except asyncio.TimeoutError:
                    task.error = "Timeout waiting for subagent result"
                    task.status = "failed"

        asyncio.run(_run_nanobot())
        return tasks

    def __len__(self) -> int:
        return len(self._tasks)
