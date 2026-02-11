"""ToolRegistry — centralized tool registration, discovery, and execution.

Mirrors nanobot's ToolRegistry pattern:
  - Tools are registered with name, description, parameter schema, and a callable.
  - The registry provides get/list/execute operations.
  - Tools can be registered via decorator or explicit call.
  - Supports both sync and async callables.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    callable: Callable[..., Any]
    is_async: bool = False
    tags: list[str] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        """Return OpenAI-compatible function/tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Centralized registry for all tools available to agents.

    Usage:
        registry = ToolRegistry()

        # Register via decorator
        @registry.register(
            name="rga_search",
            description="Search documents via ripgrep-all.",
            parameters={...},
        )
        def rga_search(query: str) -> str:
            ...

        # Register explicitly
        registry.add(ToolSpec(name="eval", ...))

        # Execute
        result = registry.execute("rga_search", query="transformer")

        # List all for LLM tool-use
        schemas = registry.get_schemas()
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> None:
        """Register a tool specification."""
        if spec.name in self._tools:
            logger.warning("Overwriting existing tool: %s", spec.name)
        self._tools[spec.name] = spec
        logger.debug("Registered tool: %s", spec.name)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a callable as a tool.

        Args:
            name: Tool name (must be unique).
            description: Human-readable description.
            parameters: JSON Schema for the tool's parameters.
            tags: Optional tags for filtering (e.g. ["search", "knowledge"]).

        Returns:
            Decorator that registers and returns the original function.
        """
        def decorator(fn: Callable) -> Callable:
            spec = ToolSpec(
                name=name,
                description=description,
                parameters=parameters or _infer_parameters(fn),
                callable=fn,
                is_async=asyncio.iscoroutinefunction(fn),
                tags=tags or [],
            )
            self.add(spec)
            return fn
        return decorator

    def get(self, name: str) -> ToolSpec | None:
        """Get a tool spec by name."""
        return self._tools.get(name)

    def list_tools(self, tags: Sequence[str] | None = None) -> list[ToolSpec]:
        """List all registered tools, optionally filtered by tags."""
        tools = list(self._tools.values())
        if tags:
            tag_set = set(tags)
            tools = [t for t in tools if tag_set & set(t.tags)]
        return tools

    def get_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self, tags: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [t.to_schema() for t in self.list_tools(tags=tags)]

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name (sync).

        Raises:
            KeyError: If tool not found.
            RuntimeError: If tool is async (use execute_async instead).
        """
        spec = self._tools.get(tool_name)
        if spec is None:
            raise KeyError(f"Tool not found: {tool_name}")

        if spec.is_async:
            # Run async callable in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, spec.callable(**kwargs)).result()
            return asyncio.run(spec.callable(**kwargs))

        return spec.callable(**kwargs)

    async def execute_async(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool by name (async).

        Raises:
            KeyError: If tool not found.
        """
        spec = self._tools.get(tool_name)
        if spec is None:
            raise KeyError(f"Tool not found: {tool_name}")

        if spec.is_async:
            return await spec.callable(**kwargs)
        # Run sync callable in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: spec.callable(**kwargs))

    def remove(self, name: str) -> bool:
        """Unregister a tool. Returns True if it existed."""
        if name in self._tools:
            del self._tools[name]
            logger.debug("Removed tool: %s", name)
            return True
        return False

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def _infer_parameters(fn: Callable) -> dict[str, Any]:
    """Infer JSON Schema parameters from function signature."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for name, param in sig.parameters.items():
        if name in ("self", "cls", "kwargs"):
            continue

        annotation = param.annotation
        json_type = type_map.get(annotation, "string")
        prop: dict[str, Any] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

        properties[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


# ── Global registry singleton ──

_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry (creates if needed)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """Module-level decorator to register a tool in the global registry."""
    return get_registry().register(name, description, parameters, tags)


def register_default_tools() -> ToolRegistry:
    """Register built-in tools into the global registry.

    Registers:
      - rga_search: Document search via ripgrep-all.
      - evaluate_state: ToT state evaluator via LLM.

    Returns the global registry with tools registered.
    """
    registry = get_registry()

    if "rga_search" not in registry:
        from src.tools.rga_search import rga_search

        registry.add(ToolSpec(
            name="rga_search",
            description=(
                "Search local documents (pdf, docx, md, code, etc.) using ripgrep-all. "
                "Returns matching passages with file name, line number, and context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or keyword to search for."},
                    "file_pattern": {"type": "string", "description": "Optional glob filter, e.g. '*.pdf'."},
                },
                "required": ["query"],
            },
            callable=rga_search,
            tags=["search", "knowledge"],
        ))

    if "evaluate_state" not in registry:
        from src.tot.evaluator import evaluate_state

        registry.add(ToolSpec(
            name="evaluate_state",
            description="Evaluate how promising a ToT reasoning state is (0.0-1.0).",
            parameters={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "The reasoning state to evaluate."},
                    "task_description": {"type": "string", "description": "The original task."},
                    "model_id": {"type": "string", "description": "LLM model identifier."},
                },
                "required": ["state", "task_description"],
            },
            callable=evaluate_state,
            tags=["evaluation", "tot"],
        ))

    return registry
