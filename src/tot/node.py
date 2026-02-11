"""Tree of Thoughts node data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToTNode:
    """A single node in the Tree of Thoughts."""

    thought: str = ""
    state: str = ""
    parent: Optional[ToTNode] = field(default=None, repr=False)
    value: float = 0.0
    depth: int = 0
    children: list[ToTNode] = field(default_factory=list, repr=False)
    eval_reason: str = ""
    pruned: bool = False

    def add_child(self, child: ToTNode) -> None:
        child.parent = self
        self.children.append(child)

    def get_path(self) -> list[str]:
        """Reconstruct the reasoning path from root to this node."""
        path: list[str] = []
        current: ToTNode | None = self
        while current is not None:
            if current.thought:
                path.append(current.thought)
            current = current.parent
        path.reverse()
        return path

    def get_path_str(self) -> str:
        """Get a formatted string of the reasoning path."""
        steps = self.get_path()
        if not steps:
            return "(root)"
        return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))

    @property
    def cumulative_value(self) -> float:
        """Average value along the path from root to this node."""
        values: list[float] = []
        current: ToTNode | None = self
        while current is not None:
            if current.depth > 0:
                values.append(current.value)
            current = current.parent
        return sum(values) / len(values) if values else 0.0
