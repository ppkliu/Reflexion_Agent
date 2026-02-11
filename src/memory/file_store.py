"""FileMemoryStore — file-based memory (MEMORY.md + daily notes).

Mirrors nanobot's MemoryStore pattern:
  - MEMORY.md: Long-term accumulated insights and patterns.
  - notes/YYYY-MM-DD.md: Daily execution logs.
  - Context files (AGENTS.md, SOUL.md, USER.md): Agent persona bootstrapping.

Complements the SQLite-based EpisodicStore with human-readable files.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class FileMemoryStore:
    """File-based memory store for persistent knowledge and daily notes.

    Directory structure:
        workspace/
        ├── MEMORY.md        # Long-term key insights
        ├── AGENTS.md        # Agent definitions/personas
        ├── SOUL.md          # System-level principles
        ├── USER.md          # User preferences
        └── notes/
            ├── 2025-01-15.md
            ├── 2025-01-16.md
            └── ...

    Usage:
        store = FileMemoryStore(Path("./workspace"))
        store.append_memory("Key insight: Always check edge cases first.")
        store.write_daily_note("Ran 3 trials, best score 0.85.")
        recent = store.get_recent_notes(days=7)
    """

    MEMORY_FILE = "MEMORY.md"
    NOTES_DIR = "notes"
    CONTEXT_FILES = ("AGENTS.md", "SOUL.md", "USER.md")

    def __init__(self, workspace: Path | str | None = None) -> None:
        if workspace is None:
            workspace = Path.cwd()
        self._workspace = Path(workspace).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._notes_dir = self._workspace / self.NOTES_DIR
        self._notes_dir.mkdir(exist_ok=True)

    @property
    def workspace(self) -> Path:
        return self._workspace

    def read_memory(self) -> str:
        """Read the MEMORY.md file contents.

        Returns:
            File contents, or empty string if file doesn't exist.
        """
        path = self._workspace / self.MEMORY_FILE
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def append_memory(self, entry: str, section: str | None = None) -> None:
        """Append an entry to MEMORY.md.

        Args:
            entry: Text to append.
            section: Optional section header (e.g. "## Insights").
        """
        path = self._workspace / self.MEMORY_FILE
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines: list[str] = []
        if section:
            lines.append(f"\n## {section}\n")
        lines.append(f"- [{timestamp}] {entry}\n")

        with open(path, "a", encoding="utf-8") as f:
            f.writelines(lines)

        logger.debug("Appended to MEMORY.md: %s", entry[:60])

    def write_daily_note(self, content: str, date: datetime | None = None) -> Path:
        """Write or append to today's daily note.

        Args:
            content: Note content to append.
            date: Specific date (defaults to today UTC).

        Returns:
            Path to the daily note file.
        """
        if date is None:
            date = datetime.now(timezone.utc)

        filename = date.strftime("%Y-%m-%d") + ".md"
        path = self._notes_dir / filename
        timestamp = date.strftime("%H:%M UTC")

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n### {timestamp}\n{content}\n")

        logger.debug("Wrote daily note: %s", filename)
        return path

    def get_daily_note(self, date: datetime | None = None) -> str:
        """Read a specific day's note.

        Args:
            date: The date to read (defaults to today).

        Returns:
            Note contents, or empty string if doesn't exist.
        """
        if date is None:
            date = datetime.now(timezone.utc)

        filename = date.strftime("%Y-%m-%d") + ".md"
        path = self._notes_dir / filename

        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def get_recent_notes(self, days: int = 7) -> list[tuple[str, str]]:
        """Get notes from the last N days.

        Returns:
            List of (date_str, content) tuples, most recent first.
        """
        today = datetime.now(timezone.utc)
        results: list[tuple[str, str]] = []

        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            path = self._notes_dir / f"{date_str}.md"

            if path.exists():
                content = path.read_text(encoding="utf-8")
                results.append((date_str, content))

        return results

    def get_context_files(self) -> dict[str, str]:
        """Read all context bootstrap files (AGENTS.md, SOUL.md, USER.md).

        Returns:
            Dict mapping filename to contents. Only includes files that exist.
        """
        result: dict[str, str] = {}
        for filename in self.CONTEXT_FILES:
            path = self._workspace / filename
            if path.exists():
                result[filename] = path.read_text(encoding="utf-8")
        return result

    def write_context_file(self, filename: str, content: str) -> None:
        """Write a context bootstrap file.

        Args:
            filename: Must be one of AGENTS.md, SOUL.md, USER.md.
            content: File contents.
        """
        if filename not in self.CONTEXT_FILES:
            raise ValueError(
                f"Invalid context file: {filename}. Must be one of {self.CONTEXT_FILES}"
            )
        path = self._workspace / filename
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote context file: %s", filename)

    def build_context_prompt(self) -> str:
        """Build a combined prompt block from all context files.

        Returns:
            Formatted string with all context file contents.
        """
        ctx = self.get_context_files()
        if not ctx:
            return ""

        blocks: list[str] = []
        for filename, content in ctx.items():
            name = filename.replace(".md", "")
            blocks.append(f"## {name}\n{content.strip()}")

        return "\n\n".join(blocks)
