"""Tests for FileMemoryStore — file-based memory (MEMORY.md + daily notes)."""

from datetime import datetime, timedelta, timezone

import pytest

from src.memory.file_store import FileMemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a FileMemoryStore in a temp directory."""
    return FileMemoryStore(workspace=tmp_path)


class TestFileMemoryStore:

    def test_read_empty_memory(self, store):
        """read_memory returns empty string when MEMORY.md doesn't exist."""
        assert store.read_memory() == ""

    def test_append_and_read_memory(self, store):
        """append_memory writes to MEMORY.md, read_memory reads it back."""
        store.append_memory("Insight 1: Always verify inputs.")
        store.append_memory("Insight 2: Use structured prompts.")

        content = store.read_memory()
        assert "Insight 1: Always verify inputs." in content
        assert "Insight 2: Use structured prompts." in content

    def test_append_memory_with_section(self, store):
        """append_memory with section adds a markdown header."""
        store.append_memory("Check edge cases first.", section="Lessons Learned")

        content = store.read_memory()
        assert "## Lessons Learned" in content
        assert "Check edge cases first." in content

    def test_write_and_read_daily_note(self, store):
        """write_daily_note creates a date-stamped file."""
        date = datetime(2025, 6, 15, 10, 30, tzinfo=timezone.utc)
        path = store.write_daily_note("Ran 3 trials, best score 0.85.", date=date)

        assert path.name == "2025-06-15.md"
        assert path.exists()

        content = store.get_daily_note(date=date)
        assert "Ran 3 trials, best score 0.85." in content
        assert "10:30 UTC" in content

    def test_get_daily_note_nonexistent(self, store):
        """get_daily_note returns empty string for dates without notes."""
        date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert store.get_daily_note(date=date) == ""

    def test_get_recent_notes(self, store):
        """get_recent_notes returns notes from last N days."""
        today = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)
        two_days_ago = today - timedelta(days=2)

        store.write_daily_note("Today's note.", date=today)
        store.write_daily_note("Yesterday's note.", date=yesterday)
        store.write_daily_note("Two days ago.", date=two_days_ago)

        recent = store.get_recent_notes(days=3)
        assert len(recent) == 3
        # Most recent first
        assert "Today's note." in recent[0][1]

    def test_get_recent_notes_sparse(self, store):
        """get_recent_notes only returns days that have notes."""
        today = datetime.now(timezone.utc)
        store.write_daily_note("Only today.", date=today)

        recent = store.get_recent_notes(days=7)
        assert len(recent) == 1

    def test_context_files_empty(self, store):
        """get_context_files returns empty dict when no files exist."""
        assert store.get_context_files() == {}

    def test_write_and_read_context_files(self, store):
        """write_context_file creates context files, get_context_files reads them."""
        store.write_context_file("AGENTS.md", "# Agents\nActor, Evaluator, Reflector.")
        store.write_context_file("SOUL.md", "# Principles\nBe accurate and helpful.")

        ctx = store.get_context_files()
        assert "AGENTS.md" in ctx
        assert "SOUL.md" in ctx
        assert "USER.md" not in ctx  # Not written
        assert "Actor, Evaluator, Reflector." in ctx["AGENTS.md"]

    def test_write_invalid_context_file(self, store):
        """write_context_file rejects unknown filenames."""
        with pytest.raises(ValueError, match="Invalid context file"):
            store.write_context_file("RANDOM.md", "content")

    def test_build_context_prompt(self, store):
        """build_context_prompt combines all context files."""
        store.write_context_file("AGENTS.md", "Actor performs tasks.")
        store.write_context_file("USER.md", "Prefers concise output.")

        prompt = store.build_context_prompt()
        assert "## AGENTS" in prompt
        assert "## USER" in prompt
        assert "Actor performs tasks." in prompt

    def test_build_context_prompt_empty(self, store):
        """build_context_prompt returns empty string when no files exist."""
        assert store.build_context_prompt() == ""

    def test_workspace_created(self, tmp_path):
        """FileMemoryStore creates workspace and notes directories."""
        workspace = tmp_path / "new_workspace"
        store = FileMemoryStore(workspace=workspace)

        assert workspace.exists()
        assert (workspace / "notes").exists()
