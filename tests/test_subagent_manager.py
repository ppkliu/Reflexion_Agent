"""Tests for SubagentManager — standardized spawn interface."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.subagent_manager import SubagentManager, SubagentTask


@pytest.fixture
def mock_config():
    """Mock get_config to avoid needing config.yaml."""
    cfg = MagicMock()
    cfg.model.id = "test/model"
    with patch("src.agents.subagent_manager.get_config", return_value=cfg):
        yield cfg


class TestSubagentTask:

    def test_initial_state(self):
        task = SubagentTask(
            task_id="sa-abc",
            label="test",
            prompt="Do something",
        )
        assert task.status == "pending"
        assert task.result is None
        assert task.is_done is False

    def test_completed_state(self):
        task = SubagentTask(
            task_id="sa-abc",
            label="test",
            prompt="Do something",
            status="completed",
            result="Done!",
        )
        assert task.is_done is True

    def test_failed_state(self):
        task = SubagentTask(
            task_id="sa-abc",
            label="test",
            prompt="Do something",
            status="failed",
            error="Connection error",
        )
        assert task.is_done is True


class TestSubagentManager:

    def test_spawn_creates_task(self, mock_config):
        """spawn creates a SubagentTask with pending status."""
        mgr = SubagentManager()
        task = mgr.spawn("Analyze this", label="analyzer")

        assert task.status == "pending"
        assert task.label == "analyzer"
        assert task.prompt == "Analyze this"
        assert task.task_id.startswith("sa-")
        assert len(mgr) == 1

    def test_spawn_batch(self, mock_config):
        """spawn_batch creates multiple tasks."""
        mgr = SubagentManager()
        tasks = mgr.spawn_batch([
            ("Task A", "label-a"),
            ("Task B", "label-b"),
            ("Task C", "label-c"),
        ])

        assert len(tasks) == 3
        assert len(mgr) == 3
        assert all(t.status == "pending" for t in tasks)

    def test_get_status(self, mock_config):
        """get_status retrieves a task by ID."""
        mgr = SubagentManager()
        task = mgr.spawn("Test", label="test")

        retrieved = mgr.get_status(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

        assert mgr.get_status("nonexistent") is None

    @patch("src.agents.subagent_manager.litellm")
    def test_wait_all_threadpool(self, mock_litellm, mock_config):
        """wait_all executes tasks via threadpool and returns completed."""
        # Mock litellm.completion response
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Result text"
        mock_litellm.completion.return_value = mock_resp

        mgr = SubagentManager(mode="threadpool")
        tasks = mgr.spawn_batch([
            ("Prompt 1", "task-1"),
            ("Prompt 2", "task-2"),
        ])

        results = mgr.wait_all(tasks)

        assert len(results) == 2
        assert all(t.status == "completed" for t in results)
        assert all(t.result == "Result text" for t in results)
        assert mock_litellm.completion.call_count == 2

    @patch("src.agents.subagent_manager.litellm")
    def test_wait_all_handles_errors(self, mock_litellm, mock_config):
        """wait_all marks failed tasks correctly."""
        mock_litellm.completion.side_effect = RuntimeError("API down")

        mgr = SubagentManager(mode="threadpool")
        task = mgr.spawn("Will fail", label="fail-test")
        results = mgr.wait_all([task])

        assert results[0].status == "failed"
        assert "API down" in results[0].error

    def test_clear_removes_done(self, mock_config):
        """clear removes completed/failed tasks from tracking."""
        mgr = SubagentManager()
        t1 = mgr.spawn("A", "a")
        t2 = mgr.spawn("B", "b")
        t1.status = "completed"
        t2.status = "pending"

        mgr.clear()
        assert len(mgr) == 1
        assert mgr.get_status(t2.task_id) is not None

    def test_wait_all_skips_done(self, mock_config):
        """wait_all skips already-completed tasks."""
        mgr = SubagentManager()
        task = mgr.spawn("Done", label="done")
        task.status = "completed"
        task.result = "Already done"

        results = mgr.wait_all([task])
        assert results[0].result == "Already done"
