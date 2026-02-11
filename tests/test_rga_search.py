"""Tests for rga_search tool (requires rga installed for integration tests)."""

import subprocess

import pytest

from src.tools.rga_search import rga_search


def rga_available() -> bool:
    try:
        subprocess.run(["rga", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not rga_available(), reason="rga not installed")
def test_rga_search_no_matches():
    result = rga_search(query="xyznonexistent12345", root_dir="./knowledge")
    assert "No matches" in result or "not found" in result.lower()


@pytest.mark.skipif(not rga_available(), reason="rga not installed")
def test_rga_search_knowledge_dir_missing():
    result = rga_search(query="test", root_dir="/tmp/nonexistent_dir_12345")
    assert "not found" in result.lower() or "error" in result.lower()


def test_rga_search_not_installed(monkeypatch):
    """Simulate rga not being installed."""
    def fake_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = rga_search(query="test")
    assert "not installed" in result.lower()
