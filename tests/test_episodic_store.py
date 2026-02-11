"""Tests for EpisodicStore."""

import tempfile
from pathlib import Path

import pytest

from src.memory.episodic_store import EpisodicStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    # Override config by passing db_path directly
    return EpisodicStore(db_path=db_path)


def test_save_and_retrieve(store):
    store.save_trial(
        category="coding",
        task_key="detect cycle linked list",
        trial_id="t1-abc",
        query="Write a function to detect cycle in linked list",
        trajectory_digest="Used two pointers",
        final_answer="def has_cycle(head): ...",
        score=0.6,
        reflection="Should have used Floyd's algorithm from the start.",
        used_reflections=0,
    )

    results = store.get_relevant_reflections(task_key="detect cycle", category="coding")
    assert len(results) == 1
    assert "Floyd" in results[0]["reflection"]
    assert results[0]["score"] == 0.6


def test_multiple_trials_ordering(store):
    for i in range(5):
        store.save_trial(
            category="math",
            task_key="game of 24",
            trial_id=f"t{i}",
            query="Use 1,5,6,7 to make 24",
            score=i * 0.2,
            reflection=f"Lesson {i}: try multiplication first" if i > 0 else None,
        )

    results = store.get_relevant_reflections(task_key="game of 24", top_k=3)
    # Should get most recent first, only non-null reflections
    assert len(results) <= 3
    for r in results:
        assert r["reflection"] is not None


def test_empty_retrieval(store):
    results = store.get_relevant_reflections(task_key="nonexistent task")
    assert results == []


def test_get_all_trials(store):
    store.save_trial(
        category="test", task_key="key1", trial_id="t1",
        query="q1", score=0.5,
    )
    store.save_trial(
        category="test", task_key="key2", trial_id="t2",
        query="q2", score=0.8,
    )

    all_trials = store.get_all_trials(category="test")
    assert len(all_trials) == 2
