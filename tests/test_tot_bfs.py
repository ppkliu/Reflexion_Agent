"""Tests for ToT node and utility functions (no LLM calls)."""

from src.tot.node import ToTNode
from src.tot.utils import extract_final_answer, extract_float_score, parse_numbered_thoughts


def test_parse_numbered_thoughts():
    raw = """1. First approach: try direct calculation
2. Second approach: use algebra
3. Third approach: enumerate possibilities
4. Fourth approach: work backwards"""

    thoughts = parse_numbered_thoughts(raw, expect_k=4)
    assert len(thoughts) == 4
    assert "direct calculation" in thoughts[0]
    assert "work backwards" in thoughts[3]


def test_parse_numbered_thoughts_with_parentheses():
    raw = """1) Start by checking edge cases
2) Apply the main algorithm
3) Verify the result"""

    thoughts = parse_numbered_thoughts(raw, expect_k=3)
    assert len(thoughts) == 3


def test_extract_float_score():
    assert extract_float_score("0.85 - looks promising") == 0.85
    assert extract_float_score("Score: 0.3") == 0.3
    assert extract_float_score("1.0") == 1.0
    assert extract_float_score("The score is 7 out of 10") == 0.7
    assert extract_float_score("no number here") == 0.0


def test_extract_final_answer():
    state = "Some reasoning...\n\nFinal answer: 42"
    assert extract_final_answer(state) == "42"

    state2 = "Step 1: Think\n\nStep 2: Calculate\n\nThe result is 42."
    assert "42" in extract_final_answer(state2)


def test_tot_node_path():
    root = ToTNode(state="root", depth=0)
    child1 = ToTNode(thought="Step A", state="root+A", depth=1)
    root.add_child(child1)
    child2 = ToTNode(thought="Step B", state="root+A+B", depth=2)
    child1.add_child(child2)

    path = child2.get_path()
    assert path == ["Step A", "Step B"]


def test_tot_node_cumulative_value():
    root = ToTNode(depth=0, value=0)
    child = ToTNode(thought="a", depth=1, value=0.8)
    root.add_child(child)
    grandchild = ToTNode(thought="b", depth=2, value=0.6)
    child.add_child(grandchild)

    assert grandchild.cumulative_value == 0.7  # (0.8 + 0.6) / 2
