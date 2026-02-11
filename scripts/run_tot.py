#!/usr/bin/env python3
"""Phase 4: Single ToT run — BFS or DFS tree search."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.tot.bfs import run_tot_bfs
from src.tot.dfs import run_tot_dfs


def main():
    cfg = load_config()

    query = input("Enter your question (or press Enter for default): ").strip()
    if not query:
        query = "Solve: Use the numbers 1, 5, 6, 7 with +, -, *, / to make 24."

    algo = input("Search algorithm? [bfs/dfs] (default: bfs): ").strip() or "bfs"

    system_prompt = "You are a careful problem solver. Think step by step."

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Config: k={cfg.tot.branch_factor}, depth={cfg.tot.max_depth}, beam={cfg.tot.beam_width}")
    print(f"{'='*60}\n")

    if algo == "dfs":
        result = run_tot_dfs(
            query=query,
            system_prompt=system_prompt,
        )
    else:
        result = run_tot_bfs(
            query=query,
            system_prompt=system_prompt,
        )

    print(f"\n{'='*60}")
    print(f"Final answer: {result['final_answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Depth reached: {result['depth_reached']}")
    print(f"Nodes evaluated: {result['nodes_evaluated']}")
    print(f"{'='*60}")

    print("\nReasoning path:")
    for i, step in enumerate(result["path"], 1):
        print(f"  {i}. {step[:120]}...")


if __name__ == "__main__":
    main()
