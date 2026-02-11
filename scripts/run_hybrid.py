#!/usr/bin/env python3
"""Phase 5: Hybrid Reflexion + ToT — cross-trial learning with tree search."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.hybrid.reflexion_tot import HybridReflexionToT


def main():
    load_config()
    hybrid = HybridReflexionToT()

    query = input("Enter your question (or press Enter for default): ").strip()
    if not query:
        query = "Design a Python class for an LRU cache with O(1) get and put operations."

    algo = input("Search algorithm? [bfs/dfs] (default: bfs): ").strip() or "bfs"
    instruction = "The answer should be correct, efficient, and well-explained."

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Mode: Hybrid Reflexion + ToT ({algo.upper()})")
    print(f"{'='*60}\n")

    result = hybrid.run(
        task_query=query,
        instruction=instruction,
        category="coding",
        search_algo=algo,
    )

    print(f"\n{'='*60}")
    print(f"Best score: {result['best_score']:.2f}")
    print(f"Trials used: {result['trials_used']}")
    print(f"{'='*60}")

    print("\nTrial details:")
    for t in result["trial_details"]:
        status = "PASS" if t["success"] else "FAIL"
        ref = " + reflection" if t["has_reflection"] else ""
        print(
            f"  Trial {t['trial']}: score={t['score']:.2f} [{status}] "
            f"algo={t['search_algo']} nodes={t['nodes_evaluated']} "
            f"depth={t['depth_reached']}{ref}"
        )

    print(f"\nBest answer:\n{result['best_answer']}")


if __name__ == "__main__":
    main()
