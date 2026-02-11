#!/usr/bin/env python3
"""Phase 3: Reflexion loop — cross-trial self-improvement."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.reflexion.loop import ReflexionRunner


def main():
    load_config()
    runner = ReflexionRunner()

    query = input("Enter your question (or press Enter for default): ").strip()
    if not query:
        query = "Write a Python function to detect a cycle in a linked list. Explain your approach."

    instruction = "The answer should include correct code and a clear explanation."

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Running Reflexion loop...")
    print(f"{'='*60}\n")

    result = runner.run(
        task_query=query,
        instruction=instruction,
        category="coding",
    )

    print(f"\n{'='*60}")
    print(f"Best score: {result['best_score']:.2f}")
    print(f"Trials used: {result['trials_used']}")
    print(f"{'='*60}")

    print("\nTrial details:")
    for t in result["trial_details"]:
        status = "PASS" if t["success"] else "FAIL"
        ref = " + reflection" if t["has_reflection"] else ""
        print(f"  Trial {t['trial']}: score={t['score']:.2f} [{status}]{ref}")

    print(f"\nBest answer:\n{result['best_answer']}")


if __name__ == "__main__":
    main()
