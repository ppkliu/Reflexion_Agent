#!/usr/bin/env python3
"""Phase 1: ReAct baseline — single agent with rga_search, no reflexion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.actor import create_actor
from src.config import load_config


def main():
    load_config()
    actor = create_actor()

    query = input("Enter your question (or press Enter for default): ").strip()
    if not query:
        query = "What are the main differences between Reflexion and Tree of Thoughts?"

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    response = actor.run(query)
    print("Answer:")
    print(response.content)


if __name__ == "__main__":
    main()
