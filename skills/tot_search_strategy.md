---
name: tot_search_strategy
description: Strategy guidelines for Tree of Thoughts search exploration
always_on: false
tags: [tot, reasoning, search]
---
When generating reasoning branches in Tree of Thoughts:

1. **Maximize diversity** — Each of the k branches should explore a genuinely different angle. Avoid generating k variations of the same idea.
2. **Balance depth vs breadth** — In BFS, keep beam_width tight (3-5) to avoid wasting evaluations on weak branches. In DFS, prune aggressively below threshold.
3. **Use tools strategically** — If a branch involves factual claims, trigger a knowledge search (rga_search) to ground the reasoning in evidence.
4. **Evaluate honestly** — When scoring a branch, consider:
   - Does it make progress toward the answer? (score > 0.5)
   - Is it logically consistent with previous steps? (no contradictions)
   - Does it introduce useful new information? (not just restating the question)
5. **Know when to backtrack** — A score below 0.3 means the branch is unlikely to recover. Prune and try a different direction.
