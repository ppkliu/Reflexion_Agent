"""Hybrid Reflexion + ToT — cross-trial learning with tree search inner loop."""

from __future__ import annotations

import uuid
from typing import Any

import litellm

from src.agents.evaluator import create_evaluator, parse_evaluation
from src.config import get_config
from src.memory.episodic_store import EpisodicStore
from src.memory.reflection_retriever import format_reflections
from src.reflexion.prompts import EVALUATOR_PROMPT_TEMPLATE
from src.tools.rga_search import rga_search
from src.tot.bfs import run_tot_bfs
from src.tot.dfs import run_tot_dfs

TOT_REFLECTION_PROMPT = """\
A reasoning agent used Tree of Thoughts to solve this task but the result was unsatisfactory.

Task: {task_query}
Search algorithm: {search_algo}
Tree search summary: {tree_digest}
Best reasoning path:
{best_path}

Final answer: {final_answer}
Evaluation: score={score}, feedback={reason}

Past reflections used: {num_past_reflections}

Analyze what went wrong in the tree search process:
- Were the generated thoughts diverse enough?
- Was a promising branch pruned too early?
- What key insight or information was missing?
- Should the search strategy (depth/breadth/pruning) be adjusted?

Provide 3-5 specific, actionable lessons for the next attempt.
"""


class HybridReflexionToT:
    """Reflexion (outer loop) + ToT (inner reasoning engine).

    Each trial runs a full ToT search. On failure, a tree-aware reflection
    is generated and stored for future trials.
    """

    def __init__(
        self,
        store: EpisodicStore | None = None,
        model_id: str | None = None,
    ):
        self.store = store or EpisodicStore()
        cfg = get_config()
        self.model_id = model_id or cfg.model.id
        self.evaluator = create_evaluator()

    def run(
        self,
        task_query: str,
        instruction: str = "Answer the question accurately and completely.",
        category: str = "general",
        max_trials: int | None = None,
        min_score: float | None = None,
        search_algo: str | None = None,
    ) -> dict[str, Any]:
        """Execute the hybrid Reflexion + ToT loop.

        Args:
            task_query: The question/task to solve.
            instruction: Evaluation guidance.
            category: Task category for memory grouping.
            max_trials: Maximum reflexion trials.
            min_score: Minimum score for early stopping.
            search_algo: 'bfs' or 'dfs'. Defaults to config.

        Returns:
            {best_answer, best_score, trials_used, trial_details}
        """
        cfg = get_config()
        max_trials = max_trials or cfg.reflexion.max_trials
        min_score = min_score or cfg.reflexion.min_success_score
        search_algo = search_algo or cfg.tot.search_algo

        task_key = task_query[:80]
        best_answer: str | None = None
        best_score: float = -1.0
        trial_details: list[dict[str, Any]] = []

        # rga tool executor for ToT nodes
        def tool_exec(thought: str) -> str:
            keywords = thought.split()[:5]
            query = " ".join(keywords)
            return rga_search(query=query)

        for trial_no in range(1, max_trials + 1):
            trial_id = f"hybrid-t{trial_no}-{uuid.uuid4().hex[:6]}"

            # 1. Retrieve past reflections
            past = self.store.get_relevant_reflections(
                task_key=task_key, category=category,
            )
            reflection_block = format_reflections(past)

            system_prompt = f"""{instruction}

{reflection_block}

Use step-by-step reasoning. Be thorough and consider multiple approaches.
"""

            # 2. Run ToT (BFS or DFS)
            if search_algo == "dfs":
                tot_result = run_tot_dfs(
                    query=task_query,
                    system_prompt=system_prompt,
                    model_id=self.model_id,
                    tool_executor=tool_exec,
                )
            else:
                tot_result = run_tot_bfs(
                    query=task_query,
                    system_prompt=system_prompt,
                    model_id=self.model_id,
                    tool_executor=tool_exec,
                )

            final_answer = tot_result["final_answer"]
            tree_digest = tot_result["tree_digest"]

            # 3. Evaluate
            eval_prompt = EVALUATOR_PROMPT_TEMPLATE.format(
                task_query=task_query,
                answer=final_answer,
                instruction=instruction,
            )
            eval_response = self.evaluator.run(eval_prompt)
            evaluation = parse_evaluation(eval_response.content or "")
            score = evaluation["score"]

            # 4. Reflect on tree search (on failure)
            reflection: str | None = None
            if not evaluation["success"] and score < min_score:
                best_path_str = "\n".join(
                    f"  {i+1}. {s}" for i, s in enumerate(tot_result["path"])
                )
                reflect_prompt = TOT_REFLECTION_PROMPT.format(
                    task_query=task_query,
                    search_algo=search_algo,
                    tree_digest=tree_digest,
                    best_path=best_path_str,
                    final_answer=final_answer,
                    score=score,
                    reason=evaluation["reason"],
                    num_past_reflections=len(past),
                )
                reflect_response = litellm.completion(
                    model=self.model_id,
                    messages=[{"role": "user", "content": reflect_prompt}],
                    max_tokens=600,
                    temperature=0.3,
                )
                reflection = reflect_response.choices[0].message.content or ""

            # 5. Store trial
            self.store.save_trial(
                category=category,
                task_key=task_key,
                trial_id=trial_id,
                query=task_query,
                trajectory_digest=tree_digest,
                final_answer=final_answer,
                score=score,
                reflection=reflection,
                used_reflections=len(past),
            )

            if score > best_score:
                best_score = score
                best_answer = final_answer

            trial_details.append({
                "trial": trial_no,
                "trial_id": trial_id,
                "score": score,
                "success": evaluation["success"],
                "search_algo": search_algo,
                "nodes_evaluated": tot_result["nodes_evaluated"],
                "depth_reached": tot_result["depth_reached"],
                "has_reflection": reflection is not None,
            })

            if evaluation["success"] or score >= min_score:
                break

        return {
            "best_answer": best_answer,
            "best_score": best_score,
            "trials_used": trial_no,
            "trial_details": trial_details,
        }
