"""Reflexion loop — cross-trial self-improvement via verbal reflection."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.agents.actor import create_actor
from src.agents.evaluator import create_evaluator, parse_evaluation
from src.agents.reflector import create_reflector
from src.config import get_config
from src.memory.episodic_store import EpisodicStore
from src.memory.reflection_retriever import format_reflections
from src.reflexion.prompts import (
    ACTOR_SYSTEM_TEMPLATE,
    EVALUATOR_PROMPT_TEMPLATE,
    REFLECTOR_PROMPT_TEMPLATE,
)
from src.skills.loader import SkillsLoader


class ReflexionRunner:
    """Runs a Reflexion loop: retrieve → act → evaluate → reflect → store → repeat."""

    def __init__(
        self,
        store: EpisodicStore | None = None,
        actor_model: str | None = None,
        evaluator_model: str | None = None,
        reflector_model: str | None = None,
        skills_dir: Path | str | None = None,
    ):
        self.store = store or EpisodicStore()
        self.actor = create_actor(model_id=actor_model)
        self.evaluator = create_evaluator(model_id=evaluator_model)
        self.reflector = create_reflector(model_id=reflector_model)

        # Load skills — always_on skills are injected into actor prompts
        self._skills_loader = SkillsLoader(skills_dir)
        self._skills_loader.load_all()
        self._skills_prompt = self._skills_loader.build_always_on_prompt()

    def run(
        self,
        task_query: str,
        instruction: str = "Answer the question accurately and completely.",
        category: str = "general",
        max_trials: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Execute the Reflexion loop.

        Returns:
            {best_answer, best_score, trials_used, trial_details}
        """
        cfg = get_config().reflexion
        max_trials = max_trials or cfg.max_trials
        min_score = min_score or cfg.min_success_score

        task_key = task_query[:80]
        best_answer: str | None = None
        best_score: float = -1.0
        trial_details: list[dict[str, Any]] = []

        for trial_no in range(1, max_trials + 1):
            trial_id = f"t{trial_no}-{uuid.uuid4().hex[:6]}"

            # 1. Retrieve past reflections
            past = self.store.get_relevant_reflections(
                task_key=task_key,
                category=category,
            )
            reflection_block = format_reflections(past)

            # 2. Build actor prompt (inject always-on skills + reflections)
            actor_prompt = ACTOR_SYSTEM_TEMPLATE.format(
                reflection_block=reflection_block,
                task_query=task_query,
            )
            if self._skills_prompt:
                actor_prompt = f"{self._skills_prompt}\n\n{actor_prompt}"
            actor_response = self.actor.run(actor_prompt)
            answer = actor_response.content or ""

            # 3. Evaluate
            eval_prompt = EVALUATOR_PROMPT_TEMPLATE.format(
                task_query=task_query,
                answer=answer,
                instruction=instruction,
            )
            eval_response = self.evaluator.run(eval_prompt)
            evaluation = parse_evaluation(eval_response.content or "")
            score = evaluation["score"]

            # 4. Reflect (only on failure)
            reflection: str | None = None
            if not evaluation["success"] and score < min_score:
                reflect_prompt = REFLECTOR_PROMPT_TEMPLATE.format(
                    task_query=task_query,
                    answer=answer,
                    score=score,
                    reason=evaluation["reason"],
                    trajectory_digest=answer[:500],
                )
                reflect_response = self.reflector.run(reflect_prompt)
                reflection = reflect_response.content or ""

            # 5. Store trial
            self.store.save_trial(
                category=category,
                task_key=task_key,
                trial_id=trial_id,
                query=task_query,
                trajectory_digest=answer[:500],
                final_answer=answer,
                score=score,
                reflection=reflection,
                used_reflections=len(past),
            )

            # 6. Track best
            if score > best_score:
                best_score = score
                best_answer = answer

            trial_details.append({
                "trial": trial_no,
                "trial_id": trial_id,
                "score": score,
                "success": evaluation["success"],
                "reason": evaluation["reason"],
                "has_reflection": reflection is not None,
            })

            # 7. Early stopping
            if evaluation["success"] or score >= min_score:
                break

        return {
            "best_answer": best_answer,
            "best_score": best_score,
            "trials_used": trial_no,
            "trial_details": trial_details,
        }
