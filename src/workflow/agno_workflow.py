"""Agno Workflow V2 assembly — formal pipeline using Step, Loop, Condition."""

from __future__ import annotations

from typing import Any

from agno.agent import Agent
from agno.workflow import Condition, Loop, Step, Steps, Workflow

from src.agents.actor import create_actor
from src.agents.evaluator import create_evaluator
from src.agents.reflector import create_reflector
from src.config import get_config


def build_reflexion_workflow(
    task_instruction: str = "Answer the question accurately.",
    max_trials: int | None = None,
) -> Workflow:
    """Build a Reflexion workflow using agno's Workflow V2 primitives.

    Structure:
        Loop (max_trials):
            Step: Actor (solve task)
            Step: Evaluator (judge)
            Condition: if not success → Reflector (generate reflection)

    Note: This workflow uses agno's native agents. For the full hybrid
    Reflexion+ToT version, use src.hybrid.reflexion_tot.HybridReflexionToT.
    """
    cfg = get_config()
    max_trials = max_trials or cfg.reflexion.max_trials

    actor = create_actor()
    evaluator = create_evaluator()
    reflector = create_reflector()

    workflow = Workflow(
        name="ReflexionWorkflow",
        description="Cross-trial self-improvement via verbal reflection",
        steps=Loop(
            name="reflexion_trials",
            max_iterations=max_trials,
            end_condition=f"current_iteration >= {max_trials}",
            steps=Steps(
                name="single_trial",
                steps=[
                    Step(
                        name="actor_solve",
                        agent=actor,
                        description="Actor solves the task using knowledge search and past reflections",
                    ),
                    Step(
                        name="evaluator_judge",
                        agent=evaluator,
                        description="Evaluator judges the quality of the answer",
                    ),
                    Condition(
                        name="needs_reflection",
                        evaluator='last_step_content.contains("false")',
                        steps=[
                            Step(
                                name="reflector_critique",
                                agent=reflector,
                                description="Reflector generates verbal feedback on the failure",
                            ),
                        ],
                    ),
                ],
            ),
        ),
    )

    return workflow
