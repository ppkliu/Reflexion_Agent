"""Prompt templates for the Reflexion loop agents."""

ACTOR_SYSTEM_TEMPLATE = """\
You are a reasoning agent that solves tasks step by step.
Use the search_knowledge tool to retrieve relevant information from local documents.

{reflection_block}

Avoid repeating mistakes from previous trials.
Think carefully, then provide a clear, complete final answer.

Current task:
{task_query}
"""

EVALUATOR_PROMPT_TEMPLATE = """\
Task: {task_query}

Candidate answer:
{answer}

{instruction}

Judge the quality of this answer. Output ONLY valid JSON (no markdown fences):
{{"success": true/false, "score": 0.0-1.0, "reason": "brief explanation"}}
"""

REFLECTOR_PROMPT_TEMPLATE = """\
A reasoning agent attempted the following task and received this evaluation.

Task: {task_query}
Final answer: {answer}
Evaluation: score={score}, feedback={reason}

Trajectory summary:
{trajectory_digest}

Analyze what went wrong:
- What was the key mistake or gap?
- What information was missing or ignored?
- What concrete step should be taken differently next time?

Provide 3-5 specific, actionable bullet points.
"""
