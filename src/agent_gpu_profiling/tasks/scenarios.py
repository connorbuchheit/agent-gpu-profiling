"""Concrete scenario definitions: prompts and params per task type."""

from __future__ import annotations

from .types import TaskType


def get_scenario(
    task_type: TaskType,
    *,
    num_steps: int = 5,
    num_turns: int = 3,
) -> list[dict]:
    """
    Return a list of "turns" for the agent. Each turn is a dict with at least:
    - role: "user" | "assistant" | "system"
    - content: str
    Optional: max_tokens, tool_instruction (for short_loop).
    """
    if task_type == TaskType.SHORT_LOOP:
        return _short_loop_scenario(num_steps)
    if task_type == TaskType.LONG_MULTITURN:
        return _long_multiturn_scenario(num_turns)
    if task_type == TaskType.MIXED:
        return _short_loop_scenario(2) + _long_multiturn_scenario(1)
    raise ValueError(f"Unknown task type: {task_type}")


def _short_loop_scenario(n: int) -> list[dict]:
    """N small steps simulating tool use: user asks, assistant responds briefly (tool loop)."""
    steps = []
    for i in range(n):
        steps.append({
            "role": "user",
            "content": f"Step {i + 1}: What is {i + 1} + {i + 2}? Reply with one number only.",
        })
        # Assistant "response" is simulated by the agent runner
    return steps


def _long_multiturn_scenario(n: int) -> list[dict]:
    """N longer turns: more context and longer expected replies."""
    steps = []
    for i in range(n):
        steps.append({
            "role": "user",
            "content": (
                f"Turn {i + 1}: Explain in 2–3 sentences how a simple ReAct-style agent "
                "would use a calculator tool to solve a multi-step math word problem. "
                "Then give a minimal example with one tool call and one final answer."
            ),
        })
    return steps
