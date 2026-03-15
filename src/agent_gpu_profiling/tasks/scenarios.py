"""Concrete scenario definitions: prompts and params per task type."""

from __future__ import annotations

from .types import TaskType


def get_scenario(
    task_type: TaskType,
    *,
    num_steps: int = 5,
    num_turns: int = 3,
    num_reasoning_turns: int = 3,
) -> list[dict]:
    """
    Return a list of "turns" for the agent. Each turn is a dict with at least:
    - role: "user" | "assistant" | "system"
    - content: str
    """
    if task_type == TaskType.SHORT_LOOP:
        return _short_loop_scenario(num_steps)
    if task_type == TaskType.LONG_MULTITURN:
        return _long_multiturn_scenario(num_turns)
    if task_type == TaskType.MIXED:
        return _short_loop_scenario(2) + _long_multiturn_scenario(1)
    if task_type == TaskType.TOOL_LOOP:
        return _tool_loop_scenario()
    if task_type == TaskType.LONG_REASONING:
        return _long_reasoning_scenario(num_reasoning_turns)
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


def _tool_loop_scenario() -> list[dict]:
    """One user message that should trigger multiple tool calls (calculator) and back-and-forth."""
    return [
        {
            "role": "user",
            "content": (
                "Use the calculator tool to compute (1+2)*(3+4). "
                "Call the tool for each part if needed, then give me the final number."
            ),
        },
    ]


def _long_reasoning_scenario(n: int) -> list[dict]:
    """Heavier rollouts: long chain-of-thought style prompts, multi-part questions, more context."""
    return [
        {
            "role": "user",
            "content": (
                "I'm building a small agent that runs on a GPU. It does a few short tool calls and then a longer answer. "
                "Walk me through step by step: (1) what happens to GPU memory when the model loads, "
                "(2) what happens during each short tool-call request, (3) what happens during a long multi-turn reply. "
                "Give a concise 3–4 sentence explanation for each, then one sentence on how you'd profile this."
            ),
        },
        {
            "role": "user",
            "content": (
                "Compare vLLM and SGLang for this workload: many small requests (tool loops) vs fewer, longer conversations. "
                "List 2–3 pros and cons of each for (a) latency and (b) memory, and suggest which you'd pick for an agent that "
                "does 5–10 tool calls per task and then one long final answer. Keep it under 150 words."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a minimal Python pseudocode outline (just the main steps, no full code) for a profiler that: "
                "samples GPU utilization and memory every second during an agent run, logs each step (e.g. step 1/10), "
                "and at the end writes a CSV and optionally plots utilization over time. Number the steps 1–6."
            ),
        },
    ][:n]
