"""Minimal agent: call LLM (OpenAI-compat API) for each user turn in a scenario."""

from __future__ import annotations

from typing import Callable

from openai import OpenAI

from agent_gpu_profiling.tasks.types import TaskType


def run_task(
    scenario: list[dict],
    *,
    base_url: str,
    model: str,
    task_type: TaskType,
    max_tokens_per_call: int = 256,
    on_step: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Run the agent over the given scenario. Each user message is sent; we collect
    the model response and append to history. If on_step(step, total_steps) is
    provided, it is called after each completed step (1-based step index).
    Returns the full message history (user + assistant turns).
    """
    client = OpenAI(base_url=base_url, api_key="dummy")
    history: list[dict] = []
    system = (
        "You are a helpful assistant. Keep answers concise unless asked to elaborate. "
        "For math, respond with the final number or short explanation."
    )
    messages = [{"role": "system", "content": system}]

    user_turns = [t for t in scenario if t.get("role") == "user"]
    total = len(user_turns)
    step = 0

    for turn in scenario:
        if turn.get("role") != "user":
            continue
        step += 1
        messages.append({"role": "user", "content": turn["content"]})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens_per_call,
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        messages.append({"role": "assistant", "content": content})
        history.append({"role": "user", "content": turn["content"]})
        history.append({"role": "assistant", "content": content})
        if on_step:
            on_step(step, total)

    return history
