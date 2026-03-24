"""Minimal agent: LLM + optional tool-call loop (OpenAI-compat API)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from openai import OpenAI

from agent_gpu_profiling.tasks.types import TaskType

from .tools import TOOLS, execute_tool

MAX_TOOL_ROUNDS = 10  # max LLM calls per user message when using tools


def run_task(
    scenario: list[dict],
    *,
    base_url: str,
    model: str,
    task_type: TaskType,
    max_tokens_per_call: int = 256,
    on_step: Callable[[int, int, dict[str, Any] | None], None] | None = None,
    tools: list[dict] | None = None,
) -> list[dict]:
    """
    Run the agent over the given scenario. on_step(step, total, step_data) is called after each turn.
    step_data may include latency_ms, prompt_tokens, completion_tokens. Returns full message history.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or "dummy"
    client = OpenAI(base_url=base_url, api_key=api_key)
    history: list[dict] = []
    base_system = (
        "You are a helpful assistant. Use the tools when needed to answer. "
        "Keep answers concise unless asked to elaborate."
    )
    # Leading scenario entries with role "system" are merged (e.g. long shared doc for RadixAttention stress).
    system_chunks: list[str] = [base_system]
    rest: list[dict] = []
    seen_user = False
    for item in scenario:
        if not seen_user and item.get("role") == "system":
            system_chunks.append(str(item.get("content", "")))
            continue
        seen_user = True
        rest.append(item)
    merged_system = "\n\n---\n\n".join(s for s in system_chunks if s)
    messages: list[dict[str, Any]] = [{"role": "system", "content": merged_system}]

    user_turns = [t for t in rest if t.get("role") == "user"]
    total_user_turns = len(user_turns)
    step = 0

    for turn in rest:
        if turn.get("role") != "user":
            continue
        step += 1
        messages.append({"role": "user", "content": turn["content"]})
        history.append({"role": "user", "content": turn["content"]})

        if tools:
            _run_tool_loop(client, model, messages, history, max_tokens_per_call, tools, step, total_user_turns, on_step)
        else:
            _run_one_turn(client, model, messages, history, max_tokens_per_call, step, total_user_turns, on_step)

    return history


def _run_one_turn(
    client: OpenAI,
    model: str,
    messages: list[dict],
    history: list[dict],
    max_tokens: int,
    step: int,
    total: int,
    on_step: Callable[[int, int, dict[str, Any] | None], None] | None,
) -> None:
    """Single LLM call; append assistant text to messages and history. Pass latency and token usage to on_step."""
    step_data: dict[str, Any] = {}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    step_data["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    if getattr(resp, "usage", None):
        step_data["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", None)
        step_data["completion_tokens"] = getattr(resp.usage, "completion_tokens", None)
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    messages.append({"role": "assistant", "content": content})
    history.append({"role": "assistant", "content": content})
    if on_step:
        on_step(step, total, step_data)


def _run_tool_loop(
    client: OpenAI,
    model: str,
    messages: list[dict],
    history: list[dict],
    max_tokens: int,
    tools: list[dict],
    user_step: int,
    total_user_turns: int,
    on_step: Callable[[int, int, dict[str, Any] | None], None] | None,
) -> None:
    """Repeatedly call LLM; on tool_calls, execute tools, append results, call again. on_step per LLM call."""
    round_index = 0
    while round_index < MAX_TOOL_ROUNDS:
        round_index += 1
        step_data: dict[str, Any] = {}
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
        )
        step_data["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        if getattr(resp, "usage", None):
            step_data["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", None)
            step_data["completion_tokens"] = getattr(resp.usage, "completion_tokens", None)
        choice = resp.choices[0]
        msg = choice.message
        # Build assistant message for API (may have content and/or tool_calls)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)
        history.append({"role": "assistant", "content": msg.content or "", "tool_calls": getattr(msg, "tool_calls", None)})

        if on_step:
            on_step(round_index, MAX_TOOL_ROUNDS, step_data)

        if not getattr(msg, "tool_calls", None):
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return
