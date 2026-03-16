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
    """Heavier rollouts: many long chain-of-thought style prompts, multi-part questions, varied context."""
    prompts = [
        "I'm building a small agent that runs on a GPU. Walk me through step by step: (1) what happens to GPU memory when the model loads, (2) what happens during each short tool-call request, (3) what happens during a long multi-turn reply. Give a concise 3–4 sentence explanation for each.",
        "Compare vLLM and SGLang for this workload: many small requests (tool loops) vs fewer, longer conversations. List 2–3 pros and cons of each for (a) latency and (b) memory, and suggest which you'd pick for an agent that does 5–10 tool calls per task.",
        "Write a minimal Python pseudocode outline (just the main steps, no full code) for a profiler that samples GPU utilization and memory every second during an agent run, logs each step, and at the end writes a CSV and optionally plots utilization over time. Number the steps 1–6.",
        "Explain how PagedAttention in vLLM improves memory use compared to a naive KV cache. What is the main idea in 2–3 sentences, and what trade-off does it make?",
        "What is prefix caching (e.g. in SGLang RadixAttention) and why does it help for multi-turn or agent-style workloads? Give a short technical explanation.",
        "An agent does 8 short tool calls then one long final answer. How would GPU utilization typically look over time (high at start, spikes during tool calls, etc.)? Describe the expected pattern in 3–4 sentences.",
        "What metrics besides GPU % and memory would you want to collect when profiling an LLM server under agent workloads? List 4–5 and say why each matters.",
        "Describe the difference between time-to-first-token (TTFT) and tokens-per-second throughput. When is each more important for (a) interactive chat and (b) batch inference?",
        "Why might a small model (1B params) and a large model (70B) show very different memory curves when running the same number of agent steps? Keep it to 3–4 sentences.",
        "You're comparing two backends on the same hardware. One shows higher peak GPU % but the other finishes the same task faster. What could explain that? Give 2–3 possible reasons.",
        "What is KV cache and why does it grow with context length? How does that affect memory when an agent has a long conversation vs many short turns?",
        "Outline a simple experiment to see whether your agent is bottlenecked by (a) GPU compute or (b) memory bandwidth. What would you measure and what would you look for?",
        "Explain in 2–3 sentences what 'continuous batching' means in LLM serving and why it matters for throughput when many requests are in flight.",
        "If you see GPU utilization drop to 0% between agent steps, what are two likely causes? How would you confirm which one it is?",
        "Why might tool-calling (function calling) add extra latency compared to a single plain text completion? Mention decoding and any extra round-trips.",
        "Describe how you'd design a benchmark that stresses both short tool loops and long reasoning in one run, so you can profile both patterns in a single session.",
        "What is speculative decoding and in one sentence how could it affect latency for an agent that does many short generations?",
        "When profiling on a shared cluster, why might your GPU % numbers differ run to run even for the same task? List 3 factors.",
        "You want to log GPU memory used at each agent step. Why sample right after the step completes rather than at a fixed 1-second interval? When might the interval approach still be useful?",
        "Give a 2–3 sentence summary: how would you choose between vLLM and SGLang for a production agent that does 5–20 tool calls per user request and then one long answer.",
    ]
    return [{"role": "user", "content": prompts[i % len(prompts)]} for i in range(n)]
