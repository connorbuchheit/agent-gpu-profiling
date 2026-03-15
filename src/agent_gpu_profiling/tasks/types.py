"""Task type taxonomy: short tool loops vs long multi-turn rollouts."""

from enum import Enum


class TaskType(str, Enum):
    """Task types that stress different query patterns (see PLANNING.md §3)."""

    SHORT_LOOP = "short_loop"           # Many short requests (tool-style steps)
    LONG_MULTITURN = "long_multiturn"   # Fewer, longer turns
    MIXED = "mixed"                     # Combination of both
    TOOL_LOOP = "tool_loop"             # One user query that triggers real tool calls (multi round-trip)
    LONG_REASONING = "long_reasoning"   # Heavier prompts: chain-of-thought, long context, multi-part
