"""Task type taxonomy: short tool loops vs long multi-turn rollouts."""

from enum import Enum


class TaskType(str, Enum):
    """Task types that stress different query patterns (see PLANNING.md §3)."""

    SHORT_LOOP = "short_loop"       # Many short requests (tool-style steps)
    LONG_MULTITURN = "long_multiturn"  # Fewer, longer turns
    MIXED = "mixed"                # Combination of both
