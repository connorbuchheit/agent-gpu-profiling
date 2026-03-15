"""Tool definitions (OpenAI function format) and executor for the agent."""

from __future__ import annotations

import json
from typing import Any

# OpenAI-compatible tool definitions for chat.completions.create(tools=...)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Input a single expression string, e.g. '2 + 3' or '(1+2)*(3+4)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A math expression to evaluate (e.g. 2+3, 10/2)."},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time as a string (for demo).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name with the given arguments; return a string result."""
    if name == "calculator":
        expr = (arguments.get("expression") or "").strip()
        if not expr:
            return "Error: no expression given."
        try:
            # Safe eval: only numbers and basic ops (no builtins)
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expr):
                return "Error: only numbers and + - * / ( ) allowed."
            result = eval(expr)
            return str(result) if not isinstance(result, float) else str(round(result, 4))
        except Exception as e:
            return f"Error: {e}"
    if name == "get_time":
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Unknown tool: {name}"
