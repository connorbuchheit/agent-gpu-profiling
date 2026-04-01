from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# Rough OpenAI-style token estimate when tokenizer unavailable (~4 chars/token).
CHARS_PER_TOKEN_EST: float = 4.0


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def approx_chars_for_tokens(n_tokens: int) -> int:
    return max(16, int(n_tokens * CHARS_PER_TOKEN_EST))


def make_prompt(target_input_tokens: int, seed: int) -> str:
    """Deterministic pseudo-text of roughly target_input_tokens (estimated)."""
    rng = random.Random(seed)
    words = (
        "The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
        "Machine learning inference throughput latency optimization. "
    ).split()
    parts: list[str] = []
    total_chars = 0
    goal = approx_chars_for_tokens(target_input_tokens)
    while total_chars < goal:
        w = rng.choice(words)
        parts.append(w)
        total_chars += len(w) + 1
    return " ".join(parts)[:goal]


def stable_run_id(parts: dict[str, Any]) -> str:
    s = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def utc_run_dir_name() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S", time.gmtime())


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")[:120]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
