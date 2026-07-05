from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Optional, Union


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def load_prompts(
    prompts_file: Union[str, Path],
    mode: str = "lines",
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Load prompts from a text file.

    Modes:
      lines: one non-empty line is one prompt
      file: the entire file content is one prompt
    """
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if mode == "lines":
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
    elif mode == "file":
        content = text.strip()
        prompts = [content] if content else []
    else:
        raise ValueError(f"Unknown prompt mode: {mode!r}. Use 'lines' or 'file'.")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    return prompts
