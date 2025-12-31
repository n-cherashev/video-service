from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OllamaSettings:
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout_seconds: float = 30.0
    retries: int = 2
    backoff_seconds: float = 0.5
    keep_alive: str | None = "5m"
    options: dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "num_predict": 600  # Conservative token limit for structured output
    })
    max_input_chars: int = 4000

    cache_enabled: bool = True
    cache_max_items: int = 4096