from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class BackendResult:
    text: str
    metadata: dict = field(default_factory=dict)


class InferenceBackend(ABC):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 8.0,
        retry_jitter_seconds: float = 0.25,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.retry_jitter_seconds = retry_jitter_seconds

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> BackendResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._generate_once(prompt, system_prompt=system_prompt)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not self.is_retriable_error(exc):
                    raise
                delay = min(self.retry_max_delay_seconds, self.retry_base_delay_seconds * (2**attempt))
                if self.retry_jitter_seconds:
                    delay += random.uniform(0.0, self.retry_jitter_seconds)
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def is_retriable_error(self, error: Exception) -> bool:
        message = str(error).lower()
        retriable_markers = (
            "rate limit",
            "timeout",
            "temporarily unavailable",
            "temporary failure",
            "overloaded",
            "connection reset",
            "server disconnected",
            "service unavailable",
            "too many requests",
            "transient",
        )
        return any(marker in message for marker in retriable_markers)

    @abstractmethod
    def _generate_once(self, prompt: str, system_prompt: Optional[str] = None) -> BackendResult:
        raise NotImplementedError