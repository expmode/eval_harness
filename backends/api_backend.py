from __future__ import annotations

import time
from typing import Optional

from .base import BackendResult, InferenceBackend


class APIBackend(InferenceBackend):
    def __init__(self, *, provider: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.provider = provider.lower()

    def _load_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for API and OpenAI-compatible backends"
            ) from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _load_anthropic_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for Anthropic backends") from exc
        return anthropic.Anthropic(api_key=self.api_key)

    def _generate_once(self, prompt: str, system_prompt: Optional[str] = None) -> BackendResult:
        if self.provider in {"openai", "openai_compatible", "together"}:
            client = self._load_openai_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            started_at = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
            )
            text = response.choices[0].message.content or ""
            if not text.strip():
                raise ValueError("Empty completion returned by provider")
            return BackendResult(
                text=text,
                metadata={
                    "provider": self.provider,
                    "latency_seconds": round(time.time() - started_at, 4),
                },
            )

        if self.provider == "anthropic":
            client = self._load_anthropic_client()
            started_at = time.time()
            response = client.messages.create(
                model=self.model,
                system=system_prompt or "",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout_seconds,
            )
            parts = []
            for block in response.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            joined = "".join(parts)
            if not joined.strip():
                raise ValueError("Empty completion returned by provider")
            return BackendResult(
                text=joined,
                metadata={
                    "provider": self.provider,
                    "latency_seconds": round(time.time() - started_at, 4),
                },
            )

        raise ValueError(f"Unsupported API provider: {self.provider}")