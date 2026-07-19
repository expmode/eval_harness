from __future__ import annotations

from .api_backend import APIBackend
from .base import InferenceBackend
from .vllm_backend import VLLMBackend
from ..config import ModelConfig


def create_backend(config: ModelConfig) -> InferenceBackend:
    backend = config.resolved_backend_label.lower()
    common_kwargs = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "retry_base_delay_seconds": config.retry_base_delay_seconds,
        "retry_max_delay_seconds": config.retry_max_delay_seconds,
        "retry_jitter_seconds": config.retry_jitter_seconds,
    }

    if backend in {"openai", "anthropic", "together", "openai_compatible"}:
        return APIBackend(provider=backend, **common_kwargs)
    if backend == "vllm":
        return VLLMBackend(mode="local", **common_kwargs)
    if backend == "vllm_openai":
        return VLLMBackend(mode="openai_compatible", **common_kwargs)

    raise ValueError(f"Unsupported backend: {config.backend}")