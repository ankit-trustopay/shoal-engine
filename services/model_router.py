"""Map frontend model selections to OpenRouter model identifiers."""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_DEFAULT_MODEL",
    "deepseek/deepseek-chat",
)

FALLBACK_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "openai/gpt-4o-mini",
)

# OpenRouter slugs for requested UI models
OPENROUTER_MODEL_MAP: dict[str, str] = {
    "meta-llama/llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
    "llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
    "llama-3": "meta-llama/llama-3-8b-instruct",
    "gpt-4o": "openai/gpt-4o",
    "gpt 4o": "openai/gpt-4o",
    "gpt-4o (default)": "openai/gpt-4o",
    "openai/gpt-4o": "openai/gpt-4o",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude 3.5 sonnet": "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek v3": "deepseek/deepseek-chat",
    "deepseek/deepseek-chat": "deepseek/deepseek-chat",
    "deepseek/deepseek-v3": "deepseek/deepseek-v3",
}


def _normalize_model_key(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def resolve_openrouter_model(requested: str | None) -> str:
    """Resolve a frontend model id/label to an OpenRouter model slug."""
    if not requested or not requested.strip():
        return DEFAULT_OPENROUTER_MODEL

    key = _normalize_model_key(requested)

    if key in OPENROUTER_MODEL_MAP:
        return OPENROUTER_MODEL_MAP[key]

    for alias, model_id in OPENROUTER_MODEL_MAP.items():
        if alias in key or key in alias:
            return model_id

    if "/" in requested.strip():
        return requested.strip()

    logger.warning(
        "Unknown model %r — falling back to %s",
        requested,
        DEFAULT_OPENROUTER_MODEL,
    )
    return DEFAULT_OPENROUTER_MODEL


def get_fallback_openrouter_model(primary: str | None = None) -> str:
    """Production fallback when the requested OpenRouter model is unavailable."""
    resolved_primary = resolve_openrouter_model(primary)
    if resolved_primary == FALLBACK_OPENROUTER_MODEL:
        return DEFAULT_OPENROUTER_MODEL
    return FALLBACK_OPENROUTER_MODEL
