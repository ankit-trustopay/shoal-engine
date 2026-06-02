"""Map frontend model selections to OpenRouter model identifiers."""

from __future__ import annotations

import logging
import os
import re

from services.openrouter_llm import LITE_MODEL, PLUS_MODEL

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", LITE_MODEL)

FALLBACK_OPENROUTER_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", PLUS_MODEL)

OPENROUTER_MODEL_MAP: dict[str, str] = {
    "lite": LITE_MODEL,
    "gemma": LITE_MODEL,
    "google/gemma-4-31b-it": LITE_MODEL,
    "plus": PLUS_MODEL,
    "deepseek": PLUS_MODEL,
    "deepseek-v4-flash": PLUS_MODEL,
    "deepseek/deepseek-v4-flash": PLUS_MODEL,
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "deepseek-v3": "deepseek/deepseek-chat",
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
