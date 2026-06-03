"""Map frontend model selections to OpenRouter model identifiers."""

from __future__ import annotations

import logging
import os
import re

from services.openrouter_llm import OPENROUTER_MODEL

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", OPENROUTER_MODEL)

FALLBACK_OPENROUTER_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", OPENROUTER_MODEL)

OPENROUTER_MODEL_MAP: dict[str, str] = {
    "lite": OPENROUTER_MODEL,
    "plus": OPENROUTER_MODEL,
    "deepseek": OPENROUTER_MODEL,
    "deepseek-chat": OPENROUTER_MODEL,
    "deepseek/deepseek-chat": OPENROUTER_MODEL,
    "deepseek-v3": OPENROUTER_MODEL,
    "deepseek/deepseek-v3": OPENROUTER_MODEL,
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
}


def _normalize_model_key(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def resolve_openrouter_model(requested: str | None) -> str:
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
    resolved_primary = resolve_openrouter_model(primary)
    if resolved_primary == FALLBACK_OPENROUTER_MODEL:
        return DEFAULT_OPENROUTER_MODEL
    return FALLBACK_OPENROUTER_MODEL
