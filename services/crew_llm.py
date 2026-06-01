"""CrewAI LLM instances routed through OpenRouter."""

from __future__ import annotations

import os

from crewai import LLM

from services.model_router import resolve_openrouter_model

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _openrouter_model_id(requested: str | None) -> str:
    slug = resolve_openrouter_model(requested)
    if slug.startswith("openrouter/"):
        return slug
    return f"openrouter/{slug}"


def build_crew_llm(
    requested_model: str | None = None,
    *,
    temperature: float = 0.25,
    max_tokens: int = 4096,
) -> LLM:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    return LLM(
        model=_openrouter_model_id(requested_model),
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
