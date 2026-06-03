"""CrewAI-compatible LLM — always uses the shared OpenRouter DeepSeek client."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from services.openrouter_llm import get_default_llm


def build_crew_llm(
    requested_model: str | None = None,
    *,
    model_tier: str | None = None,
    model_mix: float | int | None = None,
    temperature: float = 0.25,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    """Returns shared default_llm; tier/mix/requested_model are ignored."""
    _ = (requested_model, model_tier, model_mix)

    llm = get_default_llm()
    llm.temperature = temperature
    llm.max_tokens = max_tokens
    return llm
