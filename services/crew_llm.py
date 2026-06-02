"""CrewAI-compatible LLM instances routed through OpenRouter (ChatOpenAI)."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from services.openrouter_llm import get_llm


def _model_mix_from_tier(model_tier: str | None) -> int:
    return 100 if (model_tier or "").strip().lower() == "plus" else 0


def build_crew_llm(
    requested_model: str | None = None,
    *,
    model_tier: str | None = None,
    model_mix: float | int | None = None,
    temperature: float = 0.25,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    """
    Build a LangChain ChatOpenAI client for CrewAI agents.

    requested_model is accepted for API compatibility; tier/mix select Gemma vs DeepSeek.
    """
    _ = requested_model

    if model_mix is not None:
        mix = int(model_mix)
    else:
        mix = _model_mix_from_tier(model_tier)

    llm = get_llm(mix)
    llm.temperature = temperature
    llm.max_tokens = max_tokens
    return llm
