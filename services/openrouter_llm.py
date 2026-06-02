"""OpenRouter LLM via LangChain ChatOpenAI (avoids CrewAI/LiteLLM model crashes)."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
LITE_MODEL = "google/gemma-4-31b-it"
PLUS_MODEL = "deepseek/deepseek-v4-flash"

OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://shoalai.com",
    "X-Title": "Shoal AI",
}


def get_llm(model_mix: float | int) -> ChatOpenAI:
    """
    If model_mix is low (e.g., 0), use Lite (Gemma). If high (>50), use Plus (DeepSeek).
    """
    model_name = PLUS_MODEL if int(model_mix) > 50 else LITE_MODEL

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or not str(api_key).strip():
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_API_BASE,
        model_name=model_name,
        default_headers=OPENROUTER_DEFAULT_HEADERS,
    )
