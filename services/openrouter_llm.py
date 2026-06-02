"""
Raw OpenRouter access via langchain_openai.ChatOpenAI (no CrewAI LLM wrapper).

Logs exact HTTP failures (401, 402, 404, etc.) to stdout for Railway.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_API_BASE}/chat/completions"
LITE_MODEL = "google/gemma-4-31b-it"
PLUS_MODEL = "deepseek/deepseek-v4-flash"

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://shoalai.com",
    "X-Title": "Shoal AI",
}


def _mask_key(api_key: str) -> str:
    key = api_key.strip()
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def load_openrouter_api_key() -> str:
    """Read OPENROUTER_API_KEY from environment (also checks .env via main load_dotenv)."""
    for env_name in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        raw = os.environ.get(env_name, "")
        if raw and str(raw).strip():
            key = str(raw).strip()
            print(
                f"[openrouter_llm] loaded {env_name}={_mask_key(key)} "
                f"len={len(key)}",
            )
            return key

    print(
        "[openrouter_llm] FATAL: OPENROUTER_API_KEY missing from os.environ. "
        f"keys_sample={[k for k in os.environ if 'OPEN' in k.upper()][:8]}",
    )
    raise RuntimeError("OPENROUTER_API_KEY is not configured")


def _classify_http_error(status_code: int, body: str) -> str:
    snippet = (body or "")[:600]
    if status_code == 401:
        return f"401 Unauthorized — invalid or missing API key. body={snippet}"
    if status_code == 402:
        return f"402 Insufficient Funds — add OpenRouter credits. body={snippet}"
    if status_code == 404:
        return f"404 Model Not Found — check model slug. body={snippet}"
    if status_code == 429:
        return f"429 Rate Limited. body={snippet}"
    return f"HTTP {status_code} from OpenRouter. body={snippet}"


def probe_openrouter(model_name: str, api_key: str) -> None:
    """
  Direct HTTP probe (bypasses LangChain) to surface the exact OpenRouter error.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **OPENROUTER_HEADERS,
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 16,
    }

    print(f"[openrouter_llm] probe POST {OPENROUTER_CHAT_URL} model={model_name}")

    try:
        response = requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )
    except requests.RequestException as exc:
        print(f"[openrouter_llm] probe network error: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"OpenRouter network error: {exc}") from exc

    print(
        f"[openrouter_llm] probe status={response.status_code} "
        f"body={response.text[:800]}",
    )

    if response.status_code >= 400:
        raise RuntimeError(_classify_http_error(response.status_code, response.text))


def resolve_model_name(model_mix: float | int) -> str:
    return PLUS_MODEL if int(model_mix) > 50 else LITE_MODEL


def get_llm(model_mix: float | int) -> ChatOpenAI:
    """
    Build ChatOpenAI pointed at OpenRouter. Probes API before returning.
    """
    mix = int(model_mix)
    model_name = resolve_model_name(mix)
    api_key = load_openrouter_api_key()

    probe_openrouter(model_name, api_key)

    print(
        f"[openrouter_llm] ChatOpenAI model={model_name} base={OPENROUTER_API_BASE}",
    )

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_API_BASE,
        model_name=model_name,
        default_headers=OPENROUTER_HEADERS,
        temperature=0.35,
        max_tokens=2048,
    )


def log_langchain_error(exc: Exception, *, stage: str) -> None:
    print(f"[openrouter_llm] LangChain error stage={stage} type={type(exc).__name__}")
    print(f"[openrouter_llm] LangChain error detail: {exc}")

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        text = getattr(response, "text", None)
        if status is not None:
            print(f"[openrouter_llm] LangChain response status={status}")
        if text:
            print(f"[openrouter_llm] LangChain response body={str(text)[:800]}")

    body = getattr(exc, "body", None)
    if body:
        print(f"[openrouter_llm] LangChain body={str(body)[:800]}")


def invoke_llm(llm: ChatOpenAI, system: str, user: str, *, stage: str) -> str:
    """Single chat completion; logs and re-raises with classified message."""
    try:
        result = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ],
        )
    except Exception as exc:
        log_langchain_error(exc, stage=stage)
        raise

    content: Any = getattr(result, "content", result)
    text = str(content or "").strip()
    print(f"[openrouter_llm] invoke stage={stage} chars={len(text)}")
    return text
