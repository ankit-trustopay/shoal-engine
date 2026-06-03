"""
Single OpenRouter model for all engine LLM calls (DeepSeek V3 via deepseek/deepseek-chat).
"""

from __future__ import annotations

import os
from typing import Any

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_API_BASE}/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-chat"

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://shoalai.com",
    "X-Title": "Shoal AI",
}

_default_llm: ChatOpenAI | None = None
_probe_done = False


def _mask_key(api_key: str) -> str:
    key = api_key.strip()
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def load_openrouter_api_key() -> str:
    for env_name in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        raw = os.environ.get(env_name, "")
        if raw and str(raw).strip():
            key = str(raw).strip()
            print(
                f"[openrouter_llm] loaded {env_name}={_mask_key(key)} len={len(key)}",
            )
            return key

    print("[openrouter_llm] FATAL: OPENROUTER_API_KEY missing from os.environ")
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


def get_default_llm() -> ChatOpenAI:
    """Return the shared ChatOpenAI client (DeepSeek V3 on OpenRouter)."""
    global _default_llm, _probe_done

    if _default_llm is not None:
        return _default_llm

    api_key = load_openrouter_api_key()

    if not _probe_done:
        probe_openrouter(OPENROUTER_MODEL, api_key)
        _probe_done = True

    print(
        f"[openrouter_llm] ChatOpenAI model={OPENROUTER_MODEL} "
        f"base={OPENROUTER_API_BASE}",
    )

    _default_llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_API_BASE,
        model_name=OPENROUTER_MODEL,
        default_headers=OPENROUTER_HEADERS,
        temperature=0.35,
        max_tokens=2048,
    )

    return _default_llm


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


OPENROUTER_MAX_PARALLEL = 50


def invoke_llm(llm: ChatOpenAI, system: str, user: str, *, stage: str) -> str:
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


def invoke_llm_json(llm: ChatOpenAI, system: str, user: str, *, stage: str) -> str:
    """
    CEO synthesis: request JSON object mode when the provider supports it.
    Falls back to plain invoke if json mode is rejected.
    """
    messages = [
        SystemMessage(
            content=(
                f"{system.strip()}\n\n"
                "You MUST respond with a single valid JSON object only. "
                "No markdown, no code fences, no text outside the JSON."
            ),
        ),
        HumanMessage(content=user),
    ]

    try:
        json_llm = llm.bind(
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        result = json_llm.invoke(messages)
        text = str(getattr(result, "content", result) or "").strip()
        print(f"[openrouter_llm] invoke_llm_json stage={stage} chars={len(text)}")
        return text
    except Exception as exc:
        print(
            f"[openrouter_llm] json_object mode failed stage={stage} "
            f"type={type(exc).__name__}; falling back to plain invoke",
        )
        log_langchain_error(exc, stage=f"{stage}_json_mode")
        return invoke_llm(llm, system, user, stage=stage)
