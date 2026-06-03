"""
Shoal debate orchestration — MiroFish headless adapter (2-turn token governor).

Turn 1: N parallel stateless workers (asyncio.gather + Semaphore(50))
Turn 2: CEO synthesizes executive JSON for the Vercel webhook

No CrewAI, no OASIS, no Zep memory, no agent-to-agent chat.
"""

from __future__ import annotations

import asyncio
import logging

from services.debate_result_codec import (
    DebateAgent,
    DebateResult,
    build_debate_webhook_payload,
    evidence_for_webhook,
    fallback_debate_result,
    finalize_debate_result,
)
from services.mirofish_adapter import run_debate_swarm_async
from services.scraper import EvidenceItem, scrape_for_premise

logger = logging.getLogger(__name__)

# Re-export for ignite_background / tests
__all__ = [
    "DebateAgent",
    "DebateResult",
    "build_debate_webhook_payload",
    "ensure_verdict",
    "fallback_debate_result",
    "finalize_debate_result",
    "run_debate_crew",
]


def ensure_verdict(text: str | None) -> str:
    from services.debate_result_codec import ensure_verdict as _ensure

    return _ensure(text)


def run_debate_crew(
    query: str,
    *,
    model_mix: float = 0,
    agent_count: int = 3,
    web_context: str | None = None,
    evidence_items: list[EvidenceItem] | None = None,
) -> DebateResult:
    """
    Run the MiroFish adapter debate pipeline (sync entrypoint).

    model_mix is accepted for API compatibility but ignored (single model only).
    """
    _ = model_mix
    trimmed = (query or "").strip()
    if not trimmed:
        return fallback_debate_result("Missing query")

    print(
        f"[debate_crew] MiroFish adapter agent_count={agent_count} "
        f"model=deepseek/deepseek-chat",
    )

    try:
        return asyncio.run(
            run_debate_swarm_async(
                trimmed,
                agent_count=agent_count,
                web_context=web_context,
                evidence_items=evidence_items,
            ),
        )
    except Exception as exc:
        logger.exception("debate_crew asyncio runner failed")
        print(f"[debate_crew] FATAL adapter failure: {exc}")

        if web_context is None or evidence_items is None:
            try:
                _, evidence_items = scrape_for_premise(trimmed)
            except Exception:
                evidence_items = []

        evidence_rows = evidence_for_webhook(evidence_items or [])
        return finalize_debate_result(
            {
                **fallback_debate_result(str(exc), trimmed),
                "evidence": evidence_rows,
            },
        )
