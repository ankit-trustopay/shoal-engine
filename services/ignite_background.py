"""Background CrewAI execution and webhook delivery to shoal-web."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import HTTPException

from services.debate_constants import AI_MODEL_ERROR_VERDICT
from services.debate_crew import (
    ensure_verdict,
    fallback_debate_result,
    finalize_debate_result,
    run_debate_crew,
)
from services.dynamic_personas import clamp_agent_count
from services.metrics import compute_strict_confidence, compute_swarm_credits
from services.orchestrator import run_swarm_ignite
from services.webhook_notify import notify_debate_completion, notify_swarm_failure, notify_swarm_success

logger = logging.getLogger(__name__)


def _failure_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            return detail
        return str(detail)
    return str(exc) or exc.__class__.__name__


def _normalize_message(msg: Any) -> dict[str, str] | None:
    if not isinstance(msg, dict):
        return None
    role = msg.get("role") or msg.get("Role") or "Agent"
    text = (
        msg.get("text")
        or msg.get("Text")
        or msg.get("content")
        or msg.get("message")
        or ""
    )
    text = str(text).strip()
    if not text:
        return None
    return {"role": str(role), "text": text}


def _messages_from_transcript(
    debate_transcript: list[Any],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for entry in debate_transcript:
        if hasattr(entry, "role") and hasattr(entry, "text"):
            role = getattr(entry, "role", "Agent")
            text = str(getattr(entry, "text", "")).strip()
        elif isinstance(entry, dict):
            role = entry.get("role") or entry.get("Role") or "Agent"
            text = str(
                entry.get("text") or entry.get("Text") or entry.get("content") or "",
            ).strip()
        else:
            continue
        if text:
            messages.append({"role": str(role), "text": text})
    return messages


def build_ignite_fields(
    *,
    result: Any,
    swarm_size: int,
    model: str | None,
    runtime: int,
) -> dict[str, Any]:
    """Ignite fields nested under reportData for shoal-web (camelCase keys)."""
    synthesis = result.manager_synthesis
    sentiments = synthesis.agent_sentiments

    confidence, votes_for, votes_against, votes_neutral = compute_strict_confidence(
        synthesis.consensus,
        sentiments,
        swarm_size=swarm_size,
    )

    executed_agents = result.executed_agent_count
    cost = float(compute_swarm_credits(executed_agents))

    messages: list[dict[str, str]] = []
    for msg in result.messages:
        normalized = _normalize_message(msg)
        if normalized:
            messages.append(normalized)

    if not messages:
        messages = _messages_from_transcript(result.debate_transcript)

    if not messages and synthesis.consensus:
        messages = [{"role": "Manager", "text": synthesis.consensus.strip()}]

    debate_transcript = [
        {
            "agentName": entry.agentName,
            "role": entry.role,
            "text": entry.text,
            "timestamp": entry.timestamp,
        }
        for entry in result.debate_transcript
    ]

    evidence = [
        {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in (result.evidence or [])
        if isinstance(item, dict)
    ]

    agent_profiles = list(result.agent_profiles or [])

    return {
        "messages": messages,
        "confidence": int(confidence),
        "votesFor": int(votes_for),
        "votesAgainst": int(votes_against),
        "votesNeutral": int(votes_neutral),
        "runtime": int(runtime),
        "cost": cost,
        "evidence": evidence,
        "agentProfiles": agent_profiles,
        "swarmSize": int(swarm_size),
        "agentCount": int(executed_agents),
        "model": model,
        "recommendedActions": [
            {
                "step": int(action.step),
                "title": action.title,
                "body": action.body,
            }
            for action in synthesis.recommended_actions
        ],
        "minorityDissent": synthesis.minority_dissent or None,
        "debateTranscript": debate_transcript,
    }


def run_crew_and_webhook(
    swarm_id: str,
    premise: str,
    agent_count: int,
    model: str | None,
    swarm_size: int | None,
    model_tier: str = "lite",
    target_audience: str | None = None,
    price_point: str | None = None,
    marketing_budget: str | None = None,
) -> None:
    """Sync entrypoint for FastAPI BackgroundTasks (ignite path)."""
    started = time.perf_counter()
    debate_count = clamp_agent_count(agent_count)
    resolved_swarm_size = swarm_size or debate_count

    try:
        result = asyncio.run(
            run_swarm_ignite(
                swarm_id,
                premise,
                agent_count=debate_count,
                model=model,
                model_tier=model_tier,
                target_audience=target_audience,
                price_point=price_point,
                marketing_budget=marketing_budget,
            )
        )

        elapsed_sec = time.perf_counter() - started
        runtime = max(1, int(round(elapsed_sec)))

        ignite_fields = build_ignite_fields(
            result=result,
            swarm_size=resolved_swarm_size,
            model=model,
            runtime=runtime,
        )

        if not ignite_fields.get("messages"):
            raise ValueError("CrewAI returned no debate messages")

        print(f"[ignite_background] posting ignite webhook swarm={swarm_id}")
        notify_swarm_success(swarm_id, ignite_fields)

    except Exception as exc:
        message = _failure_message(exc)
        logger.exception("Background ignite failed for swarm %s: %s", swarm_id, message)
        notify_swarm_failure(swarm_id, message)


def run_simple_debate_and_webhook(
    debate_id: str,
    query: str,
    *,
    agent_count: int = 3,
    model_mix: float = 0,
) -> None:
    """Run debate crew and POST canonical JSON to shoal-web."""
    print(f"[ignite_background] debate start id={debate_id} model=deepseek/deepseek-chat")
    started = time.perf_counter()

    try:
        result = finalize_debate_result(run_debate_crew(query))
    except Exception as exc:
        logger.exception("Debate crew unexpected failure for %s", debate_id)
        print(f"[ignite_background] debate exception: {exc}")
        result = fallback_debate_result(str(exc))

    elapsed_sec = time.perf_counter() - started
    runtime = max(1, int(round(elapsed_sec)))
    billed_agents = max(3, agent_count)
    cost = float(compute_swarm_credits(billed_agents))

    verdict = ensure_verdict(str(result.get("verdict") or ""))
    agents = list(result.get("agents") or [])
    if not agents:
        agents = [{"name": "CEO Synthesizer", "position": AI_MODEL_ERROR_VERDICT}]

    print(
        f"[ignite_background] webhook debate_id={debate_id} "
        f"verdict_len={len(verdict)} agents={len(agents)}",
    )

    notify_debate_completion(
        debate_id,
        verdict=verdict,
        confidence=int(result.get("confidence") or 0),
        agents=agents,
        tldr=list(result.get("tldr") or []),
        friction_matrix=list(result.get("friction_matrix") or []),
        pre_mortem=result.get("pre_mortem"),
        execution_roadmap=result.get("execution_roadmap"),
        runtime=runtime,
        cost=cost,
        agent_count=billed_agents,
    )
