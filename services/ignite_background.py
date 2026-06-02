"""Background CrewAI execution and webhook delivery to shoal-web."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import HTTPException

from services.dynamic_personas import clamp_agent_count
from services.metrics import compute_strict_confidence, compute_swarm_credits
from services.crew_orchestration import orchestrate_debate
from services.orchestrator import run_swarm_ignite
from services.webhook_notify import notify_swarm_failure, notify_swarm_success

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
    """
    Sync entrypoint for FastAPI BackgroundTasks.
    Runs CrewAI, then POSTs the ignite payload to the Next.js engine webhook.
    """
    started = time.perf_counter()
    debate_count = clamp_agent_count(agent_count)
    resolved_swarm_size = swarm_size or debate_count

    try:
        print(
            "[ignite_background] vars:",
            {
                "swarmId": swarm_id,
                "model_tier": model_tier,
                "target_audience": target_audience,
                "price_point": price_point,
                "marketing_budget": marketing_budget,
            },
        )

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

        print(
            f"[ignite_background] swarm {swarm_id} ready for webhook: "
            f"messages={len(ignite_fields['messages'])} "
            f"confidence={ignite_fields.get('confidence')} "
            f"transcript={len(ignite_fields.get('debateTranscript') or [])}",
        )

        logger.info(
            "CrewAI complete for swarm %s (runtime=%ss); posting webhook",
            swarm_id,
            runtime,
        )
        notify_swarm_success(swarm_id, ignite_fields)

    except Exception as exc:
        message = _failure_message(exc)
        logger.exception("Background ignite failed for swarm %s: %s", swarm_id, message)
        notify_swarm_failure(swarm_id, message)


def run_simple_debate_and_webhook(
    debate_id: str,
    query: str,
    *,
    model_tier: str = "lite",
) -> None:
    """
    Launch-day minimal debate runner.

    Runs a tiny 3-agent CrewAI workflow and posts a webhook payload with a final
    verdict string + dummy confidence (85) so shoal-web can persist COMPLETED.
    """
    started = time.perf_counter()
    try:
        verdict = orchestrate_debate(query, model_tier=model_tier)

        elapsed_sec = time.perf_counter() - started
        runtime = max(1, int(round(elapsed_sec)))

        executed_agents = 3
        cost = float(compute_swarm_credits(executed_agents))

        ignite_fields: dict[str, Any] = {
            "messages": [{"role": "Manager", "text": verdict}],
            "confidence": 85,
            "runtime": runtime,
            "cost": cost,
            "evidence": [],
            "agentProfiles": [
                {"role": "Researcher"},
                {"role": "Debater"},
                {"role": "Manager"},
            ],
            "debateTranscript": [
                {
                    "agentName": "Researcher",
                    "role": "Researcher",
                    "text": "Research completed.",
                    "timestamp": "T+00:00",
                },
                {
                    "agentName": "Debater",
                    "role": "Debater",
                    "text": "Debate completed.",
                    "timestamp": "T+00:01",
                },
                {
                    "agentName": "Manager",
                    "role": "Manager",
                    "text": verdict,
                    "timestamp": "T+00:02",
                },
            ],
            "recommendedActions": [],
            "minorityDissent": None,
            "model": model_tier,
            "swarmSize": executed_agents,
            "agentCount": executed_agents,
            "response": verdict,
            "consensus": verdict,
        }

        print(
            f"[simple_debate] debate {debate_id} ready for webhook: "
            f"verdict_chars={len(verdict)} confidence=85 runtime={runtime}s"
        )

        notify_swarm_success(debate_id, ignite_fields)
    except Exception as exc:
        message = _failure_message(exc)
        logger.exception("Simple debate failed for %s: %s", debate_id, message)
        notify_swarm_failure(debate_id, message)
