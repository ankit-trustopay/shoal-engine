"""Background CrewAI execution and webhook delivery to shoal-web."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import HTTPException

from services.dynamic_personas import clamp_agent_count
from services.metrics import compute_strict_confidence, compute_swarm_credits
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


def build_ignite_webhook_payload(
    *,
    swarm_id: str,
    result: Any,
    swarm_size: int,
    agent_count: int,
    model: str | None,
    runtime: int,
) -> dict[str, Any]:
    """Flat JSON body for POST /api/webhooks/engine (parse-engine-webhook compatible)."""
    synthesis = result.manager_synthesis
    sentiments = synthesis.agent_sentiments

    confidence, votes_for, votes_against, votes_neutral = compute_strict_confidence(
        synthesis.consensus,
        sentiments,
        swarm_size=swarm_size,
    )

    executed_agents = result.executed_agent_count
    cost = compute_swarm_credits(executed_agents)

    return {
        "swarmId": swarm_id,
        "messages": [
            {"role": msg["role"], "text": msg["text"]} for msg in result.messages
        ],
        "confidence": confidence,
        "votesFor": votes_for,
        "votesAgainst": votes_against,
        "votesNeutral": votes_neutral,
        "runtime": runtime,
        "cost": cost,
        "evidence": result.evidence,
        "agentProfiles": result.agent_profiles,
        "swarmSize": swarm_size,
        "agentCount": executed_agents,
        "model": model,
        "recommendedActions": [
            {
                "step": action.step,
                "title": action.title,
                "body": action.body,
            }
            for action in synthesis.recommended_actions
        ],
        "minorityDissent": synthesis.minority_dissent or None,
        "debateTranscript": [
            {
                "agentName": entry.agentName,
                "role": entry.role,
                "text": entry.text,
                "timestamp": entry.timestamp,
            }
            for entry in result.debate_transcript
        ],
    }


def run_crew_and_webhook(
    swarm_id: str,
    premise: str,
    agent_count: int,
    model: str | None,
    swarm_size: int | None,
) -> None:
    """
    Sync entrypoint for FastAPI BackgroundTasks.
    Runs CrewAI, then POSTs the ignite payload to the Next.js engine webhook.
    """
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
            )
        )

        elapsed_sec = time.perf_counter() - started
        runtime = max(1, int(round(elapsed_sec)))

        payload = build_ignite_webhook_payload(
            swarm_id=swarm_id,
            result=result,
            swarm_size=resolved_swarm_size,
            agent_count=debate_count,
            model=model,
            runtime=runtime,
        )

        if not payload.get("messages"):
            raise ValueError("CrewAI returned no debate messages")

        logger.info(
            "CrewAI complete for swarm %s (runtime=%ss); posting webhook",
            swarm_id,
            runtime,
        )
        notify_swarm_success(swarm_id, payload)

    except Exception as exc:
        message = _failure_message(exc)
        logger.exception("Background ignite failed for swarm %s: %s", swarm_id, message)
        notify_swarm_failure(swarm_id, message)
