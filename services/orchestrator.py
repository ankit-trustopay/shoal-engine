"""Multi-agent swarm orchestration (adversarial personas + manager synthesis)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from services.dynamic_personas import (
    DynamicPersona,
    clamp_agent_count,
    generate_dynamic_personas,
    persona_to_agent_profile,
)
from services.llm import (
    ManagerSynthesis,
    get_client,
    get_manager_synthesis,
    get_persona_debate_response,
)
from services.scraper import EvidenceItem, scrape_for_premise

logger = logging.getLogger(__name__)

SECONDS_PER_DEBATE_TURN = 12


def format_debate_timestamp(turn_index: int) -> str:
    """Synthetic war-room clock for transcript entries (00:00, 00:12, …)."""
    total_seconds = turn_index * SECONDS_PER_DEBATE_TURN
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class DebateTranscriptEntry:
    agentName: str
    role: str
    text: str
    timestamp: str


@dataclass
class SwarmIgniteResult:
    messages: list[dict[str, str]]
    debate_transcript: list[DebateTranscriptEntry]
    evidence: list[EvidenceItem]
    agent_profiles: list[dict[str, Any]]
    manager_synthesis: ManagerSynthesis
    executed_agent_count: int


async def run_swarm_ignite(
    swarm_id: str,
    premise: str,
    agent_count: int,
    model: str | None = None,
) -> SwarmIgniteResult:
    """
    Execute deep research, adversarial persona generation, sequential debate, and synthesis.
    """
    client = get_client()
    trimmed_premise = premise.strip()
    debate_count = clamp_agent_count(agent_count)

    logger.info(
        "Starting orchestration for swarm %s (agents=%s model=%r)",
        swarm_id,
        debate_count,
        model,
    )

    web_data, evidence = await asyncio.to_thread(scrape_for_premise, trimmed_premise)
    logger.debug(
        "Web data preview for swarm %s: %s",
        swarm_id,
        web_data[:500],
    )

    personas: list[DynamicPersona] = await generate_dynamic_personas(
        client,
        trimmed_premise,
        web_data,
        debate_count,
        model=model,
    )

    executed_count = len(personas)
    logger.info(
        "Swarm %s adversarial panel (%s): %s",
        swarm_id,
        executed_count,
        ", ".join(p["role"] for p in personas),
    )

    all_roles = [persona["role"] for persona in personas]

    agent_results: list[dict[str, str]] = []
    debate_transcript: list[DebateTranscriptEntry] = []
    prior_turns: list[dict[str, str]] = []

    for turn_index, persona in enumerate(personas):
        opposing_roles = [role for role in all_roles if role != persona["role"]]
        turn_message = await get_persona_debate_response(
            client,
            persona,
            trimmed_premise,
            web_data,
            model=model or "",
            opposing_roles=opposing_roles,
            prior_turns=prior_turns,
        )
        agent_results.append(turn_message)

        entry = DebateTranscriptEntry(
            agentName=str(persona.get("name") or turn_message["role"]),
            role=turn_message["role"],
            text=turn_message["text"],
            timestamp=format_debate_timestamp(turn_index),
        )
        debate_transcript.append(entry)
        prior_turns.append(
            {
                "agentName": entry.agentName,
                "role": entry.role,
                "text": entry.text,
                "timestamp": entry.timestamp,
            },
        )

        logger.info(
            "Swarm %s debate turn %s/%s — %s (%s)",
            swarm_id,
            turn_index + 1,
            executed_count,
            entry.agentName,
            entry.timestamp,
        )

    combined_perspectives = "\n\n".join(
        (
            f"[{entry.timestamp}] {entry.agentName} ({entry.role}):\n{entry.text}"
            for entry in debate_transcript
        ),
    )

    manager_synthesis = await get_manager_synthesis(
        client,
        trimmed_premise,
        web_data,
        combined_perspectives,
        [msg["role"] for msg in agent_results],
        model=model or "",
    )

    manager_result = {
        "role": "Manager",
        "text": manager_synthesis.consensus,
    }

    agent_profiles = [persona_to_agent_profile(persona) for persona in personas]

    logger.info(
        "Swarm %s complete: %s transcript turns + manager (sentiments=%s)",
        swarm_id,
        len(debate_transcript),
        manager_synthesis.agent_sentiments,
    )

    return SwarmIgniteResult(
        messages=[*agent_results, manager_result],
        debate_transcript=debate_transcript,
        evidence=evidence,
        agent_profiles=agent_profiles,
        manager_synthesis=manager_synthesis,
        executed_agent_count=executed_count,
    )
