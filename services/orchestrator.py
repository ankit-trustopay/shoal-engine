"""Swarm orchestration entrypoint — CrewAI hierarchical CEO–worker architecture."""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from services.crew_orchestration import run_hierarchical_crew
from services.dynamic_personas import (
    clamp_agent_count,
    generate_dynamic_personas,
    persona_to_agent_profile,
)
from services.llm import get_client
from services.scraper import scrape_for_premise
from services.swarm_types import SwarmIgniteResult

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
from services.swarm_types import DebateTranscriptEntry, format_debate_timestamp  # noqa: E402,F401


async def run_swarm_ignite(
    swarm_id: str,
    premise: str,
    agent_count: int,
    model: str | None = None,
) -> SwarmIgniteResult:
    """
    Deep research → adversarial personas → CrewAI hierarchical debate → CEO JSON synthesis.
    """
    client: AsyncOpenAI = get_client()
    trimmed_premise = premise.strip()
    debate_count = clamp_agent_count(agent_count)

    logger.info(
        "Starting CrewAI orchestration for swarm %s (agents=%s model=%r)",
        swarm_id,
        debate_count,
        model,
    )

    web_data, evidence = await asyncio.to_thread(scrape_for_premise, trimmed_premise)

    personas = await generate_dynamic_personas(
        client,
        trimmed_premise,
        web_data,
        debate_count,
        model=model,
    )

    manager_synthesis, debate_transcript, messages = await asyncio.to_thread(
        run_hierarchical_crew,
        trimmed_premise,
        personas,
        web_data,
        evidence,
        model,
    )

    agent_profiles = [persona_to_agent_profile(persona) for persona in personas]

    return SwarmIgniteResult(
        messages=messages,
        debate_transcript=debate_transcript,
        evidence=evidence,
        agent_profiles=agent_profiles,
        manager_synthesis=manager_synthesis,
        executed_agent_count=len(personas),
    )
