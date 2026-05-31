"""Multi-agent swarm orchestration (dynamic personas + manager synthesis)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from services.dynamic_personas import (
    DynamicPersona,
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


@dataclass
class SwarmIgniteResult:
    messages: list[dict[str, str]]
    evidence: list[EvidenceItem]
    agent_profiles: list[dict[str, Any]]
    manager_synthesis: ManagerSynthesis


async def run_swarm_ignite(swarm_id: str, premise: str) -> SwarmIgniteResult:
    """
    Execute web search, dynamic persona generation, parallel debate, and manager consensus.
    """
    client = get_client()
    trimmed_premise = premise.strip()

    logger.info("Starting orchestration for swarm %s", swarm_id)

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
    )

    logger.info(
        "Swarm %s personas: %s",
        swarm_id,
        ", ".join(p["role"] for p in personas),
    )

    agent_tasks = [
        get_persona_debate_response(
            client,
            persona,
            trimmed_premise,
            web_data,
        )
        for persona in personas
    ]

    agent_results = await asyncio.gather(*agent_tasks)
    agent_roles = [msg["role"] for msg in agent_results]

    combined_perspectives = "\n\n".join(
        f"{msg['role']} ({personas[i]['name']}):\n{msg['text']}"
        for i, msg in enumerate(agent_results)
    )

    manager_synthesis = await get_manager_synthesis(
        client,
        trimmed_premise,
        web_data,
        combined_perspectives,
        agent_roles,
    )

    manager_result = {
        "role": "Manager",
        "text": manager_synthesis.consensus,
    }

    agent_profiles = [persona_to_agent_profile(persona) for persona in personas]

    logger.info(
        "Swarm %s complete: %s agent messages + manager (sentiments=%s)",
        swarm_id,
        len(agent_results),
        manager_synthesis.agent_sentiments,
    )

    return SwarmIgniteResult(
        messages=[*agent_results, manager_result],
        evidence=evidence,
        agent_profiles=agent_profiles,
        manager_synthesis=manager_synthesis,
    )
