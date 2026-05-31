"""Multi-agent swarm orchestration (parallel human vectors + manager synthesis)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from services.agent_profiles import AgentProfile, generate_agent_profiles
from services.llm import get_agent_response, get_client
from services.scraper import EvidenceItem, scrape_for_premise

logger = logging.getLogger(__name__)

MANAGER_SYSTEM = (
    "You are the Manager. Read the live web data and the 5 human agent "
    "perspectives, then provide a final 2-sentence definitive, data-backed "
    "consensus."
)

PERSONAS: list[tuple[str, str]] = [
    (
        "Budget-Conscious Buyer",
        "You are a highly frugal consumer. Focus strictly on price, "
        "maintenance costs, and ROI using the web data.",
    ),
    (
        "Performance Enthusiast",
        "You care only about specs, speed, tech, and premium features. "
        "Argue for the highest quality option using the web data.",
    ),
    (
        "Safety & Practicality Parent",
        "You are risk-averse. Focus on safety ratings, reliability, and "
        "everyday usability using the web data.",
    ),
    (
        "Brand Status Fanboy",
        "You care about luxury, brand perception, and social status. "
        "Argue based on prestige using the web data.",
    ),
    (
        "Skeptical Mechanic",
        "You are a cynical expert. Look for flaws, recalls, or hidden issues "
        "in the web data to warn the user.",
    ),
]


@dataclass
class SwarmIgniteResult:
    messages: list[dict[str, str]]
    evidence: list[EvidenceItem]
    agent_profiles: list[AgentProfile]


async def run_swarm_ignite(swarm_id: str, premise: str) -> SwarmIgniteResult:
    """
    Execute web search, parallel agent debate, and manager consensus.
    Returns debate messages plus structured evidence from the scraper.
    """
    client = get_client()
    trimmed_premise = premise.strip()

    logger.info("Starting parallel orchestration for swarm %s", swarm_id)

    web_data, evidence = await asyncio.to_thread(scrape_for_premise, trimmed_premise)
    logger.debug(
        "Web data preview for swarm %s: %s",
        swarm_id,
        web_data[:500],
    )

    agent_tasks = [
        get_agent_response(
            client,
            role_name,
            instruction,
            trimmed_premise,
            web_data,
        )
        for role_name, instruction in PERSONAS
    ]

    agent_results, agent_profiles = await asyncio.gather(
        asyncio.gather(*agent_tasks),
        generate_agent_profiles(client, trimmed_premise, web_data),
    )

    combined_perspectives = "\n\n".join(
        f"{msg['role']}:\n{msg['text']}" for msg in agent_results
    )
    manager_user = (
        f"User premise:\n{trimmed_premise}\n\n"
        f"Live web data:\n{web_data}\n\n"
        f"Human agent perspectives:\n{combined_perspectives}"
    )

    manager_result = await get_agent_response(
        client,
        "Manager",
        MANAGER_SYSTEM,
        manager_user,
        web_data,
    )

    logger.info(
        "Swarm %s complete: %s agent messages + manager",
        swarm_id,
        len(agent_results),
    )

    return SwarmIgniteResult(
        messages=[*agent_results, manager_result],
        evidence=evidence,
        agent_profiles=agent_profiles,
    )
