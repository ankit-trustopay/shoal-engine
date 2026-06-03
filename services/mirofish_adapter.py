"""
Headless MiroFish-style adapter for Shoal debates.

Inspired by MiroFish's asyncio.gather + semaphore concurrency (see
mirofish_core/backend/scripts/run_parallel_simulation.py) but WITHOUT:
- OASIS social simulation environments
- Zep graph memory / vector DB
- Frontend or project database

Token Governor: exactly 2 LLM turns
  Turn 1 — N stateless workers in parallel (one argument each, then exit)
  Turn 2 — CEO synthesizes executive JSON for the webhook
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from services.debate_result_codec import (
    CEO_JSON_SPEC,
    DebateResult,
    evidence_for_webhook,
    fallback_debate_result,
    format_evidence_for_prompt,
    parse_ceo_json,
)
from services.dynamic_personas import (
    ADVERSARIAL_ARCHETYPES,
    DynamicPersona,
    build_adversarial_persona,
)
from services.openrouter_llm import (
    get_default_llm,
    invoke_llm,
    log_langchain_error,
)
from services.scraper import EvidenceItem, scrape_for_premise

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# MiroFish OASIS simulations use semaphore=30; Shoal debate cap is 50 per spec.
MAX_PARALLEL_LLM = 50
# Hard cap on Turn-1 worker LLM calls (API cost protection).
MAX_WORKER_LLM_CALLS = 50


@dataclass(frozen=True)
class WorkerArgument:
    """Stateless output from a single Turn-1 worker (no memory, no follow-up)."""

    persona_id: int
    name: str
    role: str
    stance_label: str
    argument: str


def _effective_worker_count(agent_count: int) -> int:
    return max(1, min(int(agent_count), MAX_WORKER_LLM_CALLS))


def build_stateless_personas(premise: str, agent_count: int) -> list[DynamicPersona]:
    """
  Materialize Shoal adversarial personas without extra LLM calls (cost-safe).
  Cycles archetypes when agent_count exceeds the archetype catalog.
    """
    count = _effective_worker_count(agent_count)
    personas: list[DynamicPersona] = []
    for index in range(count):
        archetype = ADVERSARIAL_ARCHETYPES[index % len(ADVERSARIAL_ARCHETYPES)]
        personas.append(build_adversarial_persona(archetype, index, premise))
    return personas


def _stance_label_from_persona(persona: DynamicPersona) -> str:
    stance = str(persona.get("adversarial_stance") or "").upper()
    if "AGAINST" in stance:
        return "DISAGREES"
    if "FOR" in stance or "PROSECUTE" in stance:
        return "AGREES"
    return "NEUTRAL"


def _worker_system_prompt(persona: DynamicPersona) -> str:
    return (
        "You are a STATELESS Shoal AI debate worker. You receive one prompt, "
        "output exactly ONE adversarial argument, then stop.\n"
        "You have NO memory, NO tools, and NO ability to chat with other agents.\n"
        "Do not hedge into both sides — prosecute your assigned stance aggressively.\n\n"
        f"NAME: {persona['name']}\n"
        f"ROLE: {persona['role']}\n"
        f"ASSIGNED STANCE: {persona.get('adversarial_stance', '')}\n"
        f"DEBATE MANDATE: {persona['debate_instruction']}\n"
        f"BACKSTORY: {persona.get('backstory', '')}\n"
        f"BIASES: {persona.get('biases', '')}\n"
        f"RISK TOLERANCE: {persona.get('riskTolerance', 'Medium')}\n"
        f"LOCATION: {persona.get('location', '')}\n"
    )


def _worker_user_prompt(
    query: str,
    research_block: str,
    web_digest: str,
) -> str:
    return (
        f"USER QUERY:\n{query}\n\n"
        f"LIVE WEB RESEARCH (Tavily — cite URLs when relevant):\n{research_block}\n\n"
        f"RESEARCH DIGEST:\n{web_digest[:5000]}\n\n"
        "Write exactly ONE argument in 3-5 sentences.\n"
        "Ground claims in the web research when possible.\n"
        "Do NOT output JSON. Do NOT address other agents. Do NOT ask questions."
    )


async def _invoke_worker(
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    persona: DynamicPersona,
    query: str,
    research_block: str,
    web_digest: str,
) -> WorkerArgument:
    """Turn 1: single stateless argument, then shutdown."""
    stage = f"worker_{persona['id']}"
    system = _worker_system_prompt(persona)
    user = _worker_user_prompt(query, research_block, web_digest)

    async with semaphore:
        try:
            text = await asyncio.to_thread(
                invoke_llm,
                llm,
                system,
                user,
                stage=stage,
            )
        except Exception as exc:
            log_langchain_error(exc, stage=stage)
            text = (
                f"{persona['name']} could not complete an argument due to an "
                f"upstream model error ({type(exc).__name__})."
            )

    argument = (text or "").strip()[:1200]
    if not argument:
        argument = (
            f"{persona['name']} ({persona['role']}) withheld a position pending "
            "clearer evidence on the query."
        )

    return WorkerArgument(
        persona_id=int(persona["id"]),
        name=str(persona["name"]),
        role=str(persona["role"]),
        stance_label=_stance_label_from_persona(persona),
        argument=argument,
    )


async def _run_turn1_workers(
    llm: ChatOpenAI,
    personas: list[DynamicPersona],
    query: str,
    research_block: str,
    web_digest: str,
) -> list[WorkerArgument]:
    """
  MiroFish-style parallel fan-out: asyncio.gather over all workers with a
  bounded semaphore to avoid OpenRouter rate limits.
    """
    semaphore = asyncio.Semaphore(MAX_PARALLEL_LLM)
    tasks = [
        _invoke_worker(llm, semaphore, persona, query, research_block, web_digest)
        for persona in personas
    ]
    print(
        f"[mirofish_adapter] Turn 1: spawning {len(tasks)} stateless workers "
        f"(semaphore={MAX_PARALLEL_LLM})",
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    workers: list[WorkerArgument] = []
    for index, result in enumerate(results):
        if isinstance(result, WorkerArgument):
            workers.append(result)
            continue
        persona = personas[index]
        logger.exception("Worker %s failed", persona.get("name"))
        workers.append(
            WorkerArgument(
                persona_id=int(persona["id"]),
                name=str(persona["name"]),
                role=str(persona["role"]),
                stance_label=_stance_label_from_persona(persona),
                argument=(
                    f"{persona['name']} failed to argue ({type(result).__name__})."
                ),
            ),
        )
    return workers


def _format_worker_digest(workers: list[WorkerArgument]) -> str:
    blocks: list[str] = []
    for worker in workers:
        blocks.append(
            f"### {worker.name} — {worker.role} [{worker.stance_label}]\n"
            f"{worker.argument}\n",
        )
    return "\n".join(blocks)


async def _run_turn2_ceo(
    llm: ChatOpenAI,
    query: str,
    research_block: str,
    web_digest: str,
    workers: list[WorkerArgument],
) -> str:
    """Turn 2: CEO reads all worker arguments and returns executive JSON."""
    digest = _format_worker_digest(workers)
    system = (
        "You are the CEO Synthesizer for Shoal AI. You receive the full panel "
        "of stateless worker arguments plus live web research.\n"
        "Produce the final institutional executive report as JSON only.\n"
        "You do NOT debate further and do NOT call tools."
    )
    user = (
        f"USER QUERY:\n{query}\n\n"
        f"LIVE WEB RESEARCH:\n{research_block}\n\n"
        f"RESEARCH DIGEST:\n{web_digest[:6000]}\n\n"
        f"WORKER ARGUMENTS (Turn 1 — read-only, no agent chat):\n{digest}\n\n"
        "Synthesize verdict, friction_matrix (one row per worker), pre_mortem, "
        "execution_roadmap, tldr, and agents.\n"
        f"{CEO_JSON_SPEC}"
    )

    semaphore = asyncio.Semaphore(1)
    async with semaphore:
        return await asyncio.to_thread(
            invoke_llm,
            llm,
            system,
            user,
            stage="ceo_synthesis",
        )


async def run_debate_swarm_async(
    query: str,
    *,
    agent_count: int = 3,
    web_context: str | None = None,
    evidence_items: list[EvidenceItem] | None = None,
) -> DebateResult:
    """
  Execute the 2-turn MiroFish adapter pipeline.
  Exactly 1 + N LLM calls (CEO + workers), no loops.
    """
    trimmed = (query or "").strip()
    if not trimmed:
        return fallback_debate_result("Missing query")

    if web_context is None or evidence_items is None:
        print("[mirofish_adapter] Tavily live web research")
        web_context, evidence_items = scrape_for_premise(trimmed)

    evidence_rows = evidence_for_webhook(evidence_items or [])
    research_block = format_evidence_for_prompt(evidence_items or [])
    worker_total = _effective_worker_count(agent_count)

    print(
        f"[mirofish_adapter] START workers={worker_total} "
        f"sources={len(evidence_rows)}",
    )

    try:
        llm = get_default_llm()
    except Exception as exc:
        logger.exception("LLM setup failed")
        return {
            **fallback_debate_result(str(exc), trimmed),
            "evidence": evidence_rows,
        }

    personas = build_stateless_personas(trimmed, agent_count)

    try:
        workers = await _run_turn1_workers(
            llm,
            personas,
            trimmed,
            research_block,
            web_context or "",
        )
        print(f"[mirofish_adapter] Turn 1 complete: {len(workers)} arguments")

        synthesis = await _run_turn2_ceo(
            llm,
            trimmed,
            research_block,
            web_context or "",
            workers,
        )
        print(f"[mirofish_adapter] Turn 2 complete: ceo_chars={len(synthesis)}")

        digest = _format_worker_digest(workers)
        result = parse_ceo_json(
            synthesis,
            worker_digest=digest,
            query=trimmed,
        )
        result["evidence"] = evidence_rows

        if not result.get("agents"):
            result["agents"] = [
                {"name": w.name, "position": w.argument[:500]} for w in workers
            ]

        print(
            f"[mirofish_adapter] SUCCESS verdict_len={len(result['verdict'])} "
            f"confidence={result['confidence']}",
        )
        return result

    except Exception as exc:
        log_langchain_error(exc, stage="mirofish_adapter")
        logger.exception("MiroFish adapter pipeline failed")
        return {
            **fallback_debate_result(str(exc), trimmed),
            "evidence": evidence_rows,
        }
