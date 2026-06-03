"""
Headless MiroFish-style adapter for Shoal debates.

Inspired by MiroFish's asyncio.gather + semaphore concurrency (see
mirofish_core/backend/scripts/run_parallel_simulation.py) but WITHOUT:
- OASIS social simulation environments
- Zep graph memory / vector DB
- Frontend or project database

Token Governor: exactly 2 LLM turns
  Turn 0 (pre-flight) — mandatory Tavily deep research (blocking, no workers yet)
  Turn 1 — N stateless workers in parallel (one argument each, then exit)
  Turn 2 — CEO synthesizes executive JSON for the webhook
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from services.debate_result_codec import (
    ANTI_HALLUCINATION_RULE,
    DebateResult,
    agents_from_workers,
    build_ceo_json_spec,
    evidence_for_webhook,
    fallback_debate_result,
    finalize_debate_result,
    friction_matrix_from_workers,
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
from services.scraper import EvidenceItem, mandatory_tavily_deep_research

logger = logging.getLogger(__name__)

MAX_PARALLEL_LLM = 50
MAX_WORKER_LLM_CALLS = 50


@dataclass(frozen=True)
class WorkerArgument:
    persona_id: int
    name: str
    role: str
    stance_label: str
    argument: str


def _effective_worker_count(agent_count: int) -> int:
    return max(1, min(int(agent_count), MAX_WORKER_LLM_CALLS))


def build_stateless_personas(premise: str, agent_count: int) -> list[DynamicPersona]:
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


def _worker_system_prompt(persona: DynamicPersona, web_research_raw: str) -> str:
    return (
        "You are a STATELESS Shoal AI debate worker. You receive one prompt, "
        "output exactly ONE adversarial argument, then stop.\n"
        "You have NO memory, NO tools, and NO ability to chat with other agents.\n"
        "Do not hedge into both sides — prosecute your assigned stance aggressively.\n\n"
        f"{ANTI_HALLUCINATION_RULE}\n\n"
        "=== MANDATORY TAVILY WEB SEARCH CONTEXT (read-only) ===\n"
        f"{web_research_raw}\n"
        "=== END WEB SEARCH CONTEXT ===\n\n"
        f"NAME: {persona['name']}\n"
        f"ROLE: {persona['role']}\n"
        f"ASSIGNED STANCE: {persona.get('adversarial_stance', '')}\n"
        f"DEBATE MANDATE: {persona['debate_instruction']}\n"
        f"BACKSTORY: {persona.get('backstory', '')}\n"
        f"BIASES: {persona.get('biases', '')}\n"
        f"RISK TOLERANCE: {persona.get('riskTolerance', 'Medium')}\n"
        f"LOCATION: {persona.get('location', '')}\n"
    )


def _worker_user_prompt(query: str) -> str:
    return (
        f"USER QUERY:\n{query}\n\n"
        "Using ONLY the Tavily web search context in your system message, "
        "write exactly ONE argument in 3-5 sentences.\n"
        "Cite specific sources or URLs when possible.\n"
        "If the web context is empty or inconclusive, say so — do not invent facts.\n"
        "Do NOT output JSON. Do NOT address other agents. Do NOT ask questions."
    )


def _ceo_system_prompt(web_research_raw: str) -> str:
    return (
        "You are the CEO Synthesizer for Shoal AI. You receive the full panel "
        "of stateless worker arguments plus mandatory Tavily web research.\n"
        "Produce the final institutional executive report as JSON only.\n"
        "You do NOT debate further and do NOT call tools.\n\n"
        f"{ANTI_HALLUCINATION_RULE}\n\n"
        "=== MANDATORY TAVILY WEB SEARCH CONTEXT (read-only) ===\n"
        f"{web_research_raw}\n"
        "=== END WEB SEARCH CONTEXT ===\n"
    )


async def _invoke_worker(
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    persona: DynamicPersona,
    query: str,
    web_research_raw: str,
) -> WorkerArgument:
    stage = f"worker_{persona['id']}"
    system = _worker_system_prompt(persona, web_research_raw)
    user = _worker_user_prompt(query)

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
                f"upstream model error ({type(exc).__name__}). "
                "Live web data was not used."
            )

    argument = (text or "").strip()[:1200]
    if not argument:
        argument = (
            f"{persona['name']} ({persona['role']}) cannot argue: Tavily returned "
            "insufficient data for this query."
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
    web_research_raw: str,
) -> list[WorkerArgument]:
    semaphore = asyncio.Semaphore(MAX_PARALLEL_LLM)
    tasks = [
        _invoke_worker(llm, semaphore, persona, query, web_research_raw)
        for persona in personas
    ]
    print(
        f"[mirofish_adapter] Turn 1: {len(tasks)} workers "
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
    web_research_raw: str,
    workers: list[WorkerArgument],
) -> str:
    digest = _format_worker_digest(workers)
    worker_count = len(workers)
    system = _ceo_system_prompt(web_research_raw)
    user = (
        f"USER QUERY:\n{query}\n\n"
        f"WORKER ARGUMENTS ({worker_count} agents — read-only, no agent chat):\n"
        f"{digest}\n\n"
        f"friction_matrix MUST contain EXACTLY {worker_count} entries "
        f"(one per worker above, same names).\n"
        f"agents MUST contain EXACTLY {worker_count} entries.\n"
        "Synthesize verdict, pre_mortem, execution_roadmap, and tldr from workers "
        "and Tavily context only.\n"
        f"{build_ceo_json_spec(worker_count)}"
    )

    async with asyncio.Semaphore(1):
        return await asyncio.to_thread(
            invoke_llm,
            llm,
            system,
            user,
            stage="ceo_synthesis",
        )


def _apply_worker_panel(result: DebateResult, workers: list[WorkerArgument]) -> DebateResult:
    """Ensure every spawned worker appears in friction_matrix and agents."""
    result["friction_matrix"] = friction_matrix_from_workers(workers)
    result["agents"] = agents_from_workers(workers)
    return result


async def run_debate_swarm_async(
    query: str,
    *,
    agent_count: int = 3,
    web_context: str | None = None,
    evidence_items: list[EvidenceItem] | None = None,
) -> DebateResult:
    trimmed = (query or "").strip()
    if not trimmed:
        return fallback_debate_result("Missing query")

    _ = web_context  # ignored — always run mandatory Tavily

    print("[mirofish_adapter] MANDATORY Tavily deep research (pre Turn 1)")
    web_research_raw, evidence_items = await asyncio.to_thread(
        mandatory_tavily_deep_research,
        trimmed,
    )

    evidence_rows = evidence_for_webhook(evidence_items or [])
    worker_total = _effective_worker_count(agent_count)

    print(
        f"[mirofish_adapter] research_chars={len(web_research_raw)} "
        f"evidence_urls={len(evidence_rows)} workers={worker_total}",
    )

    if not evidence_rows:
        print(
            "[mirofish_adapter] WARNING: zero Tavily URLs — "
            "workers/CEO must not hallucinate",
        )

    try:
        llm = get_default_llm()
    except Exception as exc:
        logger.exception("LLM setup failed")
        return finalize_debate_result(
            {
                **fallback_debate_result(str(exc), trimmed),
                "evidence": evidence_rows,
            },
        )

    personas = build_stateless_personas(trimmed, agent_count)

    try:
        workers = await _run_turn1_workers(
            llm,
            personas,
            trimmed,
            web_research_raw,
        )
        print(f"[mirofish_adapter] Turn 1 complete: {len(workers)} arguments")

        synthesis = await _run_turn2_ceo(
            llm,
            trimmed,
            web_research_raw,
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
        result = _apply_worker_panel(result, workers)

        print(
            f"[mirofish_adapter] SUCCESS verdict_len={len(result['verdict'])} "
            f"friction={len(result['friction_matrix'])} evidence={len(evidence_rows)}",
        )
        return result

    except Exception as exc:
        log_langchain_error(exc, stage="mirofish_adapter")
        logger.exception("MiroFish adapter pipeline failed")
        return finalize_debate_result(
            {
                **fallback_debate_result(str(exc), trimmed),
                "evidence": evidence_rows,
            },
        )
