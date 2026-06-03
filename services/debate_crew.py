"""
Production debate pipeline — raw LangChain + OpenRouter (no CrewAI LLM layer).

Three sequential LLM calls: research → skeptic → synthesis (JSON executive report).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from models import AgentStance, DebateCompletionPayload
from pydantic import ValidationError

from services.debate_constants import AI_MODEL_ERROR_VERDICT
from services.openrouter_llm import get_default_llm, invoke_llm, log_langchain_error
from services.scraper import EvidenceItem, scrape_for_premise

logger = logging.getLogger(__name__)

AGENT_NAMES = ("Market Researcher", "Skeptical Debater", "CEO Synthesizer")

CEO_JSON_SPEC = """
Return ONLY valid JSON (no markdown fences, no commentary) matching this exact schema:

{
  "verdict": "<2-4 sentence executive verdict>",
  "confidence": <integer 0-100>,
  "tldr": [
    "<bullet 1: why this verdict — decisive reason>",
    "<bullet 2: key risk or constraint>",
    "<bullet 3: what must be true for success>"
  ],
  "friction_matrix": [
    {
      "name": "Market Researcher",
      "stance": "AGREES",
      "argument": "<1-2 sentences from research perspective>"
    },
    {
      "name": "Skeptical Debater",
      "stance": "DISAGREES",
      "argument": "<1-2 sentences challenging the thesis>"
    },
    {
      "name": "CEO Synthesizer",
      "stance": "NEUTRAL",
      "argument": "<1-2 sentences balancing both sides>"
    }
  ],
  "pre_mortem": {
    "failure_modes": [
      "<how this decision fails in 12 months — mode 1>",
      "<mode 2>",
      "<mode 3>"
    ],
    "critical_unknowns": [
      "<data the swarm could not verify — unknown 1>",
      "<unknown 2>",
      "<unknown 3>"
    ]
  },
  "execution_roadmap": {
    "immediate_action": "<specific action for the next 48 hours>",
    "plan_b": "<credible alternative if the primary path fails>"
  },
  "agents": [
    {"name": "Market Researcher", "position": "<one sentence>"},
    {"name": "Skeptical Debater", "position": "<one sentence>"},
    {"name": "CEO Synthesizer", "position": "<one sentence>"}
  ]
}

Rules:
- stance must be exactly AGREES, DISAGREES, or NEUTRAL (uppercase).
- pre_mortem.failure_modes: 3-5 concrete failure scenarios grounded in the debate and query domain.
- pre_mortem.critical_unknowns: 3-5 gaps where evidence was insufficient.
- tldr must have exactly 3 strings.
- Match the vocabulary and time horizon of the user's query (history, politics, consumer purchase, science, career, etc.).

CONTEXT — execution_roadmap (CRITICAL):
- immediate_action and plan_b MUST be highly specific to the user's exact query.
- If the query is about history or politics: suggest relevant primary sources, archives, historians, or policy documents to consult — not business metrics.
- If the query is a consumer purchase: suggest checking specific review sites, return policies, or comparison benchmarks — not fundraising or ICPs.
- If the query is academic or scientific: suggest papers, datasets, or replication checks — not GTM sprints.
- DO NOT use generic business/SaaS jargon (e.g. "48-hour sprints", "ICP", "CAC", "Series A", "wedge offer")
  unless the query is explicitly about launching or scaling a B2B startup.
- Cite concrete next steps a real person could take in the next 48 hours for THIS question.
"""


class DebateAgent(TypedDict):
    name: str
    position: str


class DebateResult(TypedDict):
    verdict: str
    confidence: int
    agents: list[DebateAgent]
    tldr: list[str]
    friction_matrix: list[dict[str, str]]
    pre_mortem: dict[str, list[str]]
    execution_roadmap: dict[str, str]
    evidence: list[dict[str, str]]


def ensure_verdict(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        print("[debate_crew] empty verdict -> AI_MODEL_ERROR_VERDICT")
        return AI_MODEL_ERROR_VERDICT
    return cleaned


def _default_pre_mortem() -> dict[str, list[str]]:
    return {
        "failure_modes": [
            "Demand assumptions prove optimistic within two quarters.",
            "Acquisition costs exceed modeled payback under competition.",
            "Regulatory or supply-chain friction delays launch.",
        ],
        "critical_unknowns": [
            "Verified willingness-to-pay at scale in the target segment.",
            "True customer acquisition cost under the current channel mix.",
            "Regulatory exposure in priority geographies.",
        ],
    }


def _default_execution_roadmap(query: str = "") -> dict[str, str]:
    trimmed = (query or "").strip()
    if trimmed:
        return {
            "immediate_action": (
                f"List the three strongest claims about “{trimmed[:80]}” and verify each "
                "against the live web sources provided in the research block."
            ),
            "plan_b": (
                "If verification fails or sources conflict, narrow the question "
                "(time period, geography, or scope) and re-run deliberation."
            ),
        }
    return {
        "immediate_action": (
            "Verify the top three claims in the verdict against the cited live web sources."
        ),
        "plan_b": (
            "If sources conflict, narrow the question and re-run deliberation."
        ),
    }


def _default_friction_matrix() -> list[dict[str, str]]:
    return [
        {
            "name": "Market Researcher",
            "stance": AgentStance.AGREES.value,
            "argument": "Market signals support proceeding with disciplined execution.",
        },
        {
            "name": "Skeptical Debater",
            "stance": AgentStance.DISAGREES.value,
            "argument": "Competitive and unit-economic risks may erode returns within 12 months.",
        },
        {
            "name": "CEO Synthesizer",
            "stance": AgentStance.NEUTRAL.value,
            "argument": "Proceed only with gated KPIs and pre-defined downside triggers.",
        },
    ]


def _format_evidence_for_prompt(items: list[EvidenceItem]) -> str:
    if not items:
        return "No live web sources were retrieved (Tavily unavailable or returned no results)."
    lines: list[str] = []
    for index, item in enumerate(items[:10], start=1):
        title = (item.get("title") or "Untitled").strip()
        source = (item.get("source") or "Web").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()[:400]
        lines.append(
            f"{index}. [{source}] {title}\n   URL: {url}\n   Excerpt: {snippet}",
        )
    return "\n".join(lines)


def _evidence_for_webhook(items: list[EvidenceItem]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        url = (item.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        title = (item.get("title") or "").strip() or url
        source = (item.get("source") or "").strip() or "Web"
        snippet = (item.get("snippet") or "").strip() or title[:500]
        rows.append(
            {
                "title": title[:300],
                "source": source[:120],
                "url": url[:2000],
                "snippet": snippet[:1800],
            },
        )
    return rows


def fallback_debate_result(reason: str | None = None, query: str = "") -> DebateResult:
    if reason:
        print(f"[debate_crew] FALLBACK reason={reason[:400]}")
    return {
        "verdict": AI_MODEL_ERROR_VERDICT,
        "confidence": 0,
        "agents": [
            {"name": name, "position": "Deliberation did not complete."}
            for name in AGENT_NAMES
        ],
        "tldr": [
            "The AI model could not complete synthesis.",
            "No reliable risk assessment was produced.",
            "Retry the debate with a shorter or clearer query.",
        ],
        "friction_matrix": _default_friction_matrix(),
        "pre_mortem": _default_pre_mortem(),
        "execution_roadmap": _default_execution_roadmap(query),
        "evidence": [],
    }


def finalize_debate_result(result: DebateResult) -> DebateResult:
    verdict = ensure_verdict(result.get("verdict"))
    agents = list(result.get("agents") or [])
    if not agents:
        agents = [{"name": "CEO Synthesizer", "position": verdict[:500]}]

    confidence_raw = result.get("confidence", 0)
    confidence = (
        int(max(0, min(100, round(float(confidence_raw)))))
        if isinstance(confidence_raw, (int, float))
        else 0
    )

    tldr = list(result.get("tldr") or [])
    if len(tldr) < 3:
        tldr = fallback_debate_result()["tldr"]

    friction = list(result.get("friction_matrix") or [])
    if not friction:
        friction = _default_friction_matrix()

    pre_mortem = result.get("pre_mortem") or _default_pre_mortem()
    execution = result.get("execution_roadmap") or _default_execution_roadmap()
    evidence = list(result.get("evidence") or [])

    return {
        "verdict": verdict,
        "confidence": confidence,
        "agents": agents,
        "tldr": tldr[:5],
        "friction_matrix": friction,
        "pre_mortem": pre_mortem,
        "execution_roadmap": execution,
        "evidence": evidence,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _coerce_completion_payload(parsed: dict[str, Any]) -> DebateCompletionPayload | None:
    try:
        return DebateCompletionPayload.model_validate(parsed)
    except ValidationError as exc:
        print(f"[debate_crew] Pydantic validation failed: {exc}")
        return None


def _payload_to_result(payload: DebateCompletionPayload) -> DebateResult:
    return {
        "verdict": payload.verdict.strip(),
        "confidence": payload.confidence,
        "agents": [
            {"name": agent.name, "position": agent.position}
            for agent in payload.agents
        ],
        "tldr": list(payload.tldr),
        "friction_matrix": [
            {
                "name": entry.name,
                "stance": entry.stance.value,
                "argument": entry.argument,
            }
            for entry in payload.friction_matrix
        ],
        "pre_mortem": {
            "failure_modes": list(payload.pre_mortem.failure_modes),
            "critical_unknowns": list(payload.pre_mortem.critical_unknowns),
        },
        "execution_roadmap": {
            "immediate_action": payload.execution_roadmap.immediate_action,
            "plan_b": payload.execution_roadmap.plan_b,
        },
    }


def _build_result_from_partial(
    parsed: dict[str, Any],
    research: str,
    debate: str,
) -> DebateResult | None:
    """Best-effort assembly when full Pydantic validation fails."""
    verdict = str(parsed.get("verdict") or "").strip()
    if not verdict:
        return None

    confidence_raw = parsed.get("confidence", 50)
    confidence = (
        int(max(0, min(100, round(float(confidence_raw)))))
        if isinstance(confidence_raw, (int, float))
        else 50
    )

    tldr_raw = parsed.get("tldr")
    tldr = (
        [str(item).strip() for item in tldr_raw if str(item).strip()]
        if isinstance(tldr_raw, list)
        else []
    )

    friction_raw = parsed.get("friction_matrix") or parsed.get("frictionMatrix")
    friction_matrix: list[dict[str, str]] = []
    if isinstance(friction_raw, list):
        for index, item in enumerate(friction_raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or AGENT_NAMES[min(index, 2)]).strip()
            stance = str(item.get("stance") or "NEUTRAL").strip().upper()
            if stance not in ("AGREES", "DISAGREES", "NEUTRAL"):
                stance = "NEUTRAL"
            argument = str(
                item.get("argument") or item.get("summary") or "",
            ).strip()
            if name and argument:
                friction_matrix.append(
                    {"name": name, "stance": stance, "argument": argument[:500]},
                )

    pre_raw = parsed.get("pre_mortem") or parsed.get("preMortem")
    pre_mortem = _default_pre_mortem()
    if isinstance(pre_raw, dict):
        fm = pre_raw.get("failure_modes") or pre_raw.get("failureModes")
        cu = pre_raw.get("critical_unknowns") or pre_raw.get("criticalUnknowns")
        if isinstance(fm, list) and isinstance(cu, list):
            failure_modes = [str(x).strip() for x in fm if str(x).strip()]
            critical_unknowns = [str(x).strip() for x in cu if str(x).strip()]
            if failure_modes and critical_unknowns:
                pre_mortem = {
                    "failure_modes": failure_modes[:8],
                    "critical_unknowns": critical_unknowns[:8],
                }

    road_raw = parsed.get("execution_roadmap") or parsed.get("executionRoadmap")
    execution_roadmap = _default_execution_roadmap()
    if isinstance(road_raw, dict):
        immediate = str(
            road_raw.get("immediate_action") or road_raw.get("immediateAction") or "",
        ).strip()
        plan_b = str(road_raw.get("plan_b") or road_raw.get("planB") or "").strip()
        if immediate and plan_b:
            execution_roadmap = {
                "immediate_action": immediate[:1000],
                "plan_b": plan_b[:1000],
            }

    agents_raw = parsed.get("agents")
    agents: list[DebateAgent] = []
    if isinstance(agents_raw, list) and agents_raw:
        for index, item in enumerate(agents_raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or AGENT_NAMES[index]).strip()
            position = str(item.get("position") or item.get("stance") or "").strip()
            agents.append(
                {
                    "name": name or AGENT_NAMES[index],
                    "position": position or "No position recorded.",
                },
            )
    else:
        agents = [
            {"name": AGENT_NAMES[0], "position": research[:500]},
            {"name": AGENT_NAMES[1], "position": debate[:500]},
            {"name": AGENT_NAMES[2], "position": verdict[:500]},
        ]

    if len(tldr) < 3:
        tldr = [
            verdict[:200] if verdict else "Verdict synthesized from swarm debate.",
            "Key risks were raised by the skeptical debater.",
            "Validate assumptions before committing capital.",
        ]

    if not friction_matrix:
        friction_matrix = _default_friction_matrix()

    return finalize_debate_result(
        {
            "verdict": verdict,
            "confidence": confidence,
            "agents": agents,
            "tldr": tldr[:5],
            "friction_matrix": friction_matrix,
            "pre_mortem": pre_mortem,
            "execution_roadmap": execution_roadmap,
        },
    )


def _build_result(
    research: str,
    debate: str,
    synthesis: str,
) -> DebateResult:
    final_text = synthesis or debate or research
    parsed = _extract_json_object(final_text)

    if parsed:
        completion = _coerce_completion_payload(parsed)
        if completion:
            return finalize_debate_result(_payload_to_result(completion))

        partial = _build_result_from_partial(parsed, research, debate)
        if partial:
            return partial

    if not final_text.strip():
        return fallback_debate_result("Empty synthesis")

    return fallback_debate_result("Could not parse CEO JSON")


def run_debate_crew(
    query: str,
    *,
    model_mix: float = 0,
    web_context: str | None = None,
    evidence_items: list[EvidenceItem] | None = None,
) -> DebateResult:
    """
    Run 3-stage debate via direct OpenRouter ChatOpenAI calls.
    model_mix is accepted for API compatibility but ignored (single model only).
    """
    _ = model_mix
    print("[debate_crew] === START model=deepseek/deepseek-chat ===")

    trimmed = (query or "").strip()
    if not trimmed:
        return fallback_debate_result("Missing query")

    if web_context is None or evidence_items is None:
        print("[debate_crew] running Tavily live web research")
        web_context, evidence_items = scrape_for_premise(trimmed)

    evidence_rows = _evidence_for_webhook(evidence_items or [])
    research_block = _format_evidence_for_prompt(evidence_items or [])
    print(f"[debate_crew] Tavily sources={len(evidence_rows)}")

    try:
        llm = get_default_llm()
    except Exception as exc:
        logger.exception("LLM setup failed")
        print(f"[debate_crew] get_default_llm failed: {exc}")
        return finalize_debate_result(
            {**fallback_debate_result(str(exc), trimmed), "evidence": evidence_rows},
        )

    system_base = (
        "You are part of Shoal AI's institutional debate desk. "
        "Ground every claim in the user's question and the live web research provided. "
        "Be specific and concise. Never return an empty response."
    )

    try:
        print("[debate_crew] stage 1/3 Market Researcher")
        research = invoke_llm(
            llm,
            system_base,
            f"User query:\n{trimmed}\n\n"
            f"LIVE WEB RESEARCH (Tavily):\n{research_block}\n\n"
            f"Research digest:\n{web_context[:6000]}\n\n"
            "As Market Researcher, cite specific sources by name/URL where possible. "
            "List 5-8 facts, 3 assumptions, 2-3 risks tied to this exact query.",
            stage="research",
        )

        print("[debate_crew] stage 2/3 Skeptical Debater")
        debate = invoke_llm(
            llm,
            system_base,
            f"User query:\n{trimmed}\n\n"
            f"LIVE WEB RESEARCH:\n{research_block}\n\n"
            f"Researcher output:\n{research}\n\n"
            "As Skeptical Debater, challenge 3 claims using the web evidence. "
            "List key risks specific to this query domain.",
            stage="debate",
        )

        print("[debate_crew] stage 3/3 CEO Synthesizer (JSON executive report)")
        synthesis = invoke_llm(
            llm,
            system_base,
            f"User query:\n{trimmed}\n\n"
            f"LIVE WEB RESEARCH:\n{research_block}\n\n"
            f"Researcher output:\n{research}\n\n"
            f"Skeptic output:\n{debate}\n\n"
            "As CEO Synthesizer, produce the final executive decision report.\n"
            "pre_mortem and execution_roadmap MUST match the query domain — "
            "see CONTEXT rules in the schema.\n"
            f"{CEO_JSON_SPEC}",
            stage="synthesis",
        )

        result = finalize_debate_result(_build_result(research, debate, synthesis))
        result["evidence"] = evidence_rows
        print(
            f"[debate_crew] === SUCCESS verdict_len={len(result['verdict'])} "
            f"confidence={result['confidence']} tldr={len(result['tldr'])} "
            f"evidence={len(evidence_rows)} ===",
        )
        return result

    except Exception as exc:
        log_langchain_error(exc, stage="debate_pipeline")
        logger.exception("Debate pipeline failed")
        print(f"[debate_crew] pipeline exception: {exc}")
        return finalize_debate_result(
            {**fallback_debate_result(str(exc), trimmed), "evidence": evidence_rows},
        )
