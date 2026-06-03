"""
Canonical debate result parsing, validation, and fallbacks for webhook payloads.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from models import (
    AgentStance,
    BoardroomSummary,
    DebateCompletionPayload,
    DebateRoomAgent,
    EvidenceVault,
    EvidenceVaultCitation,
    EvidenceVaultClusters,
    EvidenceVaultStats,
    ExecutiveSummary,
)
from pydantic import ValidationError

from services.debate_constants import AI_MODEL_ERROR_VERDICT
from services.scraper import EvidenceItem

AGENT_NAMES = ("Market Researcher", "Skeptical Debater", "CEO Synthesizer")

ANTI_HALLUCINATION_RULE = (
    "You are a data-driven analyst. You MUST base your arguments ONLY on the "
    "provided web search context. Do not invent products, features, or future "
    "models (e.g., do not guess 'Series 11' unless it is explicitly in the search "
    "results). If the data is inconclusive, state that explicitly."
)


def build_ceo_json_spec(worker_count: int) -> str:
    count = max(1, int(worker_count))
    return f"""
Return ONLY valid JSON (no markdown fences, no commentary) matching this exact schema:

{{
  "verdict": "<2-4 sentence executive verdict>",
  "confidence": <integer 0-100>,
  "executive_summary": {{
    "recommendation": "BUY",
    "fitForYou": "Excellent",
    "oneLineReason": "Because <X, Y, Z grounded in workers + Tavily>"
  }},
  "boardroom_summary": {{
    "bullCase": "<strongest bull case from AGREES workers + Tavily>",
    "bearCase": "<strongest bear / pre-mortem risk from DISAGREES workers>",
    "shoalRecommendation": "<synthesized middle-ground verdict>",
    "mainOpportunity": "<single clearest upside>",
    "mainRisk": "<single clearest downside>",
    "hiddenTradeoff": "<non-obvious tradeoff the swarm surfaced>",
    "bestAlternative": "<credible Plan B path>",
    "explanation": "<exactly 2 sentences: why this matters for the user query>"
  }},
  "debate_room": [
    {{
      "role": "<boardroom role e.g. Product Analyst, Skeptic, Budget Buyer>",
      "conclusion": "<this worker's conclusion in 2-3 sentences>",
      "disagreement": "<what they disagreed with another seat on>",
      "mindChanged": "Moved from YES to MAYBE after reviewing pricing data"
    }}
  ],
  "evidence_vault": {{
    "stats": {{
      "totalSources": <integer — approximate Tavily corpus size reviewed>,
      "highSignal": <integer — high-relevance sources>,
      "contradictory": <integer — sources with conflicting claims>,
      "dominantConsensus": <0 or 1 — 1 if one narrative clearly dominates>
    }},
    "clusters": {{
      "reddit": [{{"title": "...", "url": "http...", "source": "...", "snippet": "..."}}],
      "youtube": [],
      "official": [],
      "news": []
    }}
  }},
  "tldr": [
    "<bullet 1: why this verdict — decisive reason>",
    "<bullet 2: key risk or constraint>",
    "<bullet 3: what must be true for success>"
  ],
  "friction_matrix": [
    {{
      "name": "<worker name>",
      "stance": "AGREES",
      "argument": "<1-2 sentence summary of that worker's argument>"
    }}
  ],
  "pre_mortem": {{
    "failure_modes": ["<mode 1>", "<mode 2>", "<mode 3>"],
    "critical_unknowns": ["<unknown 1>", "<unknown 2>", "<unknown 3>"]
  }},
  "execution_roadmap": {{
    "immediate_action": "<specific next step for this exact query>",
    "plan_b": "<credible alternative if the primary path fails>"
  }},
  "agents": [
    {{"name": "<worker name>", "position": "<one sentence>"}}
  ]
}}

Rules:
- executive_summary.recommendation MUST be exactly BUY, WAIT, or PIVOT.
- executive_summary.fitForYou MUST be exactly Excellent, Good, or Weak.
- boardroom_summary: bullCase and bearCase MUST be derived from worker friction (not generic).
- boardroom_summary.hiddenTradeoff MUST reflect a real tension between workers or Tavily sources.
- debate_room: EXACTLY {count} entries — one per worker below (unique boardroom roles).
- debate_room.mindChanged: realistic stance shift tied to Tavily or a peer argument.
- evidence_vault.clusters: assign EVERY Tavily URL from the system context to exactly one cluster:
  reddit (reddit.com), youtube (youtube.com / youtu.be), official (.gov, docs, filings, github),
  news (everything else: blogs, press, reviews).
- friction_matrix: EXACTLY {count} entries — one per worker (same names, no merging).
- agents: EXACTLY {count} entries — one per worker (name + one-sentence position).
- stance must be exactly AGREES, DISAGREES, or NEUTRAL (uppercase).
- tldr must have exactly 3 strings.
- Match vocabulary to the user's query (history, politics, purchase, science, career, etc.).

CONTEXT — execution_roadmap (CRITICAL):
- immediate_action and plan_b MUST be highly specific to the user's exact query.
- DO NOT use generic SaaS jargon unless the query is explicitly about a B2B startup launch.

{ANTI_HALLUCINATION_RULE}
"""


# Backward-compatible default (3 workers)
CEO_JSON_SPEC = build_ceo_json_spec(3)


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
    executive_summary: dict[str, str]
    boardroom_summary: dict[str, str]
    debate_room: list[dict[str, str]]
    evidence_vault: dict[str, Any]


def ensure_verdict(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        print("[debate] empty verdict -> AI_MODEL_ERROR_VERDICT")
        return AI_MODEL_ERROR_VERDICT
    return cleaned


def _default_pre_mortem() -> dict[str, list[str]]:
    return {
        "failure_modes": [
            "Core assumptions in the verdict prove wrong under real-world constraints.",
            "Available evidence was too thin or contradictory to support the conclusion.",
            "A overlooked second-order effect invalidates the recommended path.",
        ],
        "critical_unknowns": [
            "Whether the strongest claims hold up against primary sources.",
            "How much key facts in the live web bundle may be outdated or biased.",
            "What decisive data would flip the verdict if it surfaced.",
        ],
    }


def _default_execution_roadmap(query: str = "") -> dict[str, str]:
    trimmed = (query or "").strip()
    if trimmed:
        return {
            "immediate_action": (
                f"Verify the three decisive claims about “{trimmed[:80]}” "
                "using the cited live web sources in this report."
            ),
            "plan_b": (
                "If verification fails, narrow scope (time, geography, or criteria) "
                "and re-run deliberation."
            ),
        }
    return {
        "immediate_action": (
            "Verify the top claims in the verdict against the cited live web sources."
        ),
        "plan_b": "If sources conflict, narrow the question and re-run deliberation.",
    }


def _default_friction_matrix() -> list[dict[str, str]]:
    return [
        {
            "name": "Market Researcher",
            "stance": AgentStance.AGREES.value,
            "argument": "Evidence supports proceeding with disciplined execution.",
        },
        {
            "name": "Skeptical Debater",
            "stance": AgentStance.DISAGREES.value,
            "argument": "Downside risks may outweigh the upside under stress.",
        },
        {
            "name": "CEO Synthesizer",
            "stance": AgentStance.NEUTRAL.value,
            "argument": "Proceed only with explicit validation gates.",
        },
    ]


BOARDROOM_ROLE_TITLES = (
    "Product Analyst",
    "Skeptic",
    "Budget Buyer",
    "Market Analyst",
    "Domain Expert",
    "Risk Officer",
    "Growth Lead",
    "CEO Synthesizer",
)


def _classify_evidence_cluster(url: str, title: str, source: str) -> str:
    blob = f"{url} {title} {source}".lower()
    if "reddit.com" in blob or "redd.it" in blob:
        return "reddit"
    if "youtube.com" in blob or "youtu.be" in blob:
        return "youtube"
    if (
        ".gov" in blob
        or "docs." in blob
        or "documentation" in blob
        or "github.com" in blob
        or "sec.gov" in blob
        or "/doc/" in blob
    ):
        return "official"
    return "news"


def _citation_from_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "title": str(row.get("title") or row.get("url") or "Untitled")[:300],
        "url": str(row.get("url") or "")[:2000],
        "source": str(row.get("source") or "Web")[:120],
        "snippet": str(row.get("snippet") or "")[:1800],
    }


def build_evidence_vault_clusters(
    evidence_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    clusters: dict[str, list[dict[str, str]]] = {
        "reddit": [],
        "youtube": [],
        "official": [],
        "news": [],
    }
    for row in evidence_rows:
        url = str(row.get("url") or "").strip()
        if not url.lower().startswith("http"):
            continue
        key = _classify_evidence_cluster(
            url,
            str(row.get("title") or ""),
            str(row.get("source") or ""),
        )
        clusters[key].append(_citation_from_row(row))
    return clusters


def build_evidence_vault(
    evidence_rows: list[dict[str, str]],
    *,
    confidence: int,
    friction_matrix: list[dict[str, str]],
    ceo_vault: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clusters = build_evidence_vault_clusters(evidence_rows)
    cited = sum(len(items) for items in clusters.values())
    disagree_count = sum(
        1 for row in friction_matrix if str(row.get("stance") or "").upper() == "DISAGREES"
    )

    stats_raw = (
        ceo_vault.get("stats") if isinstance(ceo_vault, dict) else None
    ) or {}

    def _stat(key_snake: str, key_camel: str, default: int) -> int:
        raw = stats_raw.get(key_snake) if isinstance(stats_raw, dict) else None
        if raw is None and isinstance(stats_raw, dict):
            raw = stats_raw.get(key_camel)
        if isinstance(raw, (int, float)):
            return max(0, int(raw))
        return default

    total_sources = _stat("total_sources", "totalSources", max(cited * 4, cited, 24))
    high_signal = _stat("high_signal", "highSignal", max(cited, 1))
    contradictory = _stat(
        "contradictory",
        "contradictory",
        max(disagree_count, 1 if disagree_count else 0),
    )
    dominant_default = 1 if confidence >= 65 else 0
    dominant_consensus = _stat(
        "dominant_consensus",
        "dominantConsensus",
        dominant_default,
    )
    if dominant_consensus not in (0, 1):
        dominant_consensus = min(1, dominant_consensus)

    ceo_clusters = (
        ceo_vault.get("clusters") if isinstance(ceo_vault, dict) else None
    )
    if isinstance(ceo_clusters, dict):
        for key in ("reddit", "youtube", "official", "news"):
            raw_list = ceo_clusters.get(key)
            if isinstance(raw_list, list) and raw_list and not clusters[key]:
                for item in raw_list:
                    if isinstance(item, dict) and item.get("url"):
                        clusters[key].append(_citation_from_row(item))

    return {
        "stats": {
            "totalSources": total_sources,
            "highSignal": high_signal,
            "contradictory": contradictory,
            "dominantConsensus": dominant_consensus,
        },
        "clusters": clusters,
    }


def _derive_recommendation(verdict: str, confidence: int) -> str:
    blob = verdict.lower()
    if any(token in blob for token in ("pivot", "reposition", "change course")):
        return "PIVOT"
    if any(
        token in blob
        for token in (
            "wait",
            "hold",
            "pause",
            "caution",
            "do not launch",
            "don't launch",
            "no-go",
        )
    ):
        return "WAIT"
    if confidence >= 55:
        return "BUY"
    if confidence >= 35:
        return "WAIT"
    return "PIVOT"


def _derive_fit(confidence: int) -> str:
    if confidence >= 75:
        return "Excellent"
    if confidence >= 50:
        return "Good"
    return "Weak"


def build_executive_summary(
    *,
    verdict: str,
    confidence: int,
    tldr: list[str],
    ceo_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    if isinstance(ceo_summary, dict):
        rec = str(ceo_summary.get("recommendation") or "").strip().upper()
        fit = str(ceo_summary.get("fitForYou") or ceo_summary.get("fit_for_you") or "").strip()
        reason = str(
            ceo_summary.get("oneLineReason") or ceo_summary.get("one_line_reason") or "",
        ).strip()
        if rec in ("BUY", "WAIT", "PIVOT") and fit in ("Excellent", "Good", "Weak") and reason:
            return {
                "recommendation": rec,
                "fitForYou": fit,
                "oneLineReason": reason[:500],
            }

    parts = [s.replace(".", "") for s in tldr[:3] if s.strip()]
    reason = (
        f"Because {', '.join(parts)}."
        if parts
        else (verdict.split(".")[0].strip() + ".") if verdict else "Because the swarm could not finalize a reason chain."
    )
    return {
        "recommendation": _derive_recommendation(verdict, confidence),
        "fitForYou": _derive_fit(confidence),
        "oneLineReason": reason[:500],
    }


def _pick_friction_argument(
    friction_matrix: list[dict[str, str]],
    stance: str,
) -> str:
    matches = [
        str(row.get("argument") or "").strip()
        for row in friction_matrix
        if str(row.get("stance") or "").upper() == stance and str(row.get("argument") or "").strip()
    ]
    if not matches:
        return ""
    return max(matches, key=len)


def build_boardroom_summary(
    *,
    verdict: str,
    tldr: list[str],
    friction_matrix: list[dict[str, str]],
    execution_roadmap: dict[str, str],
    ceo_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    if isinstance(ceo_summary, dict):
        fields = (
            "bullCase",
            "bearCase",
            "shoalRecommendation",
            "mainOpportunity",
            "mainRisk",
            "hiddenTradeoff",
            "bestAlternative",
            "explanation",
        )
        snake = {
            "bullCase": "bull_case",
            "bearCase": "bear_case",
            "shoalRecommendation": "shoal_recommendation",
            "mainOpportunity": "main_opportunity",
            "mainRisk": "main_risk",
            "hiddenTradeoff": "hidden_tradeoff",
            "bestAlternative": "best_alternative",
            "explanation": "explanation",
        }
        out: dict[str, str] = {}
        for camel, snake_key in snake.items():
            val = str(ceo_summary.get(camel) or ceo_summary.get(snake_key) or "").strip()
            if val:
                out[camel] = val[:1200]
        if len(out) >= 6:
            if "explanation" not in out:
                out["explanation"] = " ".join(tldr[:2])[:600] if tldr else verdict[:600]
            return out

    bull = _pick_friction_argument(friction_matrix, "AGREES") or (tldr[0] if tldr else "")
    bear = _pick_friction_argument(friction_matrix, "DISAGREES") or (
        tldr[1] if len(tldr) > 1 else ""
    )
    neutral = _pick_friction_argument(friction_matrix, "NEUTRAL") or verdict[:400]
    explanation = " ".join(tldr[:2]) if len(tldr) >= 2 else verdict[:500]
    return {
        "bullCase": bull or "Upside case supported by favorable signals in live research.",
        "bearCase": bear or "Downside case if key assumptions in the research fail.",
        "shoalRecommendation": neutral or verdict[:500],
        "mainOpportunity": bull[:200] or tldr[0][:200] if tldr else "Credible upside if assumptions hold.",
        "mainRisk": bear[:200] or "Material downside if timing or demand slips.",
        "hiddenTradeoff": (
            tldr[1][:200] if len(tldr) > 1 else "Speed of action versus certainty on unknowns."
        ),
        "bestAlternative": execution_roadmap.get("plan_b", "")[:500]
        or "Narrow scope and re-run deliberation with tighter constraints.",
        "explanation": explanation[:1200],
    }


def _resolve_boardroom_role(name: str, index: int) -> str:
    lower = name.lower()
    if "skeptic" in lower or "debater" in lower:
        return "Skeptic"
    if "ceo" in lower or "synth" in lower:
        return "CEO Synthesizer"
    if "budget" in lower or "finance" in lower:
        return "Budget Buyer"
    if "product" in lower:
        return "Product Analyst"
    if "market" in lower:
        return "Market Analyst"
    return BOARDROOM_ROLE_TITLES[index % len(BOARDROOM_ROLE_TITLES)]


def build_debate_room(
    workers: list[Any],
    friction_matrix: list[dict[str, str]],
    *,
    ceo_room: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    if isinstance(ceo_room, list) and len(ceo_room) >= 1:
        cleaned: list[dict[str, str]] = []
        for index, item in enumerate(ceo_room):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            conclusion = str(item.get("conclusion") or "").strip()
            disagreement = str(item.get("disagreement") or "").strip()
            mind_changed = str(
                item.get("mindChanged") or item.get("mind_changed") or "",
            ).strip()
            if role and conclusion and disagreement and mind_changed:
                cleaned.append(
                    {
                        "role": role[:120],
                        "conclusion": conclusion[:800],
                        "disagreement": disagreement[:500],
                        "mindChanged": mind_changed[:500],
                    },
                )
        if cleaned:
            return cleaned

    room: list[dict[str, str]] = []
    friction_by_name = {
        str(row.get("name") or ""): row for row in friction_matrix if row.get("name")
    }
    for index, worker in enumerate(workers):
        name = str(getattr(worker, "name", "") or "").strip()
        argument = str(getattr(worker, "argument", "") or "").strip()
        stance = str(getattr(worker, "stance_label", "") or "NEUTRAL").upper()
        row = friction_by_name.get(name, {})
        opponent = next(
            (
                str(r.get("name") or "another seat")
                for r in friction_matrix
                if str(r.get("name") or "") != name
                and str(r.get("stance") or "").upper() != stance
            ),
            "the room",
        )
        to_label = "YES" if stance == "AGREES" else "NO" if stance == "DISAGREES" else "MAYBE"
        from_label = "MAYBE" if to_label == "YES" else "YES" if to_label == "NO" else "HOLD"
        room.append(
            {
                "role": _resolve_boardroom_role(name, index),
                "conclusion": (str(row.get("argument") or argument) or "No conclusion recorded.")[
                    :800
                ],
                "disagreement": f"With {opponent} on a core assumption in the live research.",
                "mindChanged": f"Moved from {from_label} to {to_label} after cross-checking Tavily sources.",
            },
        )
    return room


def _ensure_boardroom_fields(
    result: DebateResult,
    workers: list[Any] | None = None,
) -> DebateResult:
    verdict = str(result.get("verdict") or "")
    confidence = int(result.get("confidence") or 0)
    tldr = list(result.get("tldr") or [])
    friction = list(result.get("friction_matrix") or [])
    evidence = list(result.get("evidence") or [])
    execution = dict(result.get("execution_roadmap") or _default_execution_roadmap())

    ceo_exec = result.get("executive_summary")
    ceo_board = result.get("boardroom_summary")
    ceo_room = result.get("debate_room")
    ceo_vault = result.get("evidence_vault")

    result["executive_summary"] = build_executive_summary(
        verdict=verdict,
        confidence=confidence,
        tldr=tldr,
        ceo_summary=ceo_exec if isinstance(ceo_exec, dict) else None,
    )
    result["boardroom_summary"] = build_boardroom_summary(
        verdict=verdict,
        tldr=tldr,
        friction_matrix=friction,
        execution_roadmap=execution,
        ceo_summary=ceo_board if isinstance(ceo_board, dict) else None,
    )
    result["debate_room"] = build_debate_room(
        workers or [],
        friction,
        ceo_room=ceo_room if isinstance(ceo_room, list) else None,
    )
    result["evidence_vault"] = build_evidence_vault(
        evidence,
        confidence=confidence,
        friction_matrix=friction,
        ceo_vault=ceo_vault if isinstance(ceo_vault, dict) else None,
    )
    return result


def fallback_debate_result(reason: str | None = None, query: str = "") -> DebateResult:
    if reason:
        print(f"[debate] FALLBACK reason={reason[:400]}")
    base: DebateResult = {
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
        "executive_summary": {},
        "boardroom_summary": {},
        "debate_room": [],
        "evidence_vault": {},
    }
    return _ensure_boardroom_fields(base)


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

    merged: DebateResult = {
        "verdict": verdict,
        "confidence": confidence,
        "agents": agents,
        "tldr": tldr[:5],
        "friction_matrix": friction,
        "pre_mortem": pre_mortem,
        "execution_roadmap": execution,
        "evidence": evidence,
        "executive_summary": {},
        "boardroom_summary": {},
        "debate_room": [],
        "evidence_vault": {},
    }
    return _ensure_boardroom_fields(merged)


def format_evidence_for_prompt(items: list[EvidenceItem]) -> str:
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


def evidence_for_webhook(items: list[EvidenceItem]) -> list[dict[str, str]]:
    """Map Tavily (or live web) hits to webhook evidence — excludes Shoal placeholders."""
    rows: list[dict[str, str]] = []
    for item in items:
        url = (item.get("url") or "").strip()
        if not url.lower().startswith("http"):
            continue
        if "shoal.ai" in url.lower():
            continue
        title = (item.get("title") or "").strip() or url
        source = (item.get("source") or "").strip() or "Web (Tavily)"
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


def friction_matrix_from_workers(
    workers: list[Any],
) -> list[dict[str, str]]:
    """One friction row per spawned worker (source of truth for agent_count)."""
    matrix: list[dict[str, str]] = []
    for worker in workers:
        name = str(getattr(worker, "name", "") or "").strip()
        stance = str(getattr(worker, "stance_label", "") or "NEUTRAL").strip().upper()
        if stance not in ("AGREES", "DISAGREES", "NEUTRAL"):
            stance = "NEUTRAL"
        argument = str(getattr(worker, "argument", "") or "").strip()
        if not name or not argument:
            continue
        matrix.append(
            {
                "name": name,
                "stance": stance,
                "argument": argument[:500],
            },
        )
    return matrix


def agents_from_workers(workers: list[Any]) -> list[DebateAgent]:
    return [
        {
            "name": str(getattr(w, "name", "")).strip(),
            "position": str(getattr(w, "argument", "")).strip()[:500],
        }
        for w in workers
        if str(getattr(w, "name", "")).strip()
    ]


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
        print(f"[debate] Pydantic validation failed: {exc}")
        return None


def _executive_summary_to_dict(summary: ExecutiveSummary) -> dict[str, str]:
    return {
        "recommendation": summary.recommendation,
        "fitForYou": summary.fit_for_you,
        "oneLineReason": summary.one_line_reason,
    }


def _boardroom_summary_to_dict(summary: BoardroomSummary) -> dict[str, str]:
    return {
        "bullCase": summary.bull_case,
        "bearCase": summary.bear_case,
        "shoalRecommendation": summary.shoal_recommendation,
        "mainOpportunity": summary.main_opportunity,
        "mainRisk": summary.main_risk,
        "hiddenTradeoff": summary.hidden_tradeoff,
        "bestAlternative": summary.best_alternative,
        "explanation": summary.explanation,
    }


def _debate_room_to_dict(room: list[DebateRoomAgent]) -> list[dict[str, str]]:
    return [
        {
            "role": agent.role,
            "conclusion": agent.conclusion,
            "disagreement": agent.disagreement,
            "mindChanged": agent.mind_changed,
        }
        for agent in room
    ]


def _evidence_vault_to_dict(vault: EvidenceVault) -> dict[str, Any]:
    return {
        "stats": {
            "totalSources": vault.stats.total_sources,
            "highSignal": vault.stats.high_signal,
            "contradictory": vault.stats.contradictory,
            "dominantConsensus": vault.stats.dominant_consensus,
        },
        "clusters": {
            "reddit": [
                {
                    "title": c.title,
                    "url": c.url,
                    "source": c.source,
                    "snippet": c.snippet,
                }
                for c in vault.clusters.reddit
            ],
            "youtube": [
                {
                    "title": c.title,
                    "url": c.url,
                    "source": c.source,
                    "snippet": c.snippet,
                }
                for c in vault.clusters.youtube
            ],
            "official": [
                {
                    "title": c.title,
                    "url": c.url,
                    "source": c.source,
                    "snippet": c.snippet,
                }
                for c in vault.clusters.official
            ],
            "news": [
                {
                    "title": c.title,
                    "url": c.url,
                    "source": c.source,
                    "snippet": c.snippet,
                }
                for c in vault.clusters.news
            ],
        },
    }


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
        "evidence": [],
        "executive_summary": _executive_summary_to_dict(payload.executive_summary),
        "boardroom_summary": _boardroom_summary_to_dict(payload.boardroom_summary),
        "debate_room": _debate_room_to_dict(payload.debate_room),
        "evidence_vault": _evidence_vault_to_dict(payload.evidence_vault),
    }


def parse_ceo_json(
    synthesis: str,
    *,
    worker_digest: str,
    query: str,
) -> DebateResult:
    """Parse CEO Turn-2 JSON into a DebateResult."""
    final_text = synthesis or ""
    parsed = _extract_json_object(final_text)

    if parsed:
        completion = _coerce_completion_payload(parsed)
        if completion:
            return finalize_debate_result(_payload_to_result(completion))

        partial = _build_result_from_partial(parsed, worker_digest, query)
        if partial:
            return partial

    if not final_text.strip():
        return fallback_debate_result("Empty CEO synthesis", query)

    return fallback_debate_result("Could not parse CEO JSON", query)


def _build_result_from_partial(
    parsed: dict[str, Any],
    worker_digest: str,
    query: str,
) -> DebateResult | None:
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
            name = str(item.get("name") or f"Agent {index + 1}").strip()
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
    execution_roadmap = _default_execution_roadmap(query)
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
            name = str(item.get("name") or f"Agent {index + 1}").strip()
            position = str(item.get("position") or item.get("stance") or "").strip()
            agents.append(
                {
                    "name": name or f"Agent {index + 1}",
                    "position": position or "No position recorded.",
                },
            )

    if len(tldr) < 3:
        tldr = [
            verdict[:200] if verdict else "Verdict synthesized from swarm debate.",
            "Worker panel surfaced material disagreement on key claims.",
            "Validate decisive claims against the live web sources cited.",
        ]

    if not friction_matrix:
        friction_matrix = _default_friction_matrix()

    if not agents and worker_digest:
        agents = [{"name": "CEO Synthesizer", "position": verdict[:500]}]

    exec_raw = parsed.get("executive_summary") or parsed.get("executiveSummary")
    board_raw = parsed.get("boardroom_summary") or parsed.get("boardroomSummary")
    room_raw = parsed.get("debate_room") or parsed.get("debateRoom")
    vault_raw = parsed.get("evidence_vault") or parsed.get("evidenceVault")

    return finalize_debate_result(
        {
            "verdict": verdict,
            "confidence": confidence,
            "agents": agents,
            "tldr": tldr[:5],
            "friction_matrix": friction_matrix,
            "pre_mortem": pre_mortem,
            "execution_roadmap": execution_roadmap,
            "evidence": [],
            "executive_summary": exec_raw if isinstance(exec_raw, dict) else {},
            "boardroom_summary": board_raw if isinstance(board_raw, dict) else {},
            "debate_room": room_raw if isinstance(room_raw, list) else [],
            "evidence_vault": vault_raw if isinstance(vault_raw, dict) else {},
        },
    )
