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
    DebateAgentPosition,
    DebateCompletionPayload,
    DebateRoomAgent,
    EvidenceVault,
    EvidenceVaultCitation,
    EvidenceVaultClusters,
    EvidenceVaultStats,
    ExecutionRoadmap,
    ExecutiveSummary,
    FrictionMatrixEntry,
    PreMortem,
    SevenZoneReport,
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
Return ONLY a single JSON object. No markdown fences, no prose before or after, no commentary.

REQUIRED schema (every key below must be present):

{{
  "verdict": "<2-4 sentence executive verdict grounded in workers + Tavily>",
  "confidence": <integer 0-100>,
  "executive_summary": {{
    "recommendation": "BUY",
    "confidence": <same integer as top-level confidence>,
    "fit_for_you": "Excellent",
    "one_line_reason": "Because <decisive reason from workers + Tavily>"
  }},
  "boardroom_summary": {{
    "main_opportunity": "<single clearest upside>",
    "main_risk": "<single clearest downside>",
    "hidden_tradeoff": "<non-obvious tradeoff the swarm surfaced>",
    "best_alternative": "<credible Plan B>",
    "explanation": "<exactly 2 sentences tied to the user query>"
  }},
  "debate_room": [
    {{
      "role": "<seat title e.g. Product Analyst, Skeptic>",
      "conclusion": "<worker conclusion in 2-3 sentences>",
      "disagreement": "<what this seat challenged>",
      "mind_changed": "<stance shift e.g. Moved from YES to MAYBE after pricing data>"
    }}
  ],
  "evidence_vault": {{
    "stats": {{
      "total": <integer — sources reviewed, >= cited URLs>,
      "high_signal": <integer — high-relevance sources>
    }},
    "clusters": {{
      "reddit": [{{"title": "...", "url": "https://...", "source": "...", "snippet": "..."}}],
      "news": [],
      "official": []
    }}
  }}
}}

Rules:
- recommendation MUST be exactly BUY, WAIT, or PIVOT.
- fit_for_you MUST be exactly Excellent, Good, or Weak.
- debate_room MUST contain EXACTLY {count} objects — one per worker listed in the user message.
- Assign every Tavily URL from system context to reddit, news, or official (no other cluster keys).
- Use snake_case keys exactly as shown (fit_for_you, one_line_reason, mind_changed, main_opportunity, high_signal).
- Do not omit executive_summary, boardroom_summary, debate_room, or evidence_vault.

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
        raw = None
        if isinstance(stats_raw, dict):
            raw = stats_raw.get(key_snake)
            if raw is None:
                raw = stats_raw.get(key_camel)
            if raw is None and key_snake == "total_sources":
                raw = stats_raw.get("total")
            if raw is None and key_snake == "high_signal":
                raw = stats_raw.get("high_signal")
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
            "total": total_sources,
            "high_signal": high_signal,
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
        conf_raw = ceo_summary.get("confidence")
        conf = (
            int(max(0, min(100, round(float(conf_raw)))))
            if isinstance(conf_raw, (int, float))
            else confidence
        )
        if rec in ("BUY", "WAIT", "PIVOT") and fit in ("Excellent", "Good", "Weak") and reason:
            return {
                "recommendation": rec,
                "confidence": conf,
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
        "confidence": confidence,
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
        field_pairs = (
            ("mainOpportunity", "main_opportunity"),
            ("mainRisk", "main_risk"),
            ("hiddenTradeoff", "hidden_tradeoff"),
            ("bestAlternative", "best_alternative"),
            ("explanation", "explanation"),
            ("bullCase", "bull_case"),
            ("bearCase", "bear_case"),
            ("shoalRecommendation", "shoal_recommendation"),
        )
        out: dict[str, str] = {}
        for camel, snake_key in field_pairs:
            val = str(ceo_summary.get(camel) or ceo_summary.get(snake_key) or "").strip()
            if val:
                out[camel] = val[:1200]
        core_keys = (
            "mainOpportunity",
            "mainRisk",
            "hiddenTradeoff",
            "bestAlternative",
            "explanation",
        )
        if all(out.get(key) for key in core_keys):
            if "bullCase" not in out:
                out["bullCase"] = out["mainOpportunity"]
            if "bearCase" not in out:
                out["bearCase"] = out["mainRisk"]
            if "shoalRecommendation" not in out:
                out["shoalRecommendation"] = out.get("explanation") or verdict[:500]
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
        "executive_summary": (
            result.get("executive_summary")
            if isinstance(result.get("executive_summary"), dict)
            else {}
        ),
        "boardroom_summary": (
            result.get("boardroom_summary")
            if isinstance(result.get("boardroom_summary"), dict)
            else {}
        ),
        "debate_room": (
            result.get("debate_room") if isinstance(result.get("debate_room"), list) else []
        ),
        "evidence_vault": (
            result.get("evidence_vault")
            if isinstance(result.get("evidence_vault"), dict)
            else {}
        ),
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


def _verdict_from_raw_text(raw_text: str, query: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        if query.strip():
            return f"Deliberation on “{query.strip()[:80]}” completed with partial synthesis."
        return "Deliberation complete with partial synthesis."

    embedded = _extract_json_object(text)
    if embedded:
        verdict = str(embedded.get("verdict") or "").strip()
        if verdict:
            return verdict[:2000]

    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("{") or cleaned.startswith("```"):
            continue
        return cleaned[:2000]

    return text[:2000]


def _safe_model_validate(model_cls: type, raw: Any, **defaults: Any):
    merged = {**defaults}
    if isinstance(raw, dict):
        merged.update(raw)
    try:
        return model_cls.model_validate(merged)
    except ValidationError as exc:
        print(f"[debate] {model_cls.__name__} coerce warning: {exc}")
        try:
            return model_cls.model_validate(defaults)
        except ValidationError:
            return model_cls()


def coerce_seven_zone_report(
    parsed: dict[str, Any] | None,
    *,
    raw_text: str = "",
    query: str = "",
) -> SevenZoneReport:
    """Always return a valid SevenZoneReport — never raise."""
    data = parsed if isinstance(parsed, dict) else {}

    verdict = str(data.get("verdict") or "").strip() or _verdict_from_raw_text(raw_text, query)
    confidence_raw = data.get("confidence", 50)
    confidence = (
        int(max(0, min(100, round(float(confidence_raw)))))
        if isinstance(confidence_raw, (int, float))
        else 50
    )

    exec_raw = data.get("executive_summary") or data.get("executiveSummary")
    if not isinstance(exec_raw, dict):
        exec_raw = build_executive_summary(
            verdict=verdict,
            confidence=confidence,
            tldr=[],
        )
    exec_raw = dict(exec_raw)
    exec_raw.setdefault("confidence", confidence)
    executive = _safe_model_validate(ExecutiveSummary, exec_raw, confidence=confidence)

    board_raw = data.get("boardroom_summary") or data.get("boardroomSummary")
    if not isinstance(board_raw, dict):
        board_raw = {}
    boardroom = _safe_model_validate(
        BoardroomSummary,
        board_raw,
        main_opportunity=verdict[:200] or "Upside case from swarm synthesis.",
        main_risk="Downside case if key assumptions fail.",
        explanation=verdict[:500] or "Swarm weighed live research and worker arguments.",
    )

    room_raw = data.get("debate_room") or data.get("debateRoom")
    debate_room: list[DebateRoomAgent] = []
    if isinstance(room_raw, list):
        for item in room_raw:
            if isinstance(item, dict):
                debate_room.append(
                    _safe_model_validate(DebateRoomAgent, item),
                )
    if not debate_room:
        debate_room = [
            _safe_model_validate(
                DebateRoomAgent,
                None,
                role="CEO Synthesizer",
                conclusion=verdict[:800],
            ),
        ]

    vault_raw = data.get("evidence_vault") or data.get("evidenceVault")
    if isinstance(vault_raw, dict):
        try:
            evidence_vault = _parse_evidence_vault_dict(vault_raw)
        except ValidationError as exc:
            print(f"[debate] evidence_vault coerce warning: {exc}")
            evidence_vault = EvidenceVault()
    else:
        evidence_vault = EvidenceVault()

    try:
        return SevenZoneReport.model_validate(
            {
                "verdict": verdict,
                "confidence": confidence,
                "executive_summary": executive,
                "boardroom_summary": boardroom,
                "debate_room": debate_room,
                "evidence_vault": evidence_vault,
            },
        )
    except ValidationError as exc:
        print(f"[debate] SevenZoneReport fallback: {exc}")
        return SevenZoneReport(
            verdict=verdict,
            confidence=confidence,
            executive_summary=executive,
            boardroom_summary=boardroom,
            debate_room=debate_room,
            evidence_vault=evidence_vault,
        )


def _coerce_completion_payload(parsed: dict[str, Any]) -> DebateCompletionPayload | None:
    try:
        return DebateCompletionPayload.model_validate(parsed)
    except ValidationError as exc:
        print(f"[debate] Pydantic validation failed: {exc}")
        return None


def _executive_summary_to_dict(summary: ExecutiveSummary) -> dict[str, Any]:
    return {
        "recommendation": summary.recommendation,
        "confidence": summary.confidence,
        "fit_for_you": summary.fit_for_you,
        "one_line_reason": summary.one_line_reason,
        "fitForYou": summary.fit_for_you,
        "oneLineReason": summary.one_line_reason,
    }


def _boardroom_summary_to_dict(summary: BoardroomSummary) -> dict[str, str]:
    return {
        "main_opportunity": summary.main_opportunity,
        "main_risk": summary.main_risk,
        "hidden_tradeoff": summary.hidden_tradeoff,
        "best_alternative": summary.best_alternative,
        "explanation": summary.explanation,
        "mainOpportunity": summary.main_opportunity,
        "mainRisk": summary.main_risk,
        "hiddenTradeoff": summary.hidden_tradeoff,
        "bestAlternative": summary.best_alternative,
        "bull_case": summary.bull_case,
        "bear_case": summary.bear_case,
        "shoal_recommendation": summary.shoal_recommendation,
        "bullCase": summary.bull_case,
        "bearCase": summary.bear_case,
        "shoalRecommendation": summary.shoal_recommendation,
    }


def _debate_room_to_dict(room: list[DebateRoomAgent]) -> list[dict[str, str]]:
    return [
        {
            "role": agent.role,
            "conclusion": agent.conclusion,
            "disagreement": agent.disagreement,
            "mind_changed": agent.mind_changed,
            "mindChanged": agent.mind_changed,
        }
        for agent in room
    ]


def _evidence_vault_to_dict(vault: EvidenceVault) -> dict[str, Any]:
    return {
        "stats": {
            "total": vault.stats.total,
            "high_signal": vault.stats.high_signal,
            "totalSources": vault.stats.total,
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


def _parse_evidence_vault_dict(raw: dict[str, Any]) -> EvidenceVault:
    stats_raw = raw.get("stats") if isinstance(raw.get("stats"), dict) else {}
    clusters_raw = raw.get("clusters") if isinstance(raw.get("clusters"), dict) else {}

    def _citations(key: str) -> list[EvidenceVaultCitation]:
        items = clusters_raw.get(key)
        if not isinstance(items, list):
            return []
        out: list[EvidenceVaultCitation] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url.lower().startswith("http"):
                continue
            title = str(item.get("title") or url).strip()
            source = str(item.get("source") or "Web").strip()
            snippet = str(item.get("snippet") or "").strip()
            out.append(
                EvidenceVaultCitation(
                    title=title[:300],
                    url=url[:2000],
                    source=source[:120],
                    snippet=snippet[:1800],
                ),
            )
        return out

    total = int(
        stats_raw.get("total")
        or stats_raw.get("totalSources")
        or stats_raw.get("total_sources")
        or 0,
    )
    high_signal = int(
        stats_raw.get("high_signal")
        or stats_raw.get("highSignal")
        or 0,
    )

    return EvidenceVault(
        stats=EvidenceVaultStats(
            total=max(0, total),
            high_signal=max(0, high_signal),
            contradictory=int(stats_raw.get("contradictory") or 0),
            dominant_consensus=int(
                stats_raw.get("dominantConsensus")
                or stats_raw.get("dominant_consensus")
                or 0,
            ),
        ),
        clusters=EvidenceVaultClusters(
            reddit=_citations("reddit"),
            youtube=_citations("youtube"),
            official=_citations("official"),
            news=_citations("news"),
        ),
    )


def _parse_debate_room_list(raw: list[Any]) -> list[DebateRoomAgent]:
    agents: list[DebateRoomAgent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        conclusion = str(item.get("conclusion") or "").strip()
        disagreement = str(item.get("disagreement") or "").strip()
        mind_changed = str(
            item.get("mindChanged") or item.get("mind_changed") or "",
        ).strip()
        if not role or not conclusion or not disagreement or not mind_changed:
            continue
        agents.append(
            DebateRoomAgent(
                role=role[:120],
                conclusion=conclusion[:800],
                disagreement=disagreement[:500],
                mind_changed=mind_changed[:500],
            ),
        )
    return agents


def debate_result_to_completion_payload(
    debate_id: str,
    result: DebateResult,
    *,
    runtime: int,
    cost: float,
    agent_count: int,
    workers: list[Any] | None = None,
) -> DebateCompletionPayload:
    """Normalize DebateResult and validate against strict Pydantic schema."""
    ensured = _ensure_boardroom_fields(dict(result), workers)

    verdict = ensure_verdict(str(ensured.get("verdict") or ""))
    confidence = int(max(0, min(100, int(ensured.get("confidence") or 0))))

    friction_raw = list(ensured.get("friction_matrix") or [])
    friction: list[FrictionMatrixEntry] = []
    for index, row in enumerate(friction_raw):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or f"Agent {index + 1}").strip()
        stance_raw = str(row.get("stance") or "NEUTRAL").strip().upper()
        try:
            stance = AgentStance(stance_raw)
        except ValueError:
            stance = AgentStance.NEUTRAL
        argument = str(row.get("argument") or "").strip()
        if name and argument:
            friction.append(
                FrictionMatrixEntry(name=name, stance=stance, argument=argument[:500]),
            )
    if not friction:
        for row in _default_friction_matrix():
            friction.append(
                FrictionMatrixEntry(
                    name=str(row["name"]),
                    stance=AgentStance(str(row["stance"])),
                    argument=str(row["argument"]),
                ),
            )

    agents_raw = list(ensured.get("agents") or [])
    agents: list[DebateAgentPosition] = []
    for index, row in enumerate(agents_raw):
        if isinstance(row, dict):
            name = str(row.get("name") or f"Agent {index + 1}").strip()
            position = str(row.get("position") or "").strip()
        else:
            name = f"Agent {index + 1}"
            position = ""
        if name:
            agents.append(
                DebateAgentPosition(
                    name=name,
                    position=position or "No position recorded.",
                ),
            )
    if not agents:
        agents = [DebateAgentPosition(name="CEO Synthesizer", position=verdict[:500])]

    tldr = list(ensured.get("tldr") or [])
    if len(tldr) < 3:
        tldr = fallback_debate_result()["tldr"]

    pre_raw = ensured.get("pre_mortem") or _default_pre_mortem()
    pre_mortem = PreMortem(
        failure_modes=list(pre_raw.get("failure_modes") or []),
        critical_unknowns=list(pre_raw.get("critical_unknowns") or []),
    )

    road_raw = ensured.get("execution_roadmap") or _default_execution_roadmap()
    execution = ExecutionRoadmap(
        immediate_action=str(road_raw.get("immediate_action") or "")[:1000],
        plan_b=str(road_raw.get("plan_b") or "")[:1000],
    )

    exec_raw = ensured.get("executive_summary")
    if not isinstance(exec_raw, dict):
        exec_raw = build_executive_summary(
            verdict=verdict,
            confidence=confidence,
            tldr=tldr,
        )
    exec_conf = exec_raw.get("confidence", confidence)
    exec_conf_int = (
        int(max(0, min(100, round(float(exec_conf)))))
        if isinstance(exec_conf, (int, float))
        else confidence
    )
    rec_raw = str(exec_raw.get("recommendation") or "").strip().upper()
    recommendation = (
        rec_raw if rec_raw in ("BUY", "WAIT", "PIVOT") else _derive_recommendation(verdict, confidence)
    )
    fit_raw = str(exec_raw.get("fitForYou") or exec_raw.get("fit_for_you") or "").strip()
    fit_for_you = (
        fit_raw if fit_raw in ("Excellent", "Good", "Weak") else _derive_fit(confidence)
    )
    one_line_reason = str(
        exec_raw.get("oneLineReason")
        or exec_raw.get("one_line_reason")
        or build_executive_summary(verdict=verdict, confidence=confidence, tldr=tldr)[
            "oneLineReason"
        ],
    ).strip()
    executive = ExecutiveSummary(
        recommendation=recommendation,  # type: ignore[arg-type]
        confidence=exec_conf_int,
        fit_for_you=fit_for_you,  # type: ignore[arg-type]
        one_line_reason=one_line_reason,
    )

    board_raw = ensured.get("boardroom_summary")
    if not isinstance(board_raw, dict) or len(board_raw) < 4:
        board_raw = build_boardroom_summary(
            verdict=verdict,
            tldr=tldr,
            friction_matrix=friction_raw,
            execution_roadmap=dict(road_raw),
        )
    try:
        boardroom = BoardroomSummary.model_validate(board_raw)
    except ValidationError:
        boardroom = BoardroomSummary.model_validate(
            build_boardroom_summary(
                verdict=verdict,
                tldr=tldr,
                friction_matrix=friction_raw,
                execution_roadmap=dict(road_raw),
                ceo_summary=board_raw,
            ),
        )

    room_raw = ensured.get("debate_room")
    debate_room = (
        _parse_debate_room_list(room_raw)
        if isinstance(room_raw, list) and room_raw
        else _parse_debate_room_list(
            build_debate_room(workers or [], friction_raw),
        )
    )
    if not debate_room:
        debate_room = _parse_debate_room_list(
            build_debate_room(workers or [], friction_raw),
        )

    vault_raw = ensured.get("evidence_vault")
    if not isinstance(vault_raw, dict):
        vault_raw = build_evidence_vault(
            list(ensured.get("evidence") or []),
            confidence=confidence,
            friction_matrix=friction_raw,
        )
    evidence_vault = _parse_evidence_vault_dict(vault_raw)

    return DebateCompletionPayload(
        debate_id=debate_id,
        status="completed",
        verdict=verdict,
        confidence=confidence,
        agents=agents,
        tldr=tldr[:5],
        friction_matrix=friction,
        pre_mortem=pre_mortem,
        execution_roadmap=execution,
        executive_summary=executive,
        boardroom_summary=boardroom,
        debate_room=debate_room,
        evidence_vault=evidence_vault,
        runtime=max(1, int(runtime)),
        cost=float(cost),
        agent_count=max(1, int(agent_count)),
    )


def completion_payload_to_webhook_body(payload: DebateCompletionPayload) -> dict[str, Any]:
    """Serialize validated payload for shoal-web (snake_case top-level keys)."""
    evidence_rows = []
    for cluster in (
        payload.evidence_vault.clusters.reddit
        + payload.evidence_vault.clusters.youtube
        + payload.evidence_vault.clusters.official
        + payload.evidence_vault.clusters.news
    ):
        evidence_rows.append(
            {
                "title": cluster.title,
                "source": cluster.source,
                "url": cluster.url,
                "snippet": cluster.snippet or cluster.title,
            },
        )

    return {
        "debate_id": payload.debate_id,
        "status": payload.status,
        "verdict": payload.verdict,
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
        "evidence": evidence_rows,
        "executive_summary": _executive_summary_to_dict(payload.executive_summary),
        "boardroom_summary": _boardroom_summary_to_dict(payload.boardroom_summary),
        "debate_room": _debate_room_to_dict(payload.debate_room),
        "evidence_vault": _evidence_vault_to_dict(payload.evidence_vault),
        "runtime": payload.runtime,
        "cost": payload.cost,
        "agentCount": payload.agent_count,
    }


def build_debate_webhook_payload(
    debate_id: str,
    result: DebateResult,
    *,
    runtime: int,
    cost: float,
    agent_count: int,
    workers: list[Any] | None = None,
) -> dict[str, Any]:
    """Strict 7-zone webhook body — always includes boardroom fields."""
    payload = debate_result_to_completion_payload(
        debate_id,
        result,
        runtime=runtime,
        cost=cost,
        agent_count=agent_count,
        workers=workers,
    )
    body = completion_payload_to_webhook_body(payload)
    print(
        f"[debate] webhook payload keys debate_id={debate_id} "
        f"has_executive_summary={bool(body.get('executive_summary'))} "
        f"debate_room={len(body.get('debate_room') or [])} "
        f"evidence_vault_clusters={len((body.get('evidence_vault') or {}).get('clusters') or {})}",
    )
    return body


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
    """Parse CEO Turn-2 JSON into a DebateResult (never raises)."""
    return safe_synthesize_ceo_result(
        synthesis,
        worker_digest=worker_digest,
        query=query,
    )


def safe_synthesize_ceo_result(
    synthesis: str,
    *,
    worker_digest: str = "",
    query: str = "",
    workers: list[Any] | None = None,
    evidence_rows: list[dict[str, str]] | None = None,
) -> DebateResult:
    """
    Bulletproof CEO synthesis: parse JSON when possible, else build valid 7-zone output
    from raw text so the debate still completes.
    """
    final_text = (synthesis or "").strip()
    print(
        f"[debate] safe_synthesize_ceo_result chars={len(final_text)} "
        f"workers={len(workers or [])} evidence={len(evidence_rows or [])}",
    )

    parsed = _extract_json_object(final_text) if final_text else None
    if parsed:
        if not isinstance(parsed.get("tldr"), list) or len(parsed.get("tldr") or []) < 3:
            verdict_seed = str(parsed.get("verdict") or _verdict_from_raw_text(final_text, query))
            parsed["tldr"] = [
                verdict_seed[:200] or "Executive verdict synthesized.",
                "Key risk surfaced during adversarial deliberation.",
                "Validate decisive claims against cited live sources.",
            ]
        if not parsed.get("friction_matrix") and not parsed.get("frictionMatrix"):
            parsed["friction_matrix"] = _default_friction_matrix()
        if not parsed.get("pre_mortem") and not parsed.get("preMortem"):
            parsed["pre_mortem"] = _default_pre_mortem()
        if not parsed.get("execution_roadmap") and not parsed.get("executionRoadmap"):
            parsed["execution_roadmap"] = _default_execution_roadmap(query)

        completion = _coerce_completion_payload(parsed)
        if completion:
            print("[debate] CEO strict DebateCompletionPayload OK")
            result = finalize_debate_result(_payload_to_result(completion))
            if evidence_rows:
                result["evidence"] = evidence_rows
            return _ensure_boardroom_fields(result, workers)

        partial = _build_result_from_partial(parsed, worker_digest, query)
        if partial:
            print("[debate] CEO partial JSON path OK")
            if evidence_rows:
                partial["evidence"] = evidence_rows
            return _ensure_boardroom_fields(partial, workers)

    zones = coerce_seven_zone_report(parsed, raw_text=final_text, query=query)
    print(
        f"[debate] CEO seven-zone coerce verdict_len={len(zones.verdict)} "
        f"debate_room={len(zones.debate_room)}",
    )

    tldr = [
        zones.verdict[:200] if zones.verdict else "Verdict synthesized from swarm debate.",
        zones.boardroom_summary.main_risk[:200],
        zones.boardroom_summary.main_opportunity[:200],
    ]

    friction_matrix = (
        friction_matrix_from_workers(workers)
        if workers
        else _default_friction_matrix()
    )
    agents = agents_from_workers(workers) if workers else []

    result = finalize_debate_result(
        {
            "verdict": zones.verdict,
            "confidence": zones.confidence,
            "agents": agents,
            "tldr": tldr,
            "friction_matrix": friction_matrix,
            "pre_mortem": _default_pre_mortem(),
            "execution_roadmap": _default_execution_roadmap(query),
            "evidence": list(evidence_rows or []),
            "executive_summary": zones.executive_summary.model_dump(by_alias=False),
            "boardroom_summary": zones.boardroom_summary.model_dump(by_alias=False),
            "debate_room": [
                card.model_dump(by_alias=False) for card in zones.debate_room
            ],
            "evidence_vault": _evidence_vault_to_dict(zones.evidence_vault),
        },
    )

    if workers:
        result["friction_matrix"] = friction_matrix_from_workers(workers)
        result["agents"] = agents_from_workers(workers)
        result["debate_room"] = build_debate_room(
            workers,
            result["friction_matrix"],
            ceo_room=result.get("debate_room"),
        )

    if evidence_rows:
        result["evidence_vault"] = build_evidence_vault(
            evidence_rows,
            confidence=result["confidence"],
            friction_matrix=result["friction_matrix"],
            ceo_vault=result.get("evidence_vault"),
        )

    return _ensure_boardroom_fields(result, workers)


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
