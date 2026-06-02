"""
Production debate pipeline — raw LangChain + OpenRouter (no CrewAI LLM layer).

Three sequential LLM calls: research → skeptic → synthesis.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from services.debate_constants import AI_MODEL_ERROR_VERDICT
from services.openrouter_llm import get_llm, invoke_llm, log_langchain_error

logger = logging.getLogger(__name__)

AGENT_NAMES = ("Market Researcher", "Skeptical Debater", "CEO Synthesizer")

VERDICT_TASK_SPEC = """
Write PLAIN TEXT only (no markdown fences):

VERDICT:
<2-4 sentence executive verdict>

CONFIDENCE:
<integer 0-100>

AGENT POSITIONS:
Market Researcher: <one sentence>
Skeptical Debater: <one sentence>
CEO Synthesizer: <one sentence>
"""


class DebateAgent(TypedDict):
    name: str
    position: str


class DebateResult(TypedDict):
    verdict: str
    confidence: int
    agents: list[DebateAgent]


def ensure_verdict(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        print("[debate_crew] empty verdict -> AI_MODEL_ERROR_VERDICT")
        return AI_MODEL_ERROR_VERDICT
    return cleaned


def fallback_debate_result(reason: str | None = None) -> DebateResult:
    if reason:
        print(f"[debate_crew] FALLBACK reason={reason[:400]}")
    return {
        "verdict": AI_MODEL_ERROR_VERDICT,
        "confidence": 0,
        "agents": [
            {"name": name, "position": "Deliberation did not complete."}
            for name in AGENT_NAMES
        ],
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
    return {"verdict": verdict, "confidence": confidence, "agents": agents}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
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


def _parse_plaintext_verdict(text: str) -> tuple[str, int, list[DebateAgent]]:
    raw = (text or "").strip()
    if not raw:
        return "", 0, []

    verdict = ""
    confidence = 0
    agents: list[DebateAgent] = []

    verdict_match = re.search(
        r"VERDICT:\s*(.+?)(?=\n\s*CONFIDENCE:|\n\s*AGENT POSITIONS:|\Z)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if verdict_match:
        verdict = verdict_match.group(1).strip()

    conf_match = re.search(r"CONFIDENCE:\s*(\d{1,3})", raw, re.IGNORECASE)
    if conf_match:
        confidence = int(max(0, min(100, int(conf_match.group(1)))))

    for name in AGENT_NAMES:
        pattern = rf"{re.escape(name)}:\s*(.+?)(?=\n[A-Za-z]|\Z)"
        pos_match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if pos_match:
            agents.append(
                {
                    "name": name,
                    "position": pos_match.group(1).strip()[:500]
                    or "No position recorded.",
                },
            )

    if not verdict and raw:
        verdict = raw[:2000].strip()

    return verdict, confidence, agents


def _build_result(
    research: str,
    debate: str,
    synthesis: str,
) -> DebateResult:
    task_texts = [research, debate, synthesis]
    final_text = synthesis or debate or research

    parsed = _extract_json_object(final_text)
    verdict = ""
    confidence = 0
    agents_raw: Any = None

    if parsed:
        verdict = str(parsed.get("verdict") or "").strip()
        cr = parsed.get("confidence")
        if isinstance(cr, (int, float)) and not isinstance(cr, bool):
            confidence = int(max(0, min(100, round(float(cr)))))
        agents_raw = parsed.get("agents")

    plain_verdict, plain_conf, plain_agents = _parse_plaintext_verdict(final_text)
    if not verdict:
        verdict = plain_verdict
    if confidence == 0 and plain_conf > 0:
        confidence = plain_conf

    agents: list[DebateAgent] = []
    if isinstance(agents_raw, list) and agents_raw:
        for index, item in enumerate(agents_raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or AGENT_NAMES[index]).strip()
            position = str(
                item.get("position") or item.get("stance") or "",
            ).strip()
            agents.append(
                {
                    "name": name or AGENT_NAMES[index],
                    "position": position or "No position recorded.",
                },
            )
    elif plain_agents:
        agents = plain_agents
    elif len(task_texts) >= 3:
        agents = [
            {"name": AGENT_NAMES[0], "position": task_texts[0][:500]},
            {"name": AGENT_NAMES[1], "position": task_texts[1][:500]},
            {"name": AGENT_NAMES[2], "position": task_texts[2][:500]},
        ]
    else:
        agents = [
            {"name": name, "position": "No position recorded."}
            for name in AGENT_NAMES
        ]

    if not verdict.strip():
        verdict = final_text.strip()

    if not verdict.strip():
        return fallback_debate_result("Empty synthesis")

    if confidence == 0:
        confidence = 50

    return finalize_debate_result(
        {"verdict": verdict, "confidence": confidence, "agents": agents},
    )


def run_debate_crew(query: str, *, model_mix: float = 0) -> DebateResult:
    """
    Run 3-stage debate via direct OpenRouter ChatOpenAI calls.
    Never raises — returns fallback on any failure.
    """
    print(f"[debate_crew] === START model_mix={model_mix} ===")

    trimmed = (query or "").strip()
    if not trimmed:
        return fallback_debate_result("Missing query")

    try:
        llm = get_llm(model_mix)
    except Exception as exc:
        logger.exception("LLM setup failed")
        print(f"[debate_crew] get_llm failed: {exc}")
        return fallback_debate_result(str(exc))

    system_base = (
        "You are part of Shoal AI's institutional debate desk. "
        "Be specific and concise. Never return an empty response."
    )

    try:
        print("[debate_crew] stage 1/3 Market Researcher")
        research = invoke_llm(
            llm,
            system_base,
            f"Query:\n{trimmed}\n\n"
            "As Market Researcher, list 5-8 facts, 3 assumptions, 2-3 risks.",
            stage="research",
        )

        print("[debate_crew] stage 2/3 Skeptical Debater")
        debate = invoke_llm(
            llm,
            system_base,
            f"Query:\n{trimmed}\n\nResearch:\n{research}\n\n"
            "As Skeptical Debater, challenge 3 claims and list key risks.",
            stage="debate",
        )

        print("[debate_crew] stage 3/3 CEO Synthesizer")
        synthesis = invoke_llm(
            llm,
            system_base,
            f"Query:\n{trimmed}\n\nResearch:\n{research}\n\nDebate:\n{debate}\n\n"
            f"As CEO Synthesizer, output:\n{VERDICT_TASK_SPEC}",
            stage="synthesis",
        )

        result = _build_result(research, debate, synthesis)
        print(
            f"[debate_crew] === SUCCESS verdict_len={len(result['verdict'])} "
            f"confidence={result['confidence']} ===",
        )
        return finalize_debate_result(result)

    except Exception as exc:
        log_langchain_error(exc, stage="debate_pipeline")
        logger.exception("Debate pipeline failed")
        print(f"[debate_crew] pipeline exception: {exc}")
        return fallback_debate_result(str(exc))
