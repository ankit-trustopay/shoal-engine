"""
Minimal 3-agent CrewAI debate via OpenRouter (ChatOpenAI).

Returns: {"verdict": str, "confidence": number, "agents": [{"name", "position"}]}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from crewai import Agent, Crew, Process, Task

from services.openrouter_llm import get_llm

logger = logging.getLogger(__name__)

AGENT_NAMES = ("Market Researcher", "Skeptical Debater", "CEO Synthesizer")

FALLBACK_VERDICT = (
    "The deliberation engine could not produce a verdict. "
    "Verify OPENROUTER_API_KEY and model availability, then retry."
)

VERDICT_TASK_SPEC = """
Write your response as PLAIN TEXT only (no markdown code fences).

Structure exactly like this:

VERDICT:
<2-4 sentence executive verdict>

CONFIDENCE:
<integer 0-100>

AGENT POSITIONS:
Market Researcher: <one sentence stance>
Skeptical Debater: <one sentence stance>
CEO Synthesizer: <one sentence synthesis>

Optional: after the above, you may append a JSON block matching:
{"verdict":"...","confidence":0,"agents":[{"name":"...","position":"..."}]}
"""


class DebateAgent(TypedDict):
    name: str
    position: str


class DebateResult(TypedDict):
    verdict: str
    confidence: int
    agents: list[DebateAgent]


def fallback_debate_result(
    reason: str | None = None,
    *,
    raw_dump: str | None = None,
) -> DebateResult:
    detail = (reason or "Engine error").strip()[:200]
    verdict = FALLBACK_VERDICT
    if detail:
        verdict = f"{FALLBACK_VERDICT} ({detail})"
    if raw_dump and raw_dump.strip():
        excerpt = raw_dump.strip()[:1200]
        verdict = f"{verdict}\n\nRaw engine output:\n{excerpt}"

    return {
        "verdict": verdict,
        "confidence": 0,
        "agents": [
            {
                "name": name,
                "position": "Deliberation did not complete for this agent.",
            }
            for name in AGENT_NAMES
        ],
    }


def _task_output_text(task_output: object) -> str:
    if task_output is None:
        return ""
    for attr in ("raw", "output", "result", "content", "description"):
        value = getattr(task_output, attr, None)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    if isinstance(task_output, dict):
        for key in ("raw", "output", "result", "content", "text"):
            value = task_output.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    return str(task_output).strip()


def _gather_crew_texts(crew_result: object) -> list[str]:
    texts: list[str] = []

    task_outputs = getattr(crew_result, "tasks_output", None)
    if task_outputs:
        for item in list(task_outputs):
            text = _task_output_text(item)
            if text:
                texts.append(text)

    raw = str(getattr(crew_result, "raw", "") or "").strip()
    if raw and (not texts or raw != texts[-1]):
        texts.append(raw)

    if not texts:
        texts.append(str(crew_result or "").strip())

    return [t for t in texts if t]


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

    for match in re.finditer(r"\{[\s\S]*?\}", raw):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and (
                "verdict" in parsed or "agents" in parsed
            ):
                return parsed
        except json.JSONDecodeError:
            continue

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

    conf_match = re.search(
        r"CONFIDENCE:\s*(\d{1,3})",
        raw,
        re.IGNORECASE,
    )
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

    if not verdict:
        before_agents = re.split(r"AGENT POSITIONS:", raw, maxsplit=1, flags=re.I)
        candidate = before_agents[0].strip()
        candidate = re.sub(r"^VERDICT:\s*", "", candidate, flags=re.I).strip()
        candidate = re.sub(r"CONFIDENCE:\s*\d+\s*", "", candidate, flags=re.I).strip()
        if candidate and len(candidate) > 20:
            verdict = candidate

    if not verdict and len(raw) > 0:
        verdict = raw[:2000].strip()

    return verdict, confidence, agents


def _agents_from_task_texts(task_texts: list[str]) -> list[DebateAgent]:
    if len(task_texts) >= 3:
        return [
            {
                "name": AGENT_NAMES[0],
                "position": (task_texts[0][:500] or "Research completed."),
            },
            {
                "name": AGENT_NAMES[1],
                "position": (task_texts[1][:500] or "Debate completed."),
            },
            {
                "name": AGENT_NAMES[2],
                "position": (task_texts[2][:500] or "Synthesis completed."),
            },
        ]
    return []


def _normalize_agents(raw: Any, task_texts: list[str]) -> list[DebateAgent]:
    if isinstance(raw, list) and raw:
        agents: list[DebateAgent] = []
        for index, item in enumerate(raw):
            default_name = (
                AGENT_NAMES[index] if index < len(AGENT_NAMES) else f"Agent {index + 1}"
            )
            if not isinstance(item, dict):
                agents.append(
                    {"name": default_name, "position": "No position recorded."},
                )
                continue
            name = str(item.get("name") or default_name).strip() or default_name
            position = str(
                item.get("position") or item.get("stance") or item.get("role") or "",
            ).strip()
            agents.append(
                {
                    "name": name,
                    "position": position or "No position recorded.",
                },
            )
        while len(agents) < len(AGENT_NAMES):
            agents.append(
                {
                    "name": AGENT_NAMES[len(agents)],
                    "position": "No position recorded.",
                },
            )
        return agents[: len(AGENT_NAMES)]

    from_plain = _agents_from_task_texts(task_texts)
    if from_plain:
        return from_plain

    return [
        {"name": name, "position": "No position recorded."} for name in AGENT_NAMES
    ]


def _build_result_from_outputs(
    task_texts: list[str],
    final_text: str,
) -> DebateResult:
    combined = "\n\n".join(task_texts) if task_texts else final_text

    parsed = _extract_json_object(final_text) or _extract_json_object(combined)

    verdict = ""
    confidence = 0
    agents_raw: Any = None

    if parsed:
        verdict = str(parsed.get("verdict") or "").strip()
        confidence_raw = parsed.get("confidence")
        if isinstance(confidence_raw, (int, float)) and not isinstance(
            confidence_raw,
            bool,
        ):
            confidence = int(max(0, min(100, round(float(confidence_raw)))))
        agents_raw = parsed.get("agents")

    plain_verdict, plain_conf, plain_agents = _parse_plaintext_verdict(final_text)
    if not verdict:
        verdict = plain_verdict
    if confidence == 0 and plain_conf > 0:
        confidence = plain_conf

    agents = _normalize_agents(agents_raw, task_texts)
    for index, agent in enumerate(plain_agents):
        if index < len(agents):
            agents[index] = agent

    if not verdict.strip():
        verdict = final_text.strip() or combined.strip()

    if not verdict.strip():
        return fallback_debate_result(
            "Empty synthesis output",
            raw_dump=combined[:2000] if combined else None,
        )

    if confidence == 0:
        confidence = 50

    return {
        "verdict": verdict.strip(),
        "confidence": confidence,
        "agents": agents,
    }


def run_debate_crew(query: str, *, model_mix: float = 0) -> DebateResult:
    """
    Run a sequential 3-agent crew and return normalized debate JSON.
    Never raises — returns fallback JSON on any failure.
    """
    trimmed = (query or "").strip()
    if not trimmed:
        return fallback_debate_result("Missing query")

    try:
        llm = get_llm(model_mix)
        llm.temperature = 0.35
        llm.max_tokens = 2048
    except Exception as exc:
        logger.exception("OpenRouter LLM setup failed")
        return fallback_debate_result(str(exc))

    researcher = Agent(
        role="Market Researcher",
        goal="Gather concrete facts and market context for the query.",
        backstory="You cite specifics: numbers, competitors, and buyer dynamics.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    skeptic = Agent(
        role="Skeptical Debater",
        goal="Stress-test the research and surface counter-arguments.",
        backstory="You challenge assumptions and highlight execution risk.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    synthesizer = Agent(
        role="CEO Synthesizer",
        goal="Deliver a plain-text executive verdict and agent positions.",
        backstory=(
            "You write clear plain text. Never return an empty response. "
            "Always fill VERDICT, CONFIDENCE, and AGENT POSITIONS sections."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Query:\n{trimmed}\n\n"
            "List 5-8 bullet facts, 3 assumptions, and 2-3 constraints."
        ),
        expected_output="Plain-text structured research bullets.",
        agent=researcher,
    )

    debate_task = Task(
        description=(
            f"Query:\n{trimmed}\n\n"
            "Using the research, challenge at least 3 claims and list key risks."
        ),
        expected_output="Plain-text critical debate notes.",
        agent=skeptic,
        context=[research_task],
    )

    verdict_task = Task(
        description=(
            f"Query:\n{trimmed}\n\n"
            "Read the Market Researcher and Skeptical Debater outputs.\n"
            f"{VERDICT_TASK_SPEC}"
        ),
        expected_output=(
            "Plain-text with VERDICT, CONFIDENCE, and AGENT POSITIONS sections "
            "(non-empty strings only)."
        ),
        agent=synthesizer,
        context=[research_task, debate_task],
    )

    crew = Crew(
        agents=[researcher, skeptic, synthesizer],
        tasks=[research_task, debate_task, verdict_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        crew_result = crew.kickoff()
    except Exception as exc:
        logger.exception("CrewAI debate kickoff failed")
        return fallback_debate_result(str(exc))

    task_texts = _gather_crew_texts(crew_result)
    final_text = task_texts[-1] if task_texts else ""

    if not final_text.strip():
        logger.error("Debate crew returned empty final output; texts=%s", task_texts)
        return fallback_debate_result(
            "Empty crew output",
            raw_dump="\n---\n".join(task_texts)[:2000] if task_texts else None,
        )

    try:
        return _build_result_from_outputs(task_texts, final_text)
    except Exception as exc:
        logger.exception("Failed to normalize debate output")
        return fallback_debate_result(str(exc), raw_dump=final_text[:2000])
