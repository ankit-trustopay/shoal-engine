"""
Minimal 3-agent CrewAI debate via OpenRouter (ChatOpenAI).

Returns a strict JSON object:
  {"verdict": str, "confidence": number, "agents": [{"name", "position"}]}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, TypedDict

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LITE_MODEL = "meta-llama/llama-3-8b-instruct"
PLUS_MODEL = "openai/gpt-4o-mini"

AGENT_NAMES = ("Market Researcher", "Skeptical Debater", "CEO Synthesizer")

FALLBACK_VERDICT = (
    "The deliberation engine could not produce a verdict. "
    "Verify OPENROUTER_API_KEY and model availability, then retry."
)

DEBATE_JSON_SPEC = """
Return ONLY valid JSON (no markdown fences, no commentary) with this exact shape:
{
  "verdict": "<clear executive verdict in 2-4 sentences>",
  "confidence": <integer 0-100>,
  "agents": [
    {"name": "Market Researcher", "position": "<this agent's stance in one sentence>"},
    {"name": "Skeptical Debater", "position": "<this agent's stance in one sentence>"},
    {"name": "CEO Synthesizer", "position": "<this agent's synthesis in one sentence>"}
  ]
}
"""


class DebateAgent(TypedDict):
    name: str
    position: str


class DebateResult(TypedDict):
    verdict: str
    confidence: int
    agents: list[DebateAgent]


def fallback_debate_result(reason: str | None = None) -> DebateResult:
    detail = (reason or "Engine error").strip()[:240]
    return {
        "verdict": f"{FALLBACK_VERDICT} ({detail})",
        "confidence": 0,
        "agents": [
            {
                "name": name,
                "position": "Deliberation did not complete for this agent.",
            }
            for name in AGENT_NAMES
        ],
    }


def _resolve_model(model_mix: float) -> str:
    return PLUS_MODEL if model_mix > 0 else LITE_MODEL


def _build_openrouter_llm(model_mix: float) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    return ChatOpenAI(
        model=_resolve_model(model_mix),
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=0.35,
        max_tokens=2048,
    )


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


def _normalize_agents(raw: Any) -> list[DebateAgent]:
    if not isinstance(raw, list):
        return [
            {"name": name, "position": "No position recorded."} for name in AGENT_NAMES
        ]

    agents: list[DebateAgent] = []
    for index, item in enumerate(raw):
        default_name = AGENT_NAMES[index] if index < len(AGENT_NAMES) else f"Agent {index + 1}"
        if not isinstance(item, dict):
            agents.append({"name": default_name, "position": "No position recorded."})
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


def _normalize_result(payload: dict[str, Any]) -> DebateResult:
    verdict = str(payload.get("verdict") or "").strip()
    if not verdict:
        verdict = FALLBACK_VERDICT

    confidence_raw = payload.get("confidence")
    if isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool):
        confidence = int(max(0, min(100, round(float(confidence_raw)))))
    else:
        confidence = 0

    return {
        "verdict": verdict,
        "confidence": confidence,
        "agents": _normalize_agents(payload.get("agents")),
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
        llm = _build_openrouter_llm(model_mix)
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
        goal="Produce the final JSON verdict package for Shoal AI.",
        backstory="You weigh evidence vs skepticism and output strict JSON only.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Query:\n{trimmed}\n\n"
            "List 5-8 bullet facts, 3 assumptions, and 2-3 constraints."
        ),
        expected_output="Structured research bullets.",
        agent=researcher,
    )

    debate_task = Task(
        description=(
            f"Query:\n{trimmed}\n\n"
            "Using the research, challenge at least 3 claims and list key risks."
        ),
        expected_output="Critical debate notes.",
        agent=skeptic,
        context=[research_task],
    )

    verdict_task = Task(
        description=(
            f"Query:\n{trimmed}\n\n"
            "Read the researcher and skeptic outputs. "
            f"Then output the final package.\n{DEBATE_JSON_SPEC}"
        ),
        expected_output="Single JSON object matching the specification.",
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

    raw_output = str(getattr(crew_result, "raw", crew_result) or "").strip()
    task_outputs = getattr(crew_result, "tasks_output", None)
    if task_outputs:
        last = list(task_outputs)[-1]
        raw_output = str(getattr(last, "raw", last) or raw_output).strip()

    parsed = _extract_json_object(raw_output)
    if not parsed:
        logger.error("Debate crew returned non-JSON output: %s", raw_output[:500])
        return fallback_debate_result("Invalid JSON from crew")

    return _normalize_result(parsed)
