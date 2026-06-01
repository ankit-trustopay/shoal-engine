"""CrewAI hierarchical CEO–worker swarm execution."""

from __future__ import annotations

import logging
from typing import Any

from crewai import Agent, Crew, Process, Task
from fastapi import HTTPException

from services.crew_llm import build_crew_llm
from services.crew_tools import build_tavily_search_tool
from services.crew_transcript import build_transcript_from_task_outputs
from services.dynamic_personas import DynamicPersona
from services.llm import ManagerSynthesis, parse_manager_synthesis_payload
from services.scraper import EvidenceItem
from services.swarm_types import DebateTranscriptEntry

logger = logging.getLogger(__name__)

CEO_ROLE = "Institutional Swarm CEO"
CEO_GOAL = (
    "Delegate live web research and adversarial debate to your specialist panel, "
    "then compile an institutional-grade JSON verdict for Shoal AI."
)
CEO_BACKSTORY = (
    "You are the CEO of an institutional deliberation desk. You coordinate specialists, "
    "ensure they cite Tavily research, surface genuine disagreement, and deliver a "
    "production JSON package — never generic advice."
)

SYNTHESIS_JSON_SPEC = """
Return ONLY valid JSON (no markdown fences) with this exact shape:
{
  "consensus": "<final answer obeying premise formatting>",
  "agentSentiments": [
    {"role": "<exact worker role>", "sentiment": "For"|"Against"|"Neutral"}
  ],
  "recommendedActions": [
    {"step": 1, "title": "<imperative with metric or proper noun>", "body": "<$ amounts, %, companies, deadlines>"}
  ],
  "minorityDissent": "<strongest counter-argument or empty string>",
  "evidenceQualityScore": <integer 0-100>
}

Rules:
- Classify each worker sentiment from their debate text, not your own bias.
- recommendedActions: 2-3 hyper-specific steps; BAN "Monitor updates", "Do market research", "Stay informed".
- Each action body MUST include a number, $ figure, %, company name, or dated milestone.
- Do NOT include a confidence field (computed downstream).
"""


def _persona_worker_backstory(persona: DynamicPersona, premise: str) -> str:
    return (
        f"You are {persona['name']}, {persona['role']}.\n"
        f"Demographics: {persona['age']}y · {persona['location']} · {persona['income']} · "
        f"{persona['culturalBackground']} · {persona['maritalStatus']}.\n"
        f"IQ {persona['iq']} · EQ {persona['eq']} · Risk: {persona['riskTolerance']}.\n"
        f"Biases (lean in): {persona['biases']}\n"
        f"Stance: {persona.get('adversarial_stance', '')}\n"
        f"Mandate: {persona['debate_instruction']}\n"
        f"Backstory: {persona['backstory']}\n"
        f"Premise under debate: {premise[:500]}"
    )


def _build_worker_agents(
    personas: list[DynamicPersona],
    premise: str,
    model: str | None,
) -> list[Agent]:
    tavily_tool = build_tavily_search_tool()
    worker_llm = build_crew_llm(model, temperature=0.45, max_tokens=2048)
    tools = [tavily_tool] if tavily_tool is not None else []

    agents: list[Agent] = []
    for persona in personas:
        agents.append(
            Agent(
                role=str(persona["role"]),
                goal=(
                    f"Research with Tavily and prosecute the {persona['role']} stance on the premise. "
                    "Attack opposing views with metrics and named entities."
                ),
                backstory=_persona_worker_backstory(persona, premise),
                tools=tools,
                llm=worker_llm,
                allow_delegation=False,
                verbose=True,
            ),
        )
    return agents


def _build_debate_tasks(
    personas: list[DynamicPersona],
    agents: list[Agent],
    premise: str,
    web_data: str,
) -> list[Task]:
    tasks: list[Task] = []
    prior_context: list[Task] = []

    for index, (persona, agent) in enumerate(zip(personas, agents)):
        prior_names = [personas[i]["name"] for i in range(index)]
        prior_clause = (
            f"Prior speakers: {', '.join(prior_names)}. "
            "Name at least one and rebut a specific claim."
            if prior_names
            else "You open the live debate."
        )

        description = (
            f"USER PREMISE:\n{premise}\n\n"
            f"PRE-SCRAPED CONTEXT:\n{web_data[:6000]}\n\n"
            f"You are {persona['name']} ({persona['role']}). {prior_clause}\n"
            "1) Use Tavily search for fresh evidence.\n"
            "2) Open with your professional identity (e.g. 'As a risk-averse auditor…').\n"
            "3) Write 3-4 sentences attacking the premise or prior speakers.\n"
            "4) Cite metrics, dates, dollar figures, or company names.\n"
            "Return ONLY JSON:\n"
            '{"debateTurn":"<your dialogue>","sentiment":"For"|"Against"|"Neutral"}'
        )

        task = Task(
            description=description,
            expected_output=(
                'JSON with keys debateTurn (string) and sentiment ("For"|"Against"|"Neutral").'
            ),
            agent=agent,
            context=prior_context.copy() if prior_context else [],
        )
        tasks.append(task)
        prior_context.append(task)

    return tasks


def _build_synthesis_task(
    premise: str,
    personas: list[DynamicPersona],
    debate_tasks: list[Task],
    web_data: str,
) -> Task:
    roles = ", ".join(p["role"] for p in personas)
    return Task(
        description=(
            f"USER PREMISE:\n{premise}\n\n"
            f"WEB CONTEXT:\n{web_data[:6000]}\n\n"
            f"Worker roles: {roles}.\n"
            "Review every delegated debate turn and compile the final institutional package.\n"
            f"{SYNTHESIS_JSON_SPEC}"
        ),
        expected_output="Single valid JSON object matching the specification.",
        context=debate_tasks,
    )


def _parse_crew_result(
    raw_output: str,
    agent_roles: list[str],
) -> ManagerSynthesis:
    parsed = parse_manager_synthesis_payload(raw_output, agent_roles)
    if parsed is not None:
        return parsed

    logger.error("CEO JSON parse failed; returning minimal synthesis")
    return ManagerSynthesis(
        consensus=raw_output[:8000] if raw_output else "Consensus unavailable.",
        agent_sentiments=["Neutral"] * len(agent_roles),
        recommended_actions=[],
        minority_dissent="",
        confidence=0,
        evidence_quality_score=60,
    )


def run_hierarchical_crew(
    premise: str,
    personas: list[DynamicPersona],
    web_data: str,
    evidence: list[EvidenceItem],
    model: str | None,
) -> tuple[ManagerSynthesis, list, list[dict[str, str]]]:
    """
    Execute CrewAI hierarchical process and map outputs to Shoal contracts.
    """
    trimmed = premise.strip()
    agents = _build_worker_agents(personas, trimmed, model)
    debate_tasks = _build_debate_tasks(personas, agents, trimmed, web_data)
    synthesis_task = _build_synthesis_task(trimmed, personas, debate_tasks, web_data)
    all_tasks = [*debate_tasks, synthesis_task]

    manager_llm = build_crew_llm(model, temperature=0.2, max_tokens=6000)

    crew = Crew(
        agents=agents,
        tasks=all_tasks,
        process=Process.hierarchical,
        manager_llm=manager_llm,
        verbose=True,
        memory=False,
    )

    logger.info(
        "Kicking off CrewAI hierarchical crew (%s workers, %s tasks)",
        len(agents),
        len(all_tasks),
    )

    try:
        crew_result = crew.kickoff()
    except Exception as exc:
        logger.exception("CrewAI hierarchical execution failed")
        raise HTTPException(
            status_code=502,
            detail=f"CrewAI orchestration failed: {exc}",
        ) from exc

    raw_ceo_output = str(getattr(crew_result, "raw", crew_result) or "").strip()
    task_outputs: list[Any] = []

    if hasattr(crew_result, "tasks_output") and crew_result.tasks_output:
        task_outputs = list(crew_result.tasks_output)
    elif hasattr(crew, "tasks_output") and crew.tasks_output:
        task_outputs = list(crew.tasks_output)

    debate_transcript = build_transcript_from_task_outputs(
        personas,
        task_outputs,
        synthesis_task_count=1,
    )

    agent_roles = [p["role"] for p in personas]
    manager_synthesis = _parse_crew_result(raw_ceo_output, agent_roles)

    if not debate_transcript:
        debate_transcript = _fallback_transcript_from_synthesis(
            personas,
            manager_synthesis,
            raw_ceo_output,
        )

    messages: list[dict[str, str]] = [
        {"role": entry.role, "text": entry.text} for entry in debate_transcript
    ]
    messages.append({"role": "Manager", "text": manager_synthesis.consensus})

    logger.info(
        "CrewAI complete: transcript=%s sentiments=%s evidence=%s",
        len(debate_transcript),
        manager_synthesis.agent_sentiments,
        len(evidence),
    )

    return manager_synthesis, debate_transcript, messages


def _fallback_transcript_from_synthesis(
    personas: list[DynamicPersona],
    synthesis: ManagerSynthesis,
    raw_ceo: str,
) -> list[DebateTranscriptEntry]:
    """Last-resort transcript when task outputs are not exposed by CrewAI version."""
    from services.swarm_types import DebateTranscriptEntry, format_debate_timestamp

    snippet = (raw_ceo or synthesis.consensus)[:400]
    return [
        DebateTranscriptEntry(
            agentName=str(personas[0].get("name") or "Panel"),
            role=str(personas[0].get("role") or "Analyst"),
            text=f"[Crew execution log unavailable] CEO synthesis excerpt: {snippet}",
            timestamp=format_debate_timestamp(0),
        ),
    ]
