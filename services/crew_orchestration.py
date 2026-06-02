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

LITE_WORKER_MODEL = "deepseek-v3"
PLUS_WORKER_MODEL = "claude-3.5-sonnet"
CEO_FIXED_MODEL = "gpt-4o"


def _resolve_worker_model(model_tier: str | None) -> str:
    tier = (model_tier or "lite").strip().lower()
    return PLUS_WORKER_MODEL if tier == "plus" else LITE_WORKER_MODEL


def _resolve_ceo_model(requested: str | None) -> str:
    """
    Keep the CEO on a frontier model regardless of worker tier.
    If a caller explicitly requests a frontier model, honor it; otherwise default to gpt-4o.
    """
    if not requested:
        return CEO_FIXED_MODEL
    normalized = requested.strip().lower()
    if "gpt-4o" in normalized or "claude-3.5" in normalized:
        return requested
    return CEO_FIXED_MODEL


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
    worker_model: str | None,
) -> list[Agent]:
    tavily_tool = build_tavily_search_tool()
    worker_llm = build_crew_llm(worker_model, temperature=0.45, max_tokens=2048)
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
    *,
    target_audience: str | None,
    price_point: str | None,
    marketing_budget: str | None,
) -> list[Task]:
    tasks: list[Task] = []
    prior_context: list[Task] = []

    context_clause_parts: list[str] = []
    if target_audience and target_audience.strip():
        context_clause_parts.append(f"target audience is {target_audience.strip()}")
    if price_point and price_point.strip():
        context_clause_parts.append(f"price point is {price_point.strip()}")
    if marketing_budget and marketing_budget.strip():
        context_clause_parts.append(f"marketing budget is {marketing_budget.strip()}")
    context_clause = ""
    if context_clause_parts:
        context_clause = (
            "Debate this idea keeping in mind: "
            + "; ".join(context_clause_parts)
            + "."
        )

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
            f"{context_clause}\n\n" if context_clause else f"USER PREMISE:\n{premise}\n\n"
        )
        description += (
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
    *,
    target_audience: str | None,
    price_point: str | None,
    marketing_budget: str | None,
) -> Task:
    roles = ", ".join(p["role"] for p in personas)
    modifiers: list[str] = []
    if target_audience and target_audience.strip():
        modifiers.append(f"Target audience: {target_audience.strip()}")
    if price_point and price_point.strip():
        modifiers.append(f"Price point: {price_point.strip()}")
    if marketing_budget and marketing_budget.strip():
        modifiers.append(f"Marketing budget: {marketing_budget.strip()}")
    modifier_block = ("\n".join(modifiers) + "\n\n") if modifiers else ""
    return Task(
        description=(
            f"USER PREMISE:\n{premise}\n\n"
            f"{modifier_block}"
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
    *,
    model_tier: str | None = None,
    target_audience: str | None = None,
    price_point: str | None = None,
    marketing_budget: str | None = None,
) -> tuple[ManagerSynthesis, list, list[dict[str, str]]]:
    """
    Execute CrewAI hierarchical process and map outputs to Shoal contracts.
    """
    trimmed = premise.strip()
    worker_model = _resolve_worker_model(model_tier)
    ceo_model = _resolve_ceo_model(model)

    agents = _build_worker_agents(personas, trimmed, worker_model)
    debate_tasks = _build_debate_tasks(
        personas,
        agents,
        trimmed,
        web_data,
        target_audience=target_audience,
        price_point=price_point,
        marketing_budget=marketing_budget,
    )
    synthesis_task = _build_synthesis_task(
        trimmed,
        personas,
        debate_tasks,
        web_data,
        target_audience=target_audience,
        price_point=price_point,
        marketing_budget=marketing_budget,
    )
    all_tasks = [*debate_tasks, synthesis_task]

    manager_llm = build_crew_llm(ceo_model, temperature=0.2, max_tokens=6000)

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


def orchestrate_debate(
    query: str,
    *,
    model_tier: str | None = None,
) -> str:
    """
    Minimal, reliable end-to-end CrewAI debate for launch-day stability.

    Uses a small fixed crew (Researcher, Debater, Manager) routed through OpenRouter
    and returns a final string verdict.
    """
    trimmed = (query or "").strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail="Missing query")

    worker_model = _resolve_worker_model(model_tier)
    llm = build_crew_llm(worker_model, temperature=0.25, max_tokens=1400)
    tavily_tool = build_tavily_search_tool()
    tools = [tavily_tool] if tavily_tool is not None else []

    researcher = Agent(
        role="Researcher",
        goal="Gather quick factual context and constraints for the query.",
        backstory="You are a pragmatic researcher. If tools are available, use them sparingly.",
        tools=tools,
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    debater = Agent(
        role="Debater",
        goal="Formulate the strongest arguments for and against, and stress-test assumptions.",
        backstory="You are an adversarial debater who surfaces risks, edge-cases, and counters.",
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    manager = Agent(
        role="Manager",
        goal="Synthesize into a clear verdict with actionable next steps.",
        backstory="You produce concise, decisive outputs with concrete next actions.",
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    research_task = Task(
        description=(
            "You are the Researcher.\n\n"
            f"Query:\n{trimmed}\n\n"
            "Return a short bullet list of key facts/assumptions and 3-5 constraints.\n"
            "If you used any sources, include the domain name in parentheses."
        ),
        expected_output="Bullet list of facts/assumptions and constraints.",
        agent=researcher,
    )

    debate_task = Task(
        description=(
            "You are the Debater.\n\n"
            f"Query:\n{trimmed}\n\n"
            "Using the research context, write:\n"
            "- Strongest case FOR (3-6 bullets)\n"
            "- Strongest case AGAINST (3-6 bullets)\n"
            "- Biggest unknowns (3 bullets)\n"
        ),
        expected_output="Pros/cons bullets plus unknowns.",
        agent=debater,
        context=[research_task],
    )

    verdict_task = Task(
        description=(
            "You are the Manager.\n\n"
            f"Query:\n{trimmed}\n\n"
            "Based on the Researcher + Debater outputs, return a FINAL VERDICT as plain text.\n"
            "Format:\n"
            "Verdict: <one sentence>\n"
            "Rationale: <2-4 bullets>\n"
            "Next steps: <3 numbered steps with specific metrics or deadlines>\n"
        ),
        expected_output="Plain text verdict with rationale and next steps.",
        agent=manager,
        context=[research_task, debate_task],
    )

    crew = Crew(
        agents=[researcher, debater, manager],
        tasks=[research_task, debate_task, verdict_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        crew_result = crew.kickoff()
    except Exception as exc:
        logger.exception("Minimal CrewAI orchestrate_debate failed")
        raise HTTPException(
            status_code=502,
            detail=f"CrewAI minimal debate failed: {exc}",
        ) from exc

    verdict = str(getattr(crew_result, "raw", crew_result) or "").strip()
    if not verdict:
        raise HTTPException(status_code=502, detail="CrewAI returned empty verdict")
    return verdict
