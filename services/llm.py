"""OpenRouter / DeepSeek LLM client and agent completion helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException
from openai import AsyncOpenAI

from services.metrics import AgentSentiment, normalize_sentiment

logger = logging.getLogger(__name__)

MODEL = "deepseek/deepseek-chat"
FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "deepseek/deepseek-chat")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

INSTITUTIONAL_AGENT_RULES = (
    "VOICE & STYLE (mandatory):\n"
    "- Speak as a tier-1 institutional analyst (investment bank, strategy consulting, policy think tank).\n"
    "- Debate raw data, metrics, and causal mechanisms — cite specifics from the web data.\n"
    "- FORBIDDEN: roleplay stage directions or physical actions (e.g. adjusts glasses, leans forward, sighs).\n"
    "- FORBIDDEN: emotional narration, florid metaphor, or first-person character acting.\n"
    "- Write exactly 2 sentences in crisp, evidence-dense institutional prose."
)

MANAGER_SYSTEM = (
    "You are the Institutional Swarm Manager synthesizing five Archetype Leader "
    "deliberations into a production-grade consensus for a large simulated crowd.\n\n"
    "STRICT PREMISE ADHERENCE (mandatory):\n"
    "- Read the user's premise word-by-word. Honor every explicit formatting constraint.\n"
    '- If the premise asks for "Top 10", "top ten", "10 items", or similar, your consensus '
    "MUST enumerate exactly 10 distinct, numbered items — no fewer, no duplicates.\n"
    "- If the premise specifies structure (bullets, ranked list, table columns, word limit), "
    "mirror it exactly in consensus.\n"
    "- Never substitute a generic summary when the premise demands a specific format.\n\n"
    "VOTE CLASSIFICATION (mandatory):\n"
    '- For each of the five agent responses, assign exactly one sentiment: "For", "Against", or "Neutral".\n'
    "- Classify strictly from each agent's stated position relative to the premise — not your synthesis.\n"
    "- Use the exact agent role labels provided in the user message.\n\n"
    "OUTPUT (valid JSON only — no markdown fences, no commentary):\n"
    "{\n"
    '  "consensus": "<final institutional consensus; obey premise formatting constraints>",\n'
    '  "agentSentiments": [\n'
    '    {"role": "<exact agent role>", "sentiment": "For"|"Against"|"Neutral"}\n'
    "  ],\n"
    '  "recommendedActions": [\n'
    '    {"step": 1, "title": "<short imperative>", "body": "<specific actionable detail>"}\n'
    "  ],\n"
    '  "minorityDissent": "<1-2 sentences summarizing the strongest opposing argument; empty string if unanimous>",\n'
    '  "confidence": <integer 75-95 reflecting consensus strength given the sentiment split>\n'
    "}\n\n"
    "recommendedActions: provide 2-3 highly specific steps grounded in consensus and evidence.\n"
    "minorityDissent: distill the strongest counter-argument from Against/Neutral agents."
)


@dataclass
class RecommendedAction:
    step: int
    title: str
    body: str


@dataclass
class ManagerSynthesis:
    consensus: str
    agent_sentiments: list[AgentSentiment] = field(default_factory=list)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    minority_dissent: str = ""
    confidence: int = 75


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not configured")
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured",
        )

    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_recommended_actions(value: Any) -> list[RecommendedAction]:
    if not isinstance(value, list):
        return []

    actions: list[RecommendedAction] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or item.get("description") or "").strip()
        if not title or not body:
            continue
        step_raw = item.get("step")
        step = int(step_raw) if isinstance(step_raw, (int, float)) else index + 1
        actions.append(RecommendedAction(step=max(1, step), title=title, body=body))

    return actions[:3]


def _match_role_sentiment(
    role: str,
    raw_sentiments: list[dict[str, Any]],
) -> AgentSentiment | None:
    role_lower = role.lower()
    for item in raw_sentiments:
        item_role = str(item.get("role") or "").strip()
        sentiment = normalize_sentiment(str(item.get("sentiment") or ""))
        if not item_role or sentiment is None:
            continue
        item_lower = item_role.lower()
        if item_lower == role_lower or item_lower in role_lower or role_lower in item_lower:
            return sentiment
    return None


def resolve_panel_sentiments(
    agent_roles: list[str],
    raw_sentiments: list[dict[str, Any]],
) -> list[AgentSentiment]:
    """Align manager-assigned sentiments to the five agent roles in deliberation order."""
    resolved: list[AgentSentiment] = []

    for index, role in enumerate(agent_roles):
        matched = _match_role_sentiment(role, raw_sentiments)
        if matched is None and index < len(raw_sentiments):
            matched = normalize_sentiment(
                str(raw_sentiments[index].get("sentiment") or ""),
            )
        resolved.append(matched or "Neutral")

    return resolved


def parse_manager_synthesis_payload(
    raw_text: str,
    agent_roles: list[str],
) -> ManagerSynthesis | None:
    cleaned = _strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None

    consensus = str(payload.get("consensus") or "").strip()
    if not consensus:
        return None

    raw_sentiments = payload.get("agentSentiments") or payload.get("agent_sentiments") or []
    if not isinstance(raw_sentiments, list):
        raw_sentiments = []

    sentiment_dicts = [item for item in raw_sentiments if isinstance(item, dict)]
    agent_sentiments = resolve_panel_sentiments(agent_roles, sentiment_dicts)

    recommended_actions = _parse_recommended_actions(
        payload.get("recommendedActions") or payload.get("recommended_actions"),
    )

    minority_dissent = str(
        payload.get("minorityDissent") or payload.get("minority_dissent") or "",
    ).strip()

    confidence_raw = payload.get("confidence")
    confidence = (
        int(confidence_raw)
        if isinstance(confidence_raw, (int, float))
        else 75
    )

    return ManagerSynthesis(
        consensus=consensus,
        agent_sentiments=agent_sentiments,
        recommended_actions=recommended_actions,
        minority_dissent=minority_dissent,
        confidence=max(75, min(95, confidence)),
    )


async def get_agent_response(
    client: AsyncOpenAI,
    role_name: str,
    persona_instruction: str,
    user_message: str,
    web_data: str,
) -> dict[str, str]:
    system_prompt = (
        f"You are a {role_name}. {persona_instruction}\n\n"
        f"{INSTITUTIONAL_AGENT_RULES}\n\n"
        f"Live web data:\n{web_data}"
    )

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
        )
    except Exception as exc:
        logger.exception("OpenRouter error for role %s", role_name)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate response for {role_name}",
        ) from exc

    response_text = (completion.choices[0].message.content or "").strip()
    logger.info("Agent response received for %s (%s chars)", role_name, len(response_text))

    return {"role": role_name, "text": response_text}


def format_persona_system_prompt(persona: dict) -> str:
    """Build an institutional analyst prompt anchored to a stakeholder archetype."""
    return (
        f"Archetype: {persona['role']} (Key Voice for a slice of the simulated crowd)\n"
        f"Stakeholder profile — Name: {persona['name']} | Age: {persona['age']} | "
        f"Location: {persona['location']} | Income: {persona['income']}\n"
        f"Marital status: {persona['maritalStatus']} | "
        f"Cultural background: {persona['culturalBackground']}\n"
        f"Risk tolerance: {persona['riskTolerance']} | IQ: {persona['iq']} | EQ: {persona['eq']}\n\n"
        f"Analytical lens: {persona['backstory']}\n"
        f"Known biases to stress-test: {persona['biases']}\n"
        f"Argument focus: {persona['debate_instruction']}\n\n"
        f"{INSTITUTIONAL_AGENT_RULES}"
    )


async def get_persona_debate_response(
    client: AsyncOpenAI,
    persona: dict,
    user_message: str,
    web_data: str,
) -> dict[str, str]:
    """Run a debate turn as an institutional analyst representing a stakeholder archetype."""
    role_name = str(persona.get("role") or persona.get("name") or "Agent")
    system_prompt = (
        f"{format_persona_system_prompt(persona)}\n\n"
        f"Live web data:\n{web_data}"
    )

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.45,
        )
    except Exception as exc:
        logger.exception("OpenRouter error for persona %s", role_name)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate response for {role_name}",
        ) from exc

    response_text = (completion.choices[0].message.content or "").strip()
    logger.info(
        "Persona debate response for %s (%s chars)",
        role_name,
        len(response_text),
    )

    return {"role": role_name, "text": response_text}


async def get_manager_synthesis(
    client: AsyncOpenAI,
    premise: str,
    web_data: str,
    combined_perspectives: str,
    agent_roles: list[str],
) -> ManagerSynthesis:
    """Synthesize institutional consensus with strict JSON structure."""
    user_message = (
        f"User premise:\n{premise.strip()}\n\n"
        f"Live web data:\n{web_data}\n\n"
        f"Five agent perspectives (classify each sentiment strictly from these texts):\n"
        f"{combined_perspectives}\n\n"
        "Return JSON only."
    )

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": MANAGER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.25,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.exception("OpenRouter error for Manager synthesis")
        raise HTTPException(
            status_code=502,
            detail="Failed to generate manager synthesis",
        ) from exc

    raw_text = (completion.choices[0].message.content or "").strip()
    parsed = parse_manager_synthesis_payload(raw_text, agent_roles)

    if parsed is not None:
        logger.info(
            "Manager synthesis parsed: sentiments=%s actions=%s",
            parsed.agent_sentiments,
            len(parsed.recommended_actions),
        )
        return parsed

    logger.warning("Manager JSON parse failed; retrying without response_format")
    try:
        retry = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": MANAGER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        retry_text = (retry.choices[0].message.content or "").strip()
        parsed_retry = parse_manager_synthesis_payload(retry_text, agent_roles)
        if parsed_retry is not None:
            return parsed_retry
        raw_text = retry_text
    except Exception:
        logger.exception("Manager synthesis retry failed")

    logger.error("Manager synthesis fallback to plain text (%s chars)", len(raw_text))
    return ManagerSynthesis(
        consensus=raw_text or "Consensus unavailable.",
        agent_sentiments=["Neutral"] * max(1, len(agent_roles)),
        recommended_actions=[],
        minority_dissent="",
        confidence=75,
    )
