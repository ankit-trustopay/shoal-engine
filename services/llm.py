"""OpenRouter LLM client, model routing, and agent completion helpers."""

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
from services.model_router import (
    get_fallback_openrouter_model,
    resolve_openrouter_model,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CONVERSATIONAL_DEBATE_RULES = (
    "CONVERSATIONAL ADVERSARIAL DEBATE (mandatory):\n"
    "- You are in a live, sequential war room. Prior agents have already spoken — read them.\n"
    "- Name at least one prior speaker by first name and attack a specific claim "
    '(e.g. "I disagree with Rahul because…", "Priya\'s point about X ignores…").\n'
    "- Open with your professional identity tied to your background "
    '(e.g. "As a risk-averse auditor…", "As a growth-stage CFO…").\n'
    "- Weaponize the web data — cite metrics, dates, company names, and dollar figures.\n"
    "- Anticipate opposing archetypes listed below and dismantle their logic with evidence.\n"
    "- Do NOT hedge, converge, seek compromise, or validate the other side.\n"
    "- FORBIDDEN: stage directions, emotional narration, or generic filler.\n"
    "- Write exactly 3–4 sentences of confrontational, evidence-dense dialogue."
)

_BANNED_RECOMMENDATION_PHRASES = (
    "monitor updates",
    "monitor the situation",
    "do market research",
    "conduct market research",
    "conduct further research",
    "further research",
    "stay informed",
    "keep an eye on",
    "more research is needed",
    "gather more data",
    "continue to monitor",
    "assess the landscape",
    "evaluate options",
    "consider your options",
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
    evidence_quality_score: int = 75


def build_manager_system(agent_count: int) -> str:
    count = max(1, int(agent_count))
    return (
        "You are the Institutional Swarm Manager synthesizing adversarial agent "
        f"deliberations ({count} Key Voices) into a production-grade consensus.\n\n"
        "STRICT PREMISE ADHERENCE (mandatory):\n"
        "- Read the user's premise word-by-word. Honor every explicit formatting constraint.\n"
        '- If the premise asks for "Top 10", "top ten", "10 items", or similar, your consensus '
        "MUST enumerate exactly 10 distinct, numbered items — no fewer, no duplicates.\n"
        "- If the premise specifies structure (bullets, ranked list, table columns, word limit), "
        "mirror it exactly in consensus.\n"
        "- Never substitute a generic summary when the premise demands a specific format.\n\n"
        "VOTE CLASSIFICATION (mandatory):\n"
        f"- For each of the {count} agent responses, assign exactly one sentiment: "
        '"For", "Against", or "Neutral".\n'
        "- Classify strictly from each agent's stated position relative to the premise — not your synthesis.\n"
        "- Use the exact agent role labels provided in the user message.\n\n"
        "EVIDENCE QUALITY (mandatory):\n"
        "- Score how well agents used specific, verifiable data from the web context "
        "(metrics, dates, named sources, causal mechanisms).\n"
        "- Penalize vague rhetoric, uncited claims, or ignoring available deep research.\n"
        "- evidenceQualityScore (0-100): overall panel evidence quality independent of sentiment.\n"
        "- Do NOT output a confidence score — confidence is computed algorithmically downstream.\n\n"
        "HYPER-SPECIFIC RECOMMENDATIONS (mandatory):\n"
        "- recommendedActions: provide 2-3 steps that are immediately executable.\n"
        "- Each body MUST include at least one of: a dollar amount, a percentage, a headcount, "
        "a named company/competitor, a dated milestone, or a quantified KPI target.\n"
        "- BANNED generic phrases (reject entirely): "
        '"Monitor updates", "Do market research", "Conduct further research", '
        '"Stay informed", "Keep an eye on", "Gather more data", "Evaluate options".\n\n'
        "OUTPUT (valid JSON only — no markdown fences, no commentary):\n"
        "{\n"
        '  "consensus": "<final institutional consensus; obey premise formatting constraints>",\n'
        '  "agentSentiments": [\n'
        '    {"role": "<exact agent role>", "sentiment": "For"|"Against"|"Neutral"}\n'
        "  ],\n"
        '  "recommendedActions": [\n'
        '    {"step": 1, "title": "<short imperative with a number or proper noun>", '
        '"body": "<specific action with metrics, $ figures, company names, or deadlines>"}\n'
        "  ],\n"
        '  "minorityDissent": "<1-2 sentences summarizing the strongest opposing argument; empty string if unanimous>",\n'
        '  "evidenceQualityScore": <integer 0-100>\n'
        "}\n\n"
        "minorityDissent: distill the strongest counter-argument from Against/Neutral agents."
    )


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


def _contains_banned_recommendation_text(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BANNED_RECOMMENDATION_PHRASES)


def _has_specificity_signal(text: str) -> bool:
    """Require quantified or named-entity detail in recommendation copy."""
    if re.search(r"\$[\d,.]+[kmb]?", text, re.IGNORECASE):
        return True
    if re.search(r"\b\d+(\.\d+)?%|\b\d{1,3}(,\d{3})+\b|\bQ[1-4]\s*20\d{2}\b", text):
        return True
    if re.search(
        r"\b(by|before|within|in)\s+\d+\s+(days?|weeks?|months?|quarters?)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
        return True
    return False


def _is_actionable_recommendation(title: str, body: str) -> bool:
    combined = f"{title} {body}"
    if _contains_banned_recommendation_text(combined):
        return False
    return _has_specificity_signal(combined)


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
        if not _is_actionable_recommendation(title, body):
            logger.warning(
                "Rejected generic/vague recommendedAction: title=%r",
                title[:80],
            )
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
    """Align manager-assigned sentiments to agent roles in deliberation order."""
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

    quality_raw = payload.get("evidenceQualityScore") or payload.get("evidence_quality_score")
    evidence_quality_score = (
        int(quality_raw)
        if isinstance(quality_raw, (int, float))
        else 75
    )

    return ManagerSynthesis(
        consensus=consensus,
        agent_sentiments=agent_sentiments,
        recommended_actions=recommended_actions,
        minority_dissent=minority_dissent,
        confidence=0,
        evidence_quality_score=max(0, min(100, evidence_quality_score)),
    )


def format_persona_system_prompt(
    persona: dict,
    opposing_roles: list[str],
) -> str:
    """Build an adversarial institutional analyst prompt."""
    opponents = (
        "\n".join(f"- {role}" for role in opposing_roles)
        if opposing_roles
        else "- All other adversarial archetypes in this swarm"
    )

    stance = persona.get("adversarial_stance") or persona.get("debate_instruction") or ""

    return (
        f"Assigned adversarial role: {persona['role']}\n"
        f"Mandated stance: {stance}\n"
        f"Analyst: {persona['name']} | {persona['location']} | Risk: {persona['riskTolerance']}\n\n"
        f"Analytical lens: {persona['backstory']}\n"
        f"Known biases (lean into them): {persona['biases']}\n"
        f"Attack vector: {persona['debate_instruction']}\n\n"
        f"Opposing archetypes you must anticipate and dismantle:\n{opponents}\n\n"
        f"{CONVERSATIONAL_DEBATE_RULES}"
    )


def _format_prior_debate_transcript(prior_turns: list[dict[str, str]]) -> str:
    if not prior_turns:
        return "No prior speakers yet — you open the live debate."

    lines: list[str] = []
    for turn in prior_turns:
        name = turn.get("agentName") or "Agent"
        role = turn.get("role") or ""
        timestamp = turn.get("timestamp") or ""
        text = turn.get("text") or ""
        header = f"[{timestamp}] {name} ({role})".strip()
        lines.append(f"{header}:\n{text}")
    return "\n\n".join(lines)


async def chat_completion_with_fallback(
    client: AsyncOpenAI,
    requested_model: str | None,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    *,
    response_format: dict[str, str] | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str]:
    """
    Call OpenRouter with the requested model; on failure, retry once with fallback.
    Returns (response_text, model_slug_used).
    """
    primary = resolve_openrouter_model(requested_model)
    fallback = get_fallback_openrouter_model(requested_model)
    candidates = [primary] if primary == fallback else [primary, fallback]

    last_error: Exception | None = None

    for index, model in enumerate(candidates):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            completion = await client.chat.completions.create(**kwargs)
            text = (completion.choices[0].message.content or "").strip()

            if index > 0:
                logger.warning(
                    "LLM recovered via fallback model %s (requested=%r)",
                    model,
                    requested_model,
                )

            return text, model
        except Exception as exc:
            last_error = exc
            if index == 0 and len(candidates) > 1:
                logger.warning(
                    "LLM call failed for model=%s (%s); retrying with fallback=%s",
                    model,
                    exc,
                    fallback,
                )
                continue
            logger.exception("LLM call failed for model=%s", model)
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM call failed without raising a concrete error")


async def _chat_completion(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    text, _ = await chat_completion_with_fallback(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature,
    )
    return text


def _build_reflection_critique_prompt(persona: dict, draft: str) -> str:
    role = persona.get("role") or "Agent"
    stance = persona.get("adversarial_stance") or persona.get("debate_instruction") or ""

    return (
        f"You wrote this DRAFT argument:\n{draft}\n\n"
        f"Your assigned adversarial role: {role}\n"
        f"Mandated stance: {stance}\n\n"
        "Critique your draft ruthlessly as an internal red-team reviewer:\n"
        "- Am I defending my mandated stance aggressively enough?\n"
        "- Did I cite specific metrics, dates, or facts from the web data — not generic claims?\n"
        "- Did I anticipate and attack the opposing archetypes?\n"
        "- Is there hedging, vagueness, or missing evidence I must fix?\n\n"
        "Respond with 3-5 bullet critiques only. Be harsh and specific."
    )


def _build_reflection_final_prompt(draft: str, critique: str) -> str:
    return (
        f"DRAFT:\n{draft}\n\n"
        f"SELF-CRITIQUE:\n{critique}\n\n"
        "Revise into your FINAL argument.\n"
        "Output ONLY the final text: exactly 3–4 sentences of conversational, "
        "evidence-dense dialogue. Name a prior speaker if any spoke before you. "
        "No labels, bullets, or preamble."
    )


async def get_persona_debate_response(
    client: AsyncOpenAI,
    persona: dict,
    user_message: str,
    web_data: str,
    model: str,
    opposing_roles: list[str],
    prior_turns: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    """
    Run a sequential conversational debate turn with draft → critique → final loop.
    """
    role_name = str(persona.get("role") or persona.get("name") or "Agent")
    agent_name = str(persona.get("name") or role_name)
    prior_turns = prior_turns or []
    transcript_block = _format_prior_debate_transcript(prior_turns)

    system_prompt = (
        f"{format_persona_system_prompt(persona, opposing_roles)}\n\n"
        f"Live web data (includes deep page extracts where available):\n{web_data}"
    )

    premise_block = f"User premise:\n{user_message.strip()}"

    try:
        draft, model_used = await chat_completion_with_fallback(
            client,
            model,
            system_prompt,
            (
                f"{premise_block}\n\n"
                f"Debate transcript so far:\n{transcript_block}\n\n"
                f"You are {agent_name} ({role_name}). "
                "STEP 1 — DRAFT: Write your live debate turn in exactly 3–4 sentences. "
                "Reference a prior speaker by name if any spoke before you. "
                "Cite specific data from the web context."
            ),
            temperature=0.6,
        )

        critique, _ = await chat_completion_with_fallback(
            client,
            model_used,
            system_prompt,
            _build_reflection_critique_prompt(persona, draft),
            temperature=0.35,
        )

        final_text, _ = await chat_completion_with_fallback(
            client,
            model_used,
            system_prompt,
            _build_reflection_final_prompt(draft, critique),
            temperature=0.45,
        )

        if not final_text:
            final_text = draft

    except Exception as exc:
        logger.exception(
            "OpenRouter reflection loop failed for %s (requested=%r)",
            role_name,
            model,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate response for {role_name}: {exc}",
        ) from exc

    logger.info(
        "Adversarial reflection complete for %s via %s (final=%s chars)",
        role_name,
        model_used,
        len(final_text),
    )

    return {"role": role_name, "text": final_text}


async def get_manager_synthesis(
    client: AsyncOpenAI,
    premise: str,
    web_data: str,
    combined_perspectives: str,
    agent_roles: list[str],
    model: str,
) -> ManagerSynthesis:
    """Synthesize institutional consensus with strict JSON structure."""
    agent_count = max(1, len(agent_roles))
    manager_system = build_manager_system(agent_count)

    user_message = (
        f"User premise:\n{premise.strip()}\n\n"
        f"Live web data (includes deep page extracts where available):\n{web_data}\n\n"
        f"{agent_count} adversarial agent perspectives "
        "(classify each sentiment strictly from these texts; score evidence quality "
        "against the web data above, not agent rhetoric alone):\n"
        f"{combined_perspectives}\n\n"
        "Return JSON only."
    )

    try:
        raw_text, model_used = await chat_completion_with_fallback(
            client,
            model,
            manager_system,
            user_message,
            temperature=0.25,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.exception(
            "Manager synthesis failed after model fallback (requested=%r)",
            model,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate manager synthesis: {exc}",
        ) from exc

    parsed = parse_manager_synthesis_payload(raw_text, agent_roles)

    if parsed is not None:
        logger.info(
            "Manager synthesis via %s: sentiments=%s evidenceQuality=%s actions=%s",
            model_used,
            parsed.agent_sentiments,
            parsed.evidence_quality_score,
            len(parsed.recommended_actions),
        )
        return parsed

    logger.warning("Manager JSON parse failed; retrying without response_format")
    try:
        retry_text, retry_model = await chat_completion_with_fallback(
            client,
            model_used,
            manager_system,
            user_message,
            temperature=0.2,
        )
        parsed_retry = parse_manager_synthesis_payload(retry_text, agent_roles)
        if parsed_retry is not None:
            return parsed_retry
        raw_text = retry_text
        model_used = retry_model
    except Exception:
        logger.exception("Manager synthesis retry failed")

    logger.error("Manager synthesis fallback to plain text (%s chars)", len(raw_text))
    return ManagerSynthesis(
        consensus=raw_text or "Consensus unavailable.",
        agent_sentiments=["Neutral"] * agent_count,
        recommended_actions=[],
        minority_dissent="",
        confidence=75,
        evidence_quality_score=60,
    )
