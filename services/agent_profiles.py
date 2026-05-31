"""Dynamic Synthetic Society agent profiles tailored to the user's premise."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, TypedDict

from fastapi import HTTPException
from openai import AsyncOpenAI

from services.llm import MODEL

logger = logging.getLogger(__name__)

RiskTolerance = Literal["Low", "Medium", "High"]


class AgentProfile(TypedDict):
    id: int
    name: str
    role: str
    age: int
    location: str
    income: str
    iq: int
    eq: int
    riskTolerance: RiskTolerance
    biases: str
    backstory: str


PERSONA_ROLES: list[str] = [
    "Budget-Conscious Buyer",
    "Performance Enthusiast",
    "Safety & Practicality Parent",
    "Brand Status Fanboy",
    "Skeptical Mechanic",
]

DEFAULT_RISK: dict[str, RiskTolerance] = {
    "Budget-Conscious Buyer": "Low",
    "Performance Enthusiast": "High",
    "Safety & Practicality Parent": "Low",
    "Brand Status Fanboy": "Medium",
    "Skeptical Mechanic": "Low",
}

PROFILE_GENERATION_SYSTEM = (
    "You are a demographic simulation architect for Shoal AI. "
    "Create hyper-realistic, diverse human personas for a modern global debate swarm. "
    "Profiles must feel specific to the user's dilemma — reference why each person cares. "
    "Use varied geographies (India, US, UK, Brazil, Nigeria, etc.), realistic names, "
    "and credible income strings (₹, $, £). IQ and EQ must be integers from 90 to 140. "
    "Return ONLY valid JSON — no markdown fences, no commentary."
)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _normalize_risk(value: Any, role: str) -> RiskTolerance:
    if isinstance(value, str):
        normalized = value.strip().capitalize()
        if normalized in ("Low", "Medium", "High"):
            return normalized  # type: ignore[return-value]
    return DEFAULT_RISK.get(role, "Medium")


def _coerce_profile(raw: dict[str, Any], index: int, premise: str) -> AgentProfile:
    role = PERSONA_ROLES[index]
    role_from_payload = str(raw.get("role") or role).strip()
    if role_from_payload not in PERSONA_ROLES:
        role_from_payload = role

    name = str(raw.get("name") or f"Agent {index + 1}").strip()
    location = str(raw.get("location") or "Global metro").strip()
    income = str(raw.get("income") or "Undisclosed").strip()
    biases = str(
        raw.get("biases")
        or f"Filters every decision about '{premise[:60]}' through personal cost anxiety.",
    ).strip()
    backstory = str(
        raw.get("backstory")
        or (
            f"They have followed '{premise[:80]}' closely because it affects their household "
            "budget and long-term plans."
        ),
    ).strip()

    return {
        "id": index + 1,
        "name": name,
        "role": role_from_payload,
        "age": _clamp_int(raw.get("age"), 22, 65, 34 + index),
        "location": location,
        "income": income,
        "iq": _clamp_int(raw.get("iq"), 90, 140, 108 + index * 3),
        "eq": _clamp_int(raw.get("eq"), 90, 140, 118 + index * 2),
        "riskTolerance": _normalize_risk(raw.get("riskTolerance"), role_from_payload),
        "biases": biases,
        "backstory": backstory,
    }


def _parse_profiles_json(text: str) -> list[dict[str, Any]] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(payload, dict) and isinstance(payload.get("agentProfiles"), list):
        return payload["agentProfiles"]
    if isinstance(payload, list):
        return payload
    return None


def build_fallback_profiles(premise: str) -> list[AgentProfile]:
    """Deterministic, premise-aware profiles when the LLM call fails."""
    topic = premise.strip()[:120] or "this decision"
    templates: list[dict[str, Any]] = [
        {
            "name": "Rajesh K.",
            "role": "Budget-Conscious Buyer",
            "age": 34,
            "location": "Pune, India",
            "income": "₹9.5L/year",
            "iq": 112,
            "eq": 128,
            "riskTolerance": "Low",
            "biases": "Assumes headline price hides maintenance traps and resale cliffs.",
            "backstory": (
                f"Rajesh is a mid-level IT analyst supporting two school-age kids. "
                f"He has been stress-testing spreadsheets for weeks because '{topic}' "
                "could lock up savings he planned for his daughter's coaching fees."
            ),
        },
        {
            "name": "Marcus L.",
            "role": "Performance Enthusiast",
            "age": 29,
            "location": "Austin, TX",
            "income": "$152K/year",
            "iq": 126,
            "eq": 98,
            "riskTolerance": "High",
            "biases": "Believes premium specs today prevent expensive regret tomorrow.",
            "backstory": (
                f"Marcus ships product experiments at a SaaS startup and treats '{topic}' "
                "like a benchmark race. He will pay more if the top-tier option clearly "
                "wins on speed, durability, or future-proofing."
            ),
        },
        {
            "name": "Priya S.",
            "role": "Safety & Practicality Parent",
            "age": 41,
            "location": "Suburban Chicago, IL",
            "income": "$94K/year",
            "iq": 119,
            "eq": 141,
            "riskTolerance": "Low",
            "biases": "Trusts institutional safety data over influencer enthusiasm.",
            "backstory": (
                f"Priya works night shifts as a nurse and drives her teens daily. "
                f"For her, '{topic}' is less about hype and more about reliability, "
                "warranty clarity, and what happens when something breaks on a school morning."
            ),
        },
        {
            "name": "Arjun M.",
            "role": "Brand Status Fanboy",
            "age": 27,
            "location": "Mumbai, India",
            "income": "₹18L/year",
            "iq": 107,
            "eq": 116,
            "riskTolerance": "Medium",
            "biases": "Equates visible brand prestige with professional credibility.",
            "backstory": (
                f"Arjun sells luxury retail and curates his LinkedIn persona carefully. "
                f"He cares how '{topic}' looks to clients and peers — the story must feel "
                "aspirational, not bargain-bin."
            ),
        },
        {
            "name": "David C.",
            "role": "Skeptical Mechanic",
            "age": 52,
            "location": "Detroit, MI",
            "income": "$71K/year",
            "iq": 131,
            "eq": 89,
            "riskTolerance": "Low",
            "biases": "Expects hidden defects until field data proves otherwise.",
            "backstory": (
                f"David has fixed failures others ignored for three decades. "
                f"He is dissecting '{topic}' with forum threads and recall histories, "
                "convinced polished marketing hides the expensive truth."
            ),
        },
    ]

    return [_coerce_profile(item, index, premise) for index, item in enumerate(templates)]


async def generate_agent_profiles(
    client: AsyncOpenAI,
    premise: str,
    web_data: str,
) -> list[AgentProfile]:
    """
    Generate five dilemma-tailored agent profiles via LLM, with robust fallback.
    """
    trimmed = premise.strip()
    roles_block = "\n".join(
        f"{index + 1}. {role}" for index, role in enumerate(PERSONA_ROLES)
    )

    user_prompt = (
        f"User dilemma:\n{trimmed}\n\n"
        f"Live web context (for realism):\n{web_data[:2000]}\n\n"
        "Create exactly 5 agent profiles in this order:\n"
        f"{roles_block}\n\n"
        "Return a JSON array of 5 objects. Each object MUST include:\n"
        '- id (integer 1-5)\n'
        '- name (realistic, e.g. "Rajesh K.")\n'
        '- role (exact persona name from the list above)\n'
        "- age (integer)\n"
        "- location (city + country)\n"
        '- income (string with currency, e.g. "₹12L/year")\n'
        "- iq (integer 90-140)\n"
        "- eq (integer 90-140)\n"
        '- riskTolerance ("Low", "Medium", or "High")\n'
        "- biases (one short sentence)\n"
        "- backstory (2-3 sentences; mention family/life and why they care about the dilemma)"
    )

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PROFILE_GENERATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.85,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        parsed = _parse_profiles_json(raw_text)

        if not parsed or len(parsed) < 5:
            logger.warning(
                "Agent profile LLM returned invalid JSON; using fallback (%s chars)",
                len(raw_text),
            )
            return build_fallback_profiles(trimmed)

        profiles = [
            _coerce_profile(item if isinstance(item, dict) else {}, index, trimmed)
            for index, item in enumerate(parsed[:5])
        ]

        # Enforce canonical ids and roles in order
        for index, profile in enumerate(profiles):
            profile["id"] = index + 1
            profile["role"] = PERSONA_ROLES[index]

        logger.info("Generated %s dynamic agent profiles for premise", len(profiles))
        return profiles

    except Exception as exc:
        logger.exception("Agent profile generation failed: %s", exc)
        if isinstance(exc, HTTPException):
            raise
        return build_fallback_profiles(trimmed)
