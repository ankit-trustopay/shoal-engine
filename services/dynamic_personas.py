"""Context-aware persona generation from the user's premise."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, TypedDict

from fastapi import HTTPException
from openai import AsyncOpenAI

from services.llm import FAST_MODEL, MODEL

logger = logging.getLogger(__name__)

RiskTolerance = Literal["Low", "Medium", "High"]


class DynamicPersona(TypedDict):
    id: int
    name: str
    role: str
    debate_instruction: str
    age: int
    location: str
    income: str
    maritalStatus: str
    culturalBackground: str
    iq: int
    eq: int
    riskTolerance: RiskTolerance
    biases: str
    backstory: str


ARCHETYPE_LEADER_LABEL = "Archetype Leader"
KEY_VOICE_LABEL = "Key Voice"


def format_archetype_role(base_role: str, persona_id: int) -> str:
    """UI-facing role: one Key Voice plus Archetype Leaders for the simulated crowd."""
    base = base_role.strip()
    lowered = base.lower()
    if "archetype leader" in lowered or "key voice" in lowered:
        return base
    label = KEY_VOICE_LABEL if persona_id == 1 else ARCHETYPE_LEADER_LABEL
    return f"{base} · {label}"


PERSONA_GENERATION_SYSTEM = (
    "You are a market-research persona architect for Shoal AI. "
    "Given a user dilemma, invent exactly 5 Archetype Leaders who represent a much larger "
    "simulated crowd (typically 1,000 agents). These five are the Key Voices — not the entire swarm, "
    "but the deepest profiles standing in for the crowd. Personas must be hyper-localized to "
    "geography, industry, and culture mentioned in the premise — never generic Western defaults "
    "unless the premise is Western.\n\n"
    "Rules:\n"
    "- If India, Gujarat, or Indian cities appear, use specific Indian cities (e.g. Ahmedabad, Surat, Pune).\n"
    "- If real estate: include stakeholders like landlords, renters, brokers, or first-time buyers.\n"
    "- If healthcare/education/auto/etc., pick roles native to that domain.\n"
    "- culturalBackground: describe religion, regional identity, or community context respectfully "
    "as market-research vectors (no stereotypes or slurs).\n"
    "- maritalStatus: realistic (e.g. Single, Married, Married with 2 kids, Widowed).\n"
    "- iq and eq: integers 90-140, consistent with education and life story.\n"
    "- debate_instruction: 2 sentences on how THIS archetype leader argues for their slice of the crowd.\n"
    '- role: specific stakeholder title (e.g. "Ahmedabad Landlord") — do NOT append "Archetype Leader" in JSON; we label in post-processing.\n'
    "Return ONLY a JSON array of 5 objects — no markdown."
)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _normalize_risk(value: Any) -> RiskTolerance:
    if isinstance(value, str):
        normalized = value.strip().capitalize()
        if normalized in ("Low", "Medium", "High"):
            return normalized  # type: ignore[return-value]
    return "Medium"


def _parse_personas_json(text: str) -> list[dict[str, Any]] | None:
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

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("personas"), list):
        return payload["personas"]
    return None


def _premise_hints(premise: str) -> str:
    lower = premise.lower()
    hints: list[str] = []
    if any(token in lower for token in ("india", "gujarat", "ahmedabad", "surat", "mumbai", "₹")):
        hints.append("Use Indian cities, ₹ income, and locally credible names.")
    if any(token in lower for token in ("real estate", "property", "rent", "landlord", "flat")):
        hints.append("Include property-market stakeholders (buyer, seller, renter, landlord, broker).")
    if any(token in lower for token in ("car", "vehicle", "ev ", "automotive")):
        hints.append("Include buyers, mechanics, fleet owners, or safety-focused parents as relevant.")
    return " ".join(hints)


def _coerce_persona(raw: dict[str, Any], index: int, premise: str) -> DynamicPersona | None:
    role = str(raw.get("role") or raw.get("title") or "").strip()
    name = str(raw.get("name") or "").strip()
    debate_instruction = str(
        raw.get("debate_instruction") or raw.get("instruction") or "",
    ).strip()

    if not role or not name:
        return None

    if not debate_instruction:
        debate_instruction = (
            f"Argue from the perspective of a {role} directly affected by: {premise[:100]}."
        )

    persona_id = index + 1

    return {
        "id": persona_id,
        "name": name,
        "role": format_archetype_role(role, persona_id),
        "debate_instruction": debate_instruction,
        "age": _clamp_int(raw.get("age"), 22, 70, 32 + index),
        "location": str(raw.get("location") or "Unknown").strip(),
        "income": str(raw.get("income") or "Not disclosed").strip(),
        "maritalStatus": str(
            raw.get("maritalStatus") or raw.get("marital_status") or "Not specified",
        ).strip(),
        "culturalBackground": str(
            raw.get("culturalBackground")
            or raw.get("cultural_background")
            or "Not specified",
        ).strip(),
        "iq": _clamp_int(raw.get("iq"), 90, 140, 108 + index * 2),
        "eq": _clamp_int(raw.get("eq"), 90, 140, 115 + index * 3),
        "riskTolerance": _normalize_risk(raw.get("riskTolerance")),
        "biases": str(
            raw.get("biases")
            or f"Frames '{premise[:50]}' through the lens of a {role}.",
        ).strip(),
        "backstory": str(
            raw.get("backstory")
            or f"{name} is deeply invested in this decision because it affects daily life and family obligations.",
        ).strip(),
    }


def build_fallback_personas(premise: str) -> list[DynamicPersona]:
    """Heuristic fallback when LLM persona generation fails."""
    lower = premise.lower()
    india = any(
        token in lower
        for token in ("india", "gujarat", "ahmedabad", "surat", "vadodara", "₹")
    )
    real_estate = any(
        token in lower
        for token in ("real estate", "property", "rent", "flat", "landlord", "housing")
    )

    if india and real_estate:
        templates = [
            {
                "name": "Ketan P.",
                "role": "Ahmedabad Landlord",
                "debate_instruction": "Defend rental yield and capital appreciation in Gujarat tier-1 corridors.",
                "location": "Ahmedabad, Gujarat, India",
                "income": "₹22L/year",
                "maritalStatus": "Married with 2 kids",
                "culturalBackground": "Gujarati Patel family; values property as generational wealth",
            },
            {
                "name": "Fatima S.",
                "role": "Surat Renter & SME Owner",
                "debate_instruction": "Push for affordable lease terms and infrastructure near textile hubs.",
                "location": "Surat, Gujarat, India",
                "income": "₹11L/year",
                "maritalStatus": "Married",
                "culturalBackground": "Sunni Muslim, Surati business community",
            },
            {
                "name": "Vikram R.",
                "role": "First-Time Home Buyer",
                "debate_instruction": "Focus on EMI burden, RERA compliance, and commute to GIFT City.",
                "location": "Gandhinagar, Gujarat, India",
                "income": "₹14L/year",
                "maritalStatus": "Married, expecting first child",
                "culturalBackground": "Hindu, Gujarati; joint-family savings norms",
            },
            {
                "name": "Neha M.",
                "role": "Property Broker",
                "debate_instruction": "Highlight transaction velocity and developer track records.",
                "location": "Vadodara, Gujarat, India",
                "income": "₹16L/year (commission-heavy)",
                "maritalStatus": "Single",
                "culturalBackground": "Urban Gujarati professional",
            },
            {
                "name": "Hassan K.",
                "role": "Skeptical Civil Engineer",
                "debate_instruction": "Warn about construction quality, water logging, and approval risks.",
                "location": "Rajkot, Gujarat, India",
                "income": "₹9L/year",
                "maritalStatus": "Married with 1 kid",
                "culturalBackground": "Gujarati Muslim; technical skepticism",
            },
        ]
    elif india:
        templates = [
            {
                "name": "Rajesh K.",
                "role": "Budget-Conscious IT Analyst",
                "debate_instruction": "Stress total cost of ownership and family savings goals.",
                "location": "Pune, Maharashtra, India",
                "income": "₹9.5L/year",
                "maritalStatus": "Married with 2 kids",
                "culturalBackground": "Marathi-speaking middle class",
            },
            {
                "name": "Priya N.",
                "role": "Risk-Averse Parent",
                "debate_instruction": "Prioritize safety, reliability, and school commute practicality.",
                "location": "Bengaluru, Karnataka, India",
                "income": "₹12L/year",
                "maritalStatus": "Married with 2 school-age children",
                "culturalBackground": "Tamil Brahmin household",
            },
            {
                "name": "Arjun M.",
                "role": "Status-Conscious Professional",
                "debate_instruction": "Argue for premium options that signal career success.",
                "location": "Mumbai, India",
                "income": "₹18L/year",
                "maritalStatus": "Single",
                "culturalBackground": "Urban cosmopolitan",
            },
            {
                "name": "Sunita D.",
                "role": "Small Business Owner",
                "debate_instruction": "Evaluate cash-flow impact and GST/documentation clarity.",
                "location": "Jaipur, Rajasthan, India",
                "income": "₹7L/year",
                "maritalStatus": "Widowed, supports parents",
                "culturalBackground": "Rajasthani trading family",
            },
            {
                "name": "Imran Q.",
                "role": "Skeptical Field Expert",
                "debate_instruction": "Challenge marketing claims with on-the-ground experience.",
                "location": "Hyderabad, Telangana, India",
                "income": "₹10L/year",
                "maritalStatus": "Married",
                "culturalBackground": "Hyderabadi Muslim professional",
            },
        ]
    else:
        templates = [
            {
                "name": "Marcus L.",
                "role": "Performance Enthusiast",
                "debate_instruction": "Champion top specs and long-term quality over upfront price.",
                "location": "Austin, TX, USA",
                "income": "$152K/year",
                "maritalStatus": "Single",
                "culturalBackground": "White American tech worker",
            },
            {
                "name": "Priya S.",
                "role": "Safety-Focused Parent",
                "debate_instruction": "Prioritize reliability and worst-case risk mitigation.",
                "location": "Suburban Chicago, IL, USA",
                "income": "$94K/year",
                "maritalStatus": "Married with 2 teens",
                "culturalBackground": "Indian-American nurse",
            },
            {
                "name": "James T.",
                "role": "Budget Shopper",
                "debate_instruction": "Attack hidden fees and argue for delaying until value is proven.",
                "location": "Columbus, OH, USA",
                "income": "$58K/year",
                "maritalStatus": "Married with 1 child",
                "culturalBackground": "Midwest working class",
            },
            {
                "name": "Elena R.",
                "role": "Brand & Lifestyle Buyer",
                "debate_instruction": "Focus on perception, design, and social signaling.",
                "location": "Miami, FL, USA",
                "income": "$110K/year",
                "maritalStatus": "Divorced",
                "culturalBackground": "Cuban-American marketing manager",
            },
            {
                "name": "David C.",
                "role": "Skeptical Industry Veteran",
                "debate_instruction": "Expose flaws, recalls, and fine-print traps from experience.",
                "location": "Detroit, MI, USA",
                "income": "$71K/year",
                "maritalStatus": "Married",
                "culturalBackground": "African-American automotive technician",
            },
        ]

    personas: list[DynamicPersona] = []
    for index, template in enumerate(templates):
        raw = {
            **template,
            "iq": 108 + index * 4,
            "eq": 112 + index * 5,
            "riskTolerance": ["Low", "Medium", "High", "Medium", "Low"][index],
            "biases": f"Strongly filters '{premise[:40]}' through personal lived experience.",
            "backstory": (
                f"{template['name']} lives in {template['location']} and is directly affected by "
                f"the dilemma: {premise[:90]}."
            ),
        }
        persona = _coerce_persona(raw, index, premise)
        if persona:
            personas.append(persona)

    return personas


async def generate_dynamic_personas(
    client: AsyncOpenAI,
    premise: str,
    web_data: str,
) -> list[DynamicPersona]:
    """
    Fast LLM pass: 5 context-aware personas with deep demographic vectors.
    """
    trimmed = premise.strip()
    hints = _premise_hints(trimmed)

    user_prompt = (
        f"User dilemma:\n{trimmed}\n\n"
        f"Context hints: {hints or 'Infer locale and stakeholders from the dilemma.'}\n\n"
        f"Optional web context:\n{web_data[:1200]}\n\n"
        "Return a JSON array of exactly 5 persona objects. Each object MUST include:\n"
        '- id (1-5)\n'
        '- name (realistic full name with initial)\n'
        '- role (specific stakeholder title, NOT generic)\n'
        '- debate_instruction (2 sentences)\n'
        "- age (integer)\n"
        "- location (specific city + region + country)\n"
        '- income (localized string, e.g. "₹12L/year")\n'
        '- maritalStatus (string)\n'
        '- culturalBackground (respectful market-research description)\n'
        "- iq (90-140)\n"
        "- eq (90-140)\n"
        '- riskTolerance ("Low", "Medium", or "High")\n'
        "- biases (one sentence)\n"
        "- backstory (2-3 sentences)"
    )

    try:
        completion = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": PERSONA_GENERATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.75,
            max_tokens=2500,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        parsed = _parse_personas_json(raw_text)

        if not parsed or len(parsed) < 5:
            logger.warning("Persona JSON invalid; using fallback (%s chars)", len(raw_text))
            return build_fallback_personas(trimmed)

        personas: list[DynamicPersona] = []
        for index, item in enumerate(parsed[:5]):
            if not isinstance(item, dict):
                continue
            persona = _coerce_persona(item, index, trimmed)
            if persona:
                personas.append(persona)

        if len(personas) < 5:
            logger.warning("Only %s personas parsed; using fallback", len(personas))
            return build_fallback_personas(trimmed)

        for index, persona in enumerate(personas):
            persona["id"] = index + 1

        logger.info("Generated %s dynamic personas for premise", len(personas))
        return personas

    except Exception as exc:
        logger.exception("Dynamic persona generation failed: %s", exc)
        if isinstance(exc, HTTPException):
            raise
        return build_fallback_personas(trimmed)


def persona_to_agent_profile(persona: DynamicPersona) -> dict[str, Any]:
    """Strip debate-only fields for API agentProfiles payload."""
    return {
        "id": persona["id"],
        "name": persona["name"],
        "role": persona["role"],
        "age": persona["age"],
        "location": persona["location"],
        "income": persona["income"],
        "maritalStatus": persona["maritalStatus"],
        "culturalBackground": persona["culturalBackground"],
        "iq": persona["iq"],
        "eq": persona["eq"],
        "riskTolerance": persona["riskTolerance"],
        "biases": persona["biases"],
        "backstory": persona["backstory"],
    }
