"""Adversarial persona generation — strict conflicting roles scaled to agentCount."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, TypedDict

from fastapi import HTTPException
from openai import AsyncOpenAI

from services.model_router import DEFAULT_OPENROUTER_MODEL, resolve_openrouter_model

logger = logging.getLogger(__name__)

RiskTolerance = Literal["Low", "Medium", "High"]

MAX_DEBATE_AGENTS = 5


class AdversarialArchetype(TypedDict):
    slug: str
    role: str
    stance: str
    debate_instruction: str
    risk_tolerance: RiskTolerance
    biases: str


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
    adversarial_stance: str


ADVERSARIAL_ARCHETYPES: list[AdversarialArchetype] = [
    {
        "slug": "aggressive_bull",
        "role": "The Aggressive Bull",
        "stance": "FOR — prosecute the upside case relentlessly; treat caution as paralysis.",
        "debate_instruction": (
            "Champion immediate action on the premise. Weaponize growth metrics, momentum, "
            "and competitive timing from the web data. Attack delay as value destruction."
        ),
        "risk_tolerance": "High",
        "biases": "Overweights upside tails; discounts tail-risk and execution friction.",
    },
    {
        "slug": "extreme_bear",
        "role": "The Extreme Bear / Skeptic",
        "stance": "AGAINST — assume the premise fails under stress; prosecute downside first.",
        "debate_instruction": (
            "Dismantle the premise with failure modes, hidden liabilities, and adverse scenarios "
            "from the web data. Treat bullish narratives as marketing, not evidence."
        ),
        "risk_tolerance": "Low",
        "biases": "Anchors on worst-case outcomes; treats optimism as unpriced risk.",
    },
    {
        "slug": "regulatory_auditor",
        "role": "The Regulatory Auditor",
        "stance": "NEUTRAL / BLOCK — compliance, governance, and regulatory veto power.",
        "debate_instruction": (
            "Interrogate legal exposure, licensing, data/privacy constraints, and policy shifts "
            "in the web data. Block approval until regulatory landmines are bounded."
        ),
        "risk_tolerance": "Low",
        "biases": "Prioritizes enforceability and audit trails over strategic upside.",
    },
    {
        "slug": "capital_allocator",
        "role": "The Capital Allocator",
        "stance": "AGAINST / WAIT — ROI hurdle, timing, and opportunity cost.",
        "debate_instruction": (
            "Stress-test NPV, payback period, and capital efficiency using web data. "
            "Argue capital is better deployed elsewhere until hurdles are cleared."
        ),
        "risk_tolerance": "Medium",
        "biases": "Frames every decision as a portfolio trade-off against alternatives.",
    },
    {
        "slug": "strategic_contrarian",
        "role": "The Strategic Contrarian",
        "stance": "MIXED — attacks consensus logic and second-order effects others ignore.",
        "debate_instruction": (
            "Expose flawed assumptions in both bull and bear cases using web data. "
            "Force the room to confront non-obvious causal chains and reflexive market dynamics."
        ),
        "risk_tolerance": "Medium",
        "biases": "Distrusts narrative coherence; hunts disconfirming evidence.",
    },
]

PERSONA_GENERATION_SYSTEM = (
    "You are an adversarial debate architect for Shoal AI institutional swarms.\n"
    "Each persona slot is PRE-ASSIGNED a fixed adversarial archetype — do NOT change the role or stance.\n"
    "Your job: localize demographics (name, city, income, culture) to the user's premise while preserving "
    "the assigned adversarial mandate.\n\n"
    "Rules:\n"
    "- Honor geography/industry cues in the premise (India, EU, SaaS, M&A, etc.).\n"
    "- Names and locales must feel credible for that context.\n"
    "- debate_instruction must reinforce the assigned stance with premise-specific data hooks.\n"
    "- Personas must CONFLICT — never converge or soften their assigned stance.\n"
    "Return ONLY a JSON array — no markdown."
)


def clamp_agent_count(agent_count: int) -> int:
    return max(1, min(MAX_DEBATE_AGENTS, int(agent_count)))


def selected_archetypes(agent_count: int) -> list[AdversarialArchetype]:
    count = clamp_agent_count(agent_count)
    return ADVERSARIAL_ARCHETYPES[:count]


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _normalize_risk(value: Any, fallback: RiskTolerance = "Medium") -> RiskTolerance:
    if isinstance(value, str):
        normalized = value.strip().capitalize()
        if normalized in ("Low", "Medium", "High"):
            return normalized  # type: ignore[return-value]
    return fallback


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
        hints.append("Anchor to property-market stakeholders and local regulation.")
    if any(token in lower for token in ("eu ", "europe", "gdpr", "ai act")):
        hints.append("Emphasize EU regulatory and cross-border constraints.")
    return " ".join(hints)


def _fallback_name(index: int) -> str:
    names = ["Alex Chen", "Maria Santos", "James Okonkwo", "Elena Rossi", "David Park"]
    return names[index % len(names)]


def _fallback_location(premise: str, index: int) -> str:
    lower = premise.lower()
    if any(token in lower for token in ("india", "gujarat", "₹")):
        cities = ["Mumbai, India", "Bengaluru, India", "Ahmedabad, India", "Delhi, India", "Hyderabad, India"]
    elif any(token in lower for token in ("eu", "europe", "gdpr")):
        cities = ["Berlin, Germany", "Paris, France", "Amsterdam, Netherlands", "Dublin, Ireland", "Stockholm, Sweden"]
    else:
        cities = ["New York, NY, USA", "London, UK", "Singapore", "Toronto, Canada", "Sydney, Australia"]
    return cities[index % len(cities)]


def build_adversarial_persona(
    archetype: AdversarialArchetype,
    index: int,
    premise: str,
    raw: dict[str, Any] | None = None,
) -> DynamicPersona:
    """Materialize one adversarial persona from archetype + optional LLM enrichment."""
    raw = raw or {}
    persona_id = index + 1
    name = str(raw.get("name") or _fallback_name(index)).strip()
    location = str(raw.get("location") or _fallback_location(premise, index)).strip()
    debate_instruction = str(
        raw.get("debate_instruction") or archetype["debate_instruction"],
    ).strip()

    return {
        "id": persona_id,
        "name": name,
        "role": archetype["role"],
        "adversarial_stance": archetype["stance"],
        "debate_instruction": debate_instruction,
        "age": _clamp_int(raw.get("age"), 28, 62, 34 + index * 3),
        "location": location,
        "income": str(raw.get("income") or "Not disclosed").strip(),
        "maritalStatus": str(
            raw.get("maritalStatus") or raw.get("marital_status") or "Not specified",
        ).strip(),
        "culturalBackground": str(
            raw.get("culturalBackground")
            or raw.get("cultural_background")
            or "Institutional analyst background",
        ).strip(),
        "iq": _clamp_int(raw.get("iq"), 95, 140, 112 + index * 2),
        "eq": _clamp_int(raw.get("eq"), 95, 140, 118 + index),
        "riskTolerance": _normalize_risk(
            raw.get("riskTolerance"),
            archetype["risk_tolerance"],
        ),
        "biases": str(raw.get("biases") or archetype["biases"]).strip(),
        "backstory": str(
            raw.get("backstory")
            or (
                f"{name} is a tier-1 analyst assigned to prosecute the "
                f"{archetype['role']} view on: {premise[:120]}"
            ),
        ).strip(),
    }


def build_fallback_personas(premise: str, agent_count: int) -> list[DynamicPersona]:
    """Deterministic adversarial panel when LLM persona generation fails."""
    archetypes = selected_archetypes(agent_count)
    return [
        build_adversarial_persona(archetype, index, premise)
        for index, archetype in enumerate(archetypes)
    ]


def _build_generation_prompt(
    premise: str,
    web_data: str,
    agent_count: int,
) -> str:
    archetypes = selected_archetypes(agent_count)
    hints = _premise_hints(premise)
    slots = "\n".join(
        f"  Slot {index + 1}: {item['role']} — {item['stance']}"
        for index, item in enumerate(archetypes)
    )

    return (
        f"User dilemma:\n{premise.strip()}\n\n"
        f"Context hints: {hints or 'Infer locale and domain from the dilemma.'}\n\n"
        f"Optional web context:\n{web_data[:1200]}\n\n"
        f"Generate EXACTLY {len(archetypes)} persona objects for these fixed adversarial slots:\n"
        f"{slots}\n\n"
        "Each object MUST include:\n"
        "- id (matching slot number)\n"
        "- name\n"
        "- role (copy the assigned archetype role EXACTLY)\n"
        "- debate_instruction (2 sentences, aggressively defend the assigned stance)\n"
        "- age, location, income, maritalStatus, culturalBackground\n"
        "- iq (90-140), eq (90-140), riskTolerance, biases, backstory (2-3 sentences)\n"
        f"Return a JSON array of exactly {len(archetypes)} objects."
    )


async def generate_dynamic_personas(
    client: AsyncOpenAI,
    premise: str,
    web_data: str,
    agent_count: int,
    model: str | None = None,
) -> list[DynamicPersona]:
    """Generate exactly agent_count adversarial personas with conflicting mandates."""
    trimmed = premise.strip()
    count = clamp_agent_count(agent_count)
    archetypes = selected_archetypes(count)
    resolved_model = resolve_openrouter_model(model)
    fast_model = resolve_openrouter_model(
        model or DEFAULT_OPENROUTER_MODEL,
    )

    user_prompt = _build_generation_prompt(trimmed, web_data, count)

    try:
        completion = await client.chat.completions.create(
            model=fast_model,
            messages=[
                {"role": "system", "content": PERSONA_GENERATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.65,
            max_tokens=1800 + count * 400,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        parsed = _parse_personas_json(raw_text)

        if not parsed or len(parsed) < count:
            logger.warning(
                "Adversarial persona JSON invalid (%s items); using fallback",
                len(parsed or []),
            )
            return build_fallback_personas(trimmed, count)

        personas: list[DynamicPersona] = []
        for index, archetype in enumerate(archetypes):
            item = parsed[index] if index < len(parsed) and isinstance(parsed[index], dict) else {}
            personas.append(build_adversarial_persona(archetype, index, trimmed, item))

        logger.info(
            "Generated %s adversarial personas (requested=%s model=%s)",
            len(personas),
            count,
            resolved_model,
        )
        return personas

    except Exception as exc:
        logger.exception("Adversarial persona generation failed: %s", exc)
        if isinstance(exc, HTTPException):
            raise
        return build_fallback_personas(trimmed, count)


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
