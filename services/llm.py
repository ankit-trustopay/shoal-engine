"""OpenRouter / DeepSeek LLM client and agent completion helpers."""

import logging
import os

from fastapi import HTTPException
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MODEL = "deepseek/deepseek-chat"
FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "deepseek/deepseek-chat")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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


async def get_agent_response(
    client: AsyncOpenAI,
    role_name: str,
    persona_instruction: str,
    user_message: str,
    web_data: str,
) -> dict[str, str]:
    system_prompt = (
        f"You are a {role_name}. {persona_instruction}\n\n"
        f"Here is the live web data: {web_data}. "
        "Use this real-world data to form your argument. "
        "Respond in exactly 2 sentences."
    )

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
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
    """Build an in-character system prompt from a dynamic persona profile."""
    return (
        f"You ARE {persona['name']}, a real person in a swarm debate. Stay fully in character.\n\n"
        f"Archetype role (Key Voice for the simulated crowd): {persona['role']}\n"
        f"Age: {persona['age']} | Location: {persona['location']} | Income: {persona['income']}\n"
        f"Marital status: {persona['maritalStatus']}\n"
        f"Cultural background: {persona['culturalBackground']}\n"
        f"IQ: {persona['iq']} | EQ: {persona['eq']} | Risk tolerance: {persona['riskTolerance']}\n\n"
        f"Life story: {persona['backstory']}\n"
        f"Biases: {persona['biases']}\n\n"
        f"How you argue: {persona['debate_instruction']}\n\n"
        "Use the live web data provided to ground your argument. "
        "Respond in exactly 2 sentences in first person as this person."
    )


async def get_persona_debate_response(
    client: AsyncOpenAI,
    persona: dict,
    user_message: str,
    web_data: str,
) -> dict[str, str]:
    """Run a debate turn fully in character for a dynamic persona."""
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
