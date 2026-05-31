import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Shoal AI Engine", version="0.1.0")

MODEL = "deepseek/deepseek-chat"

MANAGER_SYSTEM = (
    "You are the Manager. Read these 5 agent perspectives and provide a "
    "final 2-sentence definitive verdict."
)

PERSONAS: list[tuple[str, str]] = [
    (
        "Financial Skeptic",
        "Challenge the financial viability of the premise in 2 sentences.",
    ),
    (
        "Domain Expert",
        "Provide technical/historical facts about the premise in 2 sentences.",
    ),
    (
        "Risk Analyst",
        "Identify the biggest tail-risk or downside of the premise in 2 sentences.",
    ),
    (
        "Consumer Voice",
        "Explain how average people or end-users will react to this in 2 sentences.",
    ),
    (
        "Optimist",
        "Highlight the massive upside and bullish case for this premise in 2 sentences.",
    ),
]


class IgniteRequest(BaseModel):
    swarmId: str = Field(..., min_length=1)
    premise: str = Field(..., min_length=1)


class DebateMessage(BaseModel):
    role: str
    text: str


class IgniteResponse(BaseModel):
    status: str
    swarmId: str
    messages: list[DebateMessage]


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured",
        )
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


async def get_agent_response(
    client: AsyncOpenAI,
    role_name: str,
    system_prompt: str,
    user_message: str,
) -> dict[str, str]:
    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        print(f"[ignite] OpenRouter error ({role_name}):", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate response for {role_name}",
        ) from exc

    response_text = (completion.choices[0].message.content or "").strip()
    print(f"[ignite] {role_name}:\n{response_text}\n")

    return {"role": role_name, "text": response_text}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
async def ignite(payload: IgniteRequest) -> IgniteResponse:
    client = get_client()
    premise = payload.premise.strip()

    print(
        f"[ignite] Starting parallel orchestration for swarm {payload.swarmId}",
    )

    agent_tasks = [
        get_agent_response(
            client,
            role_name,
            f"You are a {role_name}. {instruction}",
            premise,
        )
        for role_name, instruction in PERSONAS
    ]

    agent_results = await asyncio.gather(*agent_tasks)

    combined_perspectives = "\n\n".join(
        f"{msg['role']}:\n{msg['text']}" for msg in agent_results
    )
    manager_user = (
        f"User premise:\n{premise}\n\n"
        f"Agent perspectives:\n{combined_perspectives}"
    )

    manager_result = await get_agent_response(
        client,
        "Manager",
        MANAGER_SYSTEM,
        manager_user,
    )

    messages = [
        DebateMessage(role=msg["role"], text=msg["text"])
        for msg in [*agent_results, manager_result]
    ]

    return IgniteResponse(
        status="Swarm ignited",
        swarmId=payload.swarmId,
        messages=messages,
    )
