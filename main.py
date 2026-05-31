import asyncio
import os

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Shoal AI Engine", version="0.1.0")

MODEL = "deepseek/deepseek-chat"

MANAGER_SYSTEM = (
    "You are the Manager. Read the live web data and the 5 human agent "
    "perspectives, then provide a final 2-sentence definitive, data-backed "
    "consensus."
)

PERSONAS: list[tuple[str, str]] = [
    (
        "Budget-Conscious Buyer",
        "You are a highly frugal consumer. Focus strictly on price, "
        "maintenance costs, and ROI using the web data.",
    ),
    (
        "Performance Enthusiast",
        "You care only about specs, speed, tech, and premium features. "
        "Argue for the highest quality option using the web data.",
    ),
    (
        "Safety & Practicality Parent",
        "You are risk-averse. Focus on safety ratings, reliability, and "
        "everyday usability using the web data.",
    ),
    (
        "Brand Status Fanboy",
        "You care about luxury, brand perception, and social status. "
        "Argue based on prestige using the web data.",
    ),
    (
        "Skeptical Mechanic",
        "You are a cynical expert. Look for flaws, recalls, or hidden issues "
        "in the web data to warn the user.",
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


def search_web(query: str) -> str:
    """Scrape top 3 live web results for the premise."""
    try:
        results = DDGS().text(query, max_results=3)
    except Exception as exc:
        print(f"[search_web] DuckDuckGo error: {exc}")
        return "No live web data available."

    if not results:
        return "No live web results found for this query."

    chunks: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        body = result.get("body", "")
        href = result.get("href", "")
        chunks.append(
            f"[{index}] {title}\n{body}\nSource: {href}".strip(),
        )

    combined = "\n\n".join(chunks)
    print(f"[search_web] Retrieved {len(results)} results for: {query[:80]}")
    return combined


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

    web_data = await asyncio.to_thread(search_web, premise)
    print(f"[ignite] Web data preview:\n{web_data[:500]}...\n")

    agent_tasks = [
        get_agent_response(
            client,
            role_name,
            instruction,
            premise,
            web_data,
        )
        for role_name, instruction in PERSONAS
    ]

    agent_results = await asyncio.gather(*agent_tasks)

    combined_perspectives = "\n\n".join(
        f"{msg['role']}:\n{msg['text']}" for msg in agent_results
    )
    manager_user = (
        f"User premise:\n{premise}\n\n"
        f"Live web data:\n{web_data}\n\n"
        f"Human agent perspectives:\n{combined_perspectives}"
    )

    manager_result = await get_agent_response(
        client,
        "Manager",
        MANAGER_SYSTEM,
        manager_user,
        web_data,
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
