import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Shoal AI Engine", version="0.1.0")

MODEL = "deepseek/deepseek-chat"

SKEPTIC_SYSTEM = (
    "You are a Skeptic. Challenge the user's premise in 2 sentences."
)
EXPERT_SYSTEM = (
    "You are a Domain Expert. Read the Skeptic's argument and counter it "
    "with hard facts in 2 sentences."
)
MANAGER_SYSTEM = (
    "You are the Manager. Synthesize the debate and give a final 2-sentence verdict."
)


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


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured",
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_agent(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        print("[ignite] OpenRouter error:", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate AI response",
        ) from exc

    return (completion.choices[0].message.content or "").strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
def ignite(payload: IgniteRequest) -> IgniteResponse:
    client = get_client()
    premise = payload.premise.strip()

    print(f"[ignite] Starting multi-agent debate for swarm {payload.swarmId}")

    skeptic_text = call_agent(client, SKEPTIC_SYSTEM, premise)
    print(f"[ignite] Skeptic:\n{skeptic_text}")

    expert_user = (
        f"User premise:\n{premise}\n\n"
        f"Skeptic's argument:\n{skeptic_text}"
    )
    expert_text = call_agent(client, EXPERT_SYSTEM, expert_user)
    print(f"[ignite] Expert:\n{expert_text}")

    manager_user = (
        f"User premise:\n{premise}\n\n"
        f"Skeptic:\n{skeptic_text}\n\n"
        f"Expert:\n{expert_text}"
    )
    manager_text = call_agent(client, MANAGER_SYSTEM, manager_user)
    print(f"[ignite] Manager:\n{manager_text}")

    messages = [
        DebateMessage(role="Skeptic", text=skeptic_text),
        DebateMessage(role="Expert", text=expert_text),
        DebateMessage(role="Manager", text=manager_text),
    ]

    return IgniteResponse(
        status="Swarm ignited",
        swarmId=payload.swarmId,
        messages=messages,
    )
