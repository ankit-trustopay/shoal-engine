import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Shoal AI Engine", version="0.1.0")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = (
    "You are a Financial Skeptic AI agent. "
    "Give a 2-sentence skeptical argument challenging the user's premise."
)


class IgniteRequest(BaseModel):
    swarmId: str = Field(..., min_length=1)
    premise: str = Field(..., min_length=1)


class IgniteResponse(BaseModel):
    status: str
    swarmId: str
    response: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
def ignite(payload: IgniteRequest) -> IgniteResponse:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured",
        )

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.premise},
            ],
        )
    except Exception as exc:
        print(f"[ignite] OpenRouter error for swarm {payload.swarmId}:", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate AI response",
        ) from exc

    ai_message = completion.choices[0].message.content or ""
    print(f"[ignite] swarmId={payload.swarmId} AI response:\n{ai_message}")

    return IgniteResponse(
        status="Swarm ignited",
        swarmId=payload.swarmId,
        response=ai_message,
    )
