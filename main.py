import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.orchestrator import run_swarm_ignite

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Shoal AI Engine", version="0.1.0")


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
async def ignite(payload: IgniteRequest) -> IgniteResponse:
    message_dicts = await run_swarm_ignite(payload.swarmId, payload.premise)

    messages = [
        DebateMessage(role=msg["role"], text=msg["text"])
        for msg in message_dicts
    ]

    return IgniteResponse(
        status="Swarm ignited",
        swarmId=payload.swarmId,
        messages=messages,
    )
