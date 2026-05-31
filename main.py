import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.metrics import (
    compute_confidence,
    compute_mock_cost,
    compute_vote_distribution,
)
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


class EvidencePayload(BaseModel):
    title: str
    source: str
    url: str
    snippet: str


class IgniteResponse(BaseModel):
    status: str
    swarmId: str
    messages: list[DebateMessage]
    confidence: int = Field(..., ge=0, le=100)
    votesFor: int = Field(..., ge=0)
    votesAgainst: int = Field(..., ge=0)
    votesNeutral: int = Field(..., ge=0)
    runtime: int = Field(..., ge=0)
    cost: float = Field(..., ge=0)
    evidence: list[EvidencePayload]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
async def ignite(payload: IgniteRequest) -> IgniteResponse:
    started = time.perf_counter()

    result = await run_swarm_ignite(payload.swarmId, payload.premise)

    elapsed_sec = time.perf_counter() - started
    runtime = max(1, int(round(elapsed_sec)))

    messages = [
        DebateMessage(role=msg["role"], text=msg["text"])
        for msg in result.messages
    ]

    manager_text = next(
        (msg.text for msg in messages if msg.role == "Manager"),
        "",
    )

    confidence = compute_confidence(manager_text)
    votes_for, votes_against, votes_neutral = compute_vote_distribution(
        manager_text,
    )
    cost = compute_mock_cost(runtime, len(messages))

    evidence = [
        EvidencePayload(
            title=item["title"],
            source=item["source"],
            url=item["url"],
            snippet=item["snippet"],
        )
        for item in result.evidence
    ]

    return IgniteResponse(
        status="Swarm ignited",
        swarmId=payload.swarmId,
        messages=messages,
        confidence=confidence,
        votesFor=votes_for,
        votesAgainst=votes_against,
        votesNeutral=votes_neutral,
        runtime=runtime,
        cost=cost,
        evidence=evidence,
    )
