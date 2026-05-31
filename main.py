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


class AgentProfilePayload(BaseModel):
    id: int = Field(..., ge=1, le=5)
    name: str
    role: str
    age: int = Field(..., ge=18, le=80)
    location: str
    income: str
    maritalStatus: str = ""
    culturalBackground: str = ""
    iq: int = Field(..., ge=90, le=140)
    eq: int = Field(..., ge=90, le=140)
    riskTolerance: str
    biases: str
    backstory: str


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
    agentProfiles: list[AgentProfilePayload]


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

    agent_profiles = [
        AgentProfilePayload(
            id=profile["id"],
            name=profile["name"],
            role=profile["role"],
            age=profile["age"],
            location=profile["location"],
            income=profile["income"],
            maritalStatus=profile.get("maritalStatus", ""),
            culturalBackground=profile.get("culturalBackground", ""),
            iq=profile["iq"],
            eq=profile["eq"],
            riskTolerance=profile["riskTolerance"],
            biases=profile["biases"],
            backstory=profile["backstory"],
        )
        for profile in result.agent_profiles
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
        agentProfiles=agent_profiles,
    )
