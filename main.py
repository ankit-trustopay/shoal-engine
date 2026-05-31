import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import Self

from services.metrics import (
    DEFAULT_SWARM_SIZE,
    compute_confidence,
    compute_extrapolated_votes,
    compute_swarm_credits,
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
    swarmSize: int = Field(default=DEFAULT_SWARM_SIZE, ge=1, le=10_000)
    agentCount: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode="after")
    def resolve_swarm_size(self) -> Self:
        if self.agentCount is not None:
            self.swarmSize = self.agentCount
        return self


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
    swarmSize: int = Field(default=DEFAULT_SWARM_SIZE, ge=1)


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

    leader_texts = [
        msg.text for msg in messages if msg.role != "Manager"
    ]

    confidence = compute_confidence(manager_text)
    votes_for, votes_against, votes_neutral = compute_extrapolated_votes(
        confidence,
        manager_text,
        leader_texts,
        swarm_size=payload.swarmSize,
    )
    cost = compute_swarm_credits(payload.swarmSize)

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
        swarmSize=payload.swarmSize,
    )
