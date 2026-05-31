import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import Self

from services.dynamic_personas import MAX_DEBATE_AGENTS, clamp_agent_count
from services.metrics import (
    compute_confidence_from_synthesis,
    compute_extrapolated_votes_from_sentiments,
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
    agentCount: int = Field(default=5, ge=1, le=MAX_DEBATE_AGENTS)
    model: str | None = Field(default=None)
    swarmSize: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode="after")
    def resolve_swarm_size(self) -> Self:
        if self.swarmSize is None:
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
    id: int = Field(..., ge=1)
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


class RecommendedActionPayload(BaseModel):
    step: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


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
    swarmSize: int = Field(..., ge=1)
    agentCount: int = Field(..., ge=1)
    model: str | None = None
    recommendedActions: list[RecommendedActionPayload] = Field(default_factory=list)
    minorityDissent: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteResponse)
async def ignite(payload: IgniteRequest) -> IgniteResponse:
    started = time.perf_counter()

    debate_count = clamp_agent_count(payload.agentCount)

    result = await run_swarm_ignite(
        payload.swarmId,
        payload.premise,
        agent_count=debate_count,
        model=payload.model,
    )

    elapsed_sec = time.perf_counter() - started
    runtime = max(1, int(round(elapsed_sec)))

    messages = [
        DebateMessage(role=msg["role"], text=msg["text"])
        for msg in result.messages
    ]

    synthesis = result.manager_synthesis
    sentiments = synthesis.agent_sentiments

    confidence = compute_confidence_from_synthesis(
        sentiments,
        manager_confidence=synthesis.confidence,
        evidence_quality_score=synthesis.evidence_quality_score,
    )

    swarm_size = payload.swarmSize or debate_count
    votes_for, votes_against, votes_neutral = compute_extrapolated_votes_from_sentiments(
        sentiments,
        swarm_size=swarm_size,
    )

    executed_agents = result.executed_agent_count
    cost = compute_swarm_credits(executed_agents)

    recommended_actions = [
        RecommendedActionPayload(
            step=action.step,
            title=action.title,
            body=action.body,
        )
        for action in synthesis.recommended_actions
    ]

    minority_dissent = synthesis.minority_dissent or None

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
        swarmSize=swarm_size,
        agentCount=executed_agents,
        model=payload.model,
        recommendedActions=recommended_actions,
        minorityDissent=minority_dissent,
    )
