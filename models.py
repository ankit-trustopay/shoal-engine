from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from services.dynamic_personas import MAX_DEBATE_AGENTS


class IgniteRequest(BaseModel):
    """POST /ignite — legacy swarm ignition payload."""

    swarmId: str = Field(..., min_length=1)
    premise: str = Field(..., min_length=1)
    agentCount: int = Field(default=5, ge=1, le=MAX_DEBATE_AGENTS)
    model_tier: str = Field(default="lite")
    target_audience: Optional[str] = None
    price_point: Optional[str] = None
    marketing_budget: Optional[str] = None
    model: str | None = Field(default=None)
    swarmSize: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode="after")
    def resolve_swarm_size(self) -> Self:
        if self.swarmSize is None:
            self.swarmSize = self.agentCount
        return self


class DebateRequest(BaseModel):
    """POST /debate — must match shoal-web JSON body exactly."""

    debate_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    agent_count: int = Field(..., ge=1, le=10_000)
    model_mix: float = Field(default=0.0, ge=0, le=100)


class AgentStance(str, Enum):
    AGREES = "AGREES"
    DISAGREES = "DISAGREES"
    NEUTRAL = "NEUTRAL"


class FrictionMatrixEntry(BaseModel):
    name: str = Field(..., min_length=1)
    stance: AgentStance
    argument: str = Field(..., min_length=1)

    @field_validator("argument")
    @classmethod
    def trim_argument(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("argument must not be empty")
        return trimmed[:500]


class PreMortem(BaseModel):
    failure_modes: list[str] = Field(..., min_length=1, max_length=8)
    critical_unknowns: list[str] = Field(..., min_length=1, max_length=8)

    @field_validator("failure_modes", "critical_unknowns")
    @classmethod
    def non_empty_items(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items if item and item.strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty string")
        return [item[:500] for item in cleaned]


class ExecutionRoadmap(BaseModel):
    """Next steps — must be domain-specific to the user query (not generic SaaS playbooks)."""

    immediate_action: str = Field(..., min_length=1)
    plan_b: str = Field(..., min_length=1)

    @field_validator("immediate_action", "plan_b")
    @classmethod
    def trim_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("must not be empty")
        return trimmed[:1000]


class DebateAgentPosition(BaseModel):
    name: str = Field(..., min_length=1)
    position: str = Field(..., min_length=1)


class DebateCompletionPayload(BaseModel):
    """Canonical debate webhook body from the Python engine."""

    debate_id: str = Field(..., min_length=1)
    status: Literal["completed", "failed", "failure"] = "completed"
    verdict: str = Field(..., min_length=1)
    confidence: int = Field(..., ge=0, le=100)
    agents: list[DebateAgentPosition] = Field(default_factory=list)
    tldr: list[str] = Field(..., min_length=3, max_length=5)
    friction_matrix: list[FrictionMatrixEntry] = Field(..., min_length=1, max_length=50)
    pre_mortem: PreMortem
    execution_roadmap: ExecutionRoadmap
    runtime: int = Field(default=1, ge=1)
    cost: float = Field(default=0, ge=0)
    agent_count: int = Field(default=3, ge=1, alias="agentCount")

    model_config = {"populate_by_name": True}

    @field_validator("tldr")
    @classmethod
    def validate_tldr(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items if item and item.strip()]
        if len(cleaned) < 3:
            raise ValueError("tldr must contain at least 3 non-empty bullets")
        return [item[:400] for item in cleaned[:5]]
