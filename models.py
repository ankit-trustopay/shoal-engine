from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Self

from services.dynamic_personas import MAX_DEBATE_AGENTS


class DebateRequest(BaseModel):
    swarmId: str = Field(..., min_length=1)
    premise: str = Field(..., min_length=1)
    agentCount: int = Field(default=5, ge=1, le=MAX_DEBATE_AGENTS)

    # New optional variables from the frontend (safe to ignore in orchestration for now).
    model_tier: str = Field(default="lite")
    target_audience: Optional[str] = None
    price_point: Optional[str] = None
    marketing_budget: Optional[str] = None

    # Back-compat fields (older clients).
    model: str | None = Field(default=None)
    swarmSize: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode="after")
    def resolve_swarm_size(self) -> Self:
        if self.swarmSize is None:
            self.swarmSize = self.agentCount
        return self


class AdvancedVariables(BaseModel):
    targetAudience: Optional[str] = None
    pricePoint: Optional[str] = None
    marketingBudget: Optional[str] = None


class DebateStartRequest(BaseModel):
    # debate_id is required so the engine can webhook results back onto the existing DB row.
    debate_id: str = Field(..., min_length=1, alias="debateId")
    query: str = Field(..., min_length=1)
    agentCount: int = Field(default=5, ge=1, le=10_000)
    modelTier: str = Field(default="lite")
    advancedVariables: AdvancedVariables = Field(default_factory=AdvancedVariables)

    model_config = {"populate_by_name": True}

