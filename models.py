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

