from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Self

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

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


def _non_empty_or(value: str, fallback: str) -> str:
    trimmed = (value or "").strip()
    return trimmed if trimmed else fallback


class ExecutiveSummary(BaseModel):
    """Top hero: BUY | WAIT | PIVOT + confidence + fit + reason."""

    recommendation: Literal["BUY", "WAIT", "PIVOT"] = "WAIT"
    confidence: int = Field(default=50, ge=0, le=100)
    fit_for_you: Literal["Excellent", "Good", "Weak"] = Field(
        default="Good",
        validation_alias=AliasChoices("fit_for_you", "fitForYou"),
        serialization_alias="fit_for_you",
    )
    one_line_reason: str = Field(
        default="Synthesis completed from swarm deliberation.",
        validation_alias=AliasChoices("one_line_reason", "oneLineReason"),
        serialization_alias="one_line_reason",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, value: Any) -> str:
        raw = str(value or "WAIT").strip().upper()
        return raw if raw in ("BUY", "WAIT", "PIVOT") else "WAIT"

    @field_validator("fit_for_you", mode="before")
    @classmethod
    def normalize_fit(cls, value: Any) -> str:
        raw = str(value or "Good").strip()
        return raw if raw in ("Excellent", "Good", "Weak") else "Good"

    @field_validator("one_line_reason", mode="before")
    @classmethod
    def trim_reason(cls, value: Any) -> str:
        return _non_empty_or(str(value or ""), "Synthesis completed from swarm deliberation.")[:500]


class BoardroomSummary(BaseModel):
    """Boardroom findings — strict 5 fields; legacy bull/bear synthesized when omitted."""

    main_opportunity: str = Field(
        default="Primary upside identified by the swarm.",
        validation_alias=AliasChoices("main_opportunity", "mainOpportunity"),
        serialization_alias="main_opportunity",
    )
    main_risk: str = Field(
        default="Primary risk identified by the swarm.",
        validation_alias=AliasChoices("main_risk", "mainRisk"),
        serialization_alias="main_risk",
    )
    hidden_tradeoff: str = Field(
        default="Speed versus certainty on unresolved unknowns.",
        validation_alias=AliasChoices("hidden_tradeoff", "hiddenTradeoff"),
        serialization_alias="hidden_tradeoff",
    )
    best_alternative: str = Field(
        default="Narrow scope and re-run deliberation with tighter constraints.",
        validation_alias=AliasChoices("best_alternative", "bestAlternative"),
        serialization_alias="best_alternative",
    )
    explanation: str = Field(
        default="The swarm weighed live research and adversarial worker arguments.",
    )
    bull_case: str = Field(
        default="",
        validation_alias=AliasChoices("bull_case", "bullCase"),
        serialization_alias="bull_case",
    )
    bear_case: str = Field(
        default="",
        validation_alias=AliasChoices("bear_case", "bearCase"),
        serialization_alias="bear_case",
    )
    shoal_recommendation: str = Field(
        default="",
        validation_alias=AliasChoices("shoal_recommendation", "shoalRecommendation"),
        serialization_alias="shoal_recommendation",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @model_validator(mode="after")
    def fill_legacy_tension_fields(self) -> Self:
        updates: dict[str, str] = {}
        if not self.bull_case.strip():
            updates["bull_case"] = self.main_opportunity[:1200]
        if not self.bear_case.strip():
            updates["bear_case"] = self.main_risk[:1200]
        if not self.shoal_recommendation.strip():
            updates["shoal_recommendation"] = self.explanation[:1200]
        if updates:
            return self.model_copy(update=updates)
        return self

    @field_validator(
        "main_opportunity",
        "main_risk",
        "hidden_tradeoff",
        "best_alternative",
        "explanation",
        mode="before",
    )
    @classmethod
    def trim_core_fields(cls, value: Any) -> str:
        return str(value or "").strip()[:1200]


class DebateRoomAgent(BaseModel):
    role: str = Field(default="Analyst")
    conclusion: str = Field(default="No conclusion recorded.")
    disagreement: str = Field(default="No recorded disagreement.")
    mind_changed: str = Field(
        default="Held position after reviewing live research.",
        validation_alias=AliasChoices("mind_changed", "mindChanged"),
        serialization_alias="mind_changed",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("role", "conclusion", "disagreement", "mind_changed", mode="before")
    @classmethod
    def trim_fields(cls, value: Any) -> str:
        return _non_empty_or(str(value or ""), "—")[:800]


class EvidenceVaultCitation(BaseModel):
    title: str = Field(default="Untitled")
    url: str = Field(default="https://example.com")
    source: str = Field(default="Web")
    snippet: str = Field(default="")

    model_config = {"extra": "ignore"}

    @field_validator("title", "url", "source", mode="before")
    @classmethod
    def trim_citation(cls, value: Any) -> str:
        return str(value or "").strip()[:2000]


class EvidenceVaultClusters(BaseModel):
    reddit: list[EvidenceVaultCitation] = Field(default_factory=list)
    news: list[EvidenceVaultCitation] = Field(default_factory=list)
    official: list[EvidenceVaultCitation] = Field(default_factory=list)
    youtube: list[EvidenceVaultCitation] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class EvidenceVaultStats(BaseModel):
    total: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("total", "total_sources", "totalSources"),
        serialization_alias="total",
    )
    high_signal: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("high_signal", "highSignal"),
        serialization_alias="high_signal",
    )
    contradictory: int = Field(default=0, ge=0)
    dominant_consensus: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("dominant_consensus", "dominantConsensus"),
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}


class EvidenceVault(BaseModel):
    stats: EvidenceVaultStats = Field(default_factory=EvidenceVaultStats)
    clusters: EvidenceVaultClusters = Field(default_factory=EvidenceVaultClusters)

    model_config = {"extra": "ignore"}


class SevenZoneReport(BaseModel):
    """Lenient CEO output — only the four UI zones + verdict/confidence."""

    verdict: str = Field(default="Deliberation complete.")
    confidence: int = Field(default=50, ge=0, le=100)
    executive_summary: ExecutiveSummary = Field(default_factory=ExecutiveSummary)
    boardroom_summary: BoardroomSummary = Field(default_factory=BoardroomSummary)
    debate_room: list[DebateRoomAgent] = Field(default_factory=list)
    evidence_vault: EvidenceVault = Field(default_factory=EvidenceVault)

    model_config = {"extra": "ignore"}


class DebateCompletionPayload(BaseModel):
    """Canonical debate webhook body from the Python engine (7-zone boardroom)."""

    debate_id: str = Field(..., min_length=1)
    status: Literal["completed", "failed", "failure"] = "completed"
    verdict: str = Field(..., min_length=1)
    confidence: int = Field(..., ge=0, le=100)
    agents: list[DebateAgentPosition] = Field(default_factory=list)
    tldr: list[str] = Field(..., min_length=3, max_length=5)
    friction_matrix: list[FrictionMatrixEntry] = Field(..., min_length=1, max_length=50)
    pre_mortem: PreMortem
    execution_roadmap: ExecutionRoadmap
    executive_summary: ExecutiveSummary
    boardroom_summary: BoardroomSummary
    debate_room: list[DebateRoomAgent] = Field(..., min_length=1)
    evidence_vault: EvidenceVault
    runtime: int = Field(default=1, ge=1)
    cost: float = Field(default=0, ge=0)
    agent_count: int = Field(default=3, ge=1, alias="agentCount")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def sync_executive_confidence(self) -> Self:
        if self.executive_summary.confidence != self.confidence:
            self.executive_summary = self.executive_summary.model_copy(
                update={"confidence": self.confidence},
            )
        return self

    @field_validator("tldr")
    @classmethod
    def validate_tldr(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items if item and item.strip()]
        if len(cleaned) < 3:
            raise ValueError("tldr must contain at least 3 non-empty bullets")
        return [item[:400] for item in cleaned[:5]]
