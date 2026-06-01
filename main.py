import logging

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import Self

from services.dynamic_personas import MAX_DEBATE_AGENTS
from services.ignite_background import run_crew_and_webhook

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Shoal AI Engine", version="0.1.0")

logger = logging.getLogger(__name__)


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


class IgniteAcceptedResponse(BaseModel):
    status: str
    swarmId: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteAcceptedResponse)
def ignite(
    payload: IgniteRequest,
    background_tasks: BackgroundTasks,
) -> IgniteAcceptedResponse:
    """
    Accept ignite job and return immediately. CrewAI runs in a background task
    and POSTs results to WEBHOOK_URL (or SHOAL_WEBHOOK_URL) when finished.
    """
    background_tasks.add_task(
        run_crew_and_webhook,
        payload.swarmId,
        payload.premise,
        payload.agentCount,
        payload.model,
        payload.swarmSize,
    )

    logger.info("Queued background CrewAI for swarm %s", payload.swarmId)

    return IgniteAcceptedResponse(
        status="deliberating",
        swarmId=payload.swarmId,
    )
