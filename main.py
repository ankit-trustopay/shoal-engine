import logging

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from models import DebateRequest
from services.ignite_background import run_crew_and_webhook

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Shoal AI Engine", version="0.1.0")

logger = logging.getLogger(__name__)

class IgniteAcceptedResponse(BaseModel):
    status: str
    swarmId: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ignite", response_model=IgniteAcceptedResponse)
def ignite(
    payload: DebateRequest,
    background_tasks: BackgroundTasks,
) -> IgniteAcceptedResponse:
    """
    Accept ignite job and return immediately. CrewAI runs in a background task
    and POSTs results to WEBHOOK_URL (or SHOAL_WEBHOOK_URL) when finished.
    """
    resolved_model = payload.model or payload.model_tier

    logger.info(
        "Ignite accepted: swarm=%s tier=%s audience=%r price=%r budget=%r agents=%s model=%r",
        payload.swarmId,
        payload.model_tier,
        payload.target_audience,
        payload.price_point,
        payload.marketing_budget,
        payload.agentCount,
        resolved_model,
    )

    background_tasks.add_task(
        run_crew_and_webhook,
        payload.swarmId,
        payload.premise,
        payload.agentCount,
        resolved_model,
        payload.swarmSize,
        payload.model_tier,
        payload.target_audience,
        payload.price_point,
        payload.marketing_budget,
    )

    logger.info("Queued background CrewAI for swarm %s", payload.swarmId)

    return IgniteAcceptedResponse(
        status="deliberating",
        swarmId=payload.swarmId,
    )
