import logging

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from models import DebateRequest, DebateStartRequest
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


class DebateAcceptedResponse(BaseModel):
    status: str
    debateId: str


@app.post("/debate", response_model=DebateAcceptedResponse)
def debate(
    payload: DebateStartRequest,
    background_tasks: BackgroundTasks,
) -> DebateAcceptedResponse:
    """
    Accept a debate job and return immediately. The full CrewAI pipeline runs in a
    background task and POSTs results to SHOAL_WEB_URL/api/webhooks/engine.

    Expected JSON body:
    {
      "debateId": "...",
      "query": "...",
      "agentCount": 50,
      "modelTier": "lite" | "plus",
      "advancedVariables": {
        "targetAudience": "...",
        "pricePoint": "...",
        "marketingBudget": "..."
      }
    }
    """
    logger.info(
        "Debate accepted: debate_id=%s tier=%s audience=%r price=%r budget=%r agents=%s",
        payload.debate_id,
        payload.modelTier,
        payload.advancedVariables.targetAudience,
        payload.advancedVariables.pricePoint,
        payload.advancedVariables.marketingBudget,
        payload.agentCount,
    )

    # Keep CEO model fixed downstream; use modelTier for worker routing.
    background_tasks.add_task(
        run_crew_and_webhook,
        payload.debate_id,
        payload.query,
        payload.agentCount,
        None,  # CEO model resolved inside crew_orchestration.py (fixed frontier)
        payload.agentCount,
        payload.modelTier,
        payload.advancedVariables.targetAudience,
        payload.advancedVariables.pricePoint,
        payload.advancedVariables.marketingBudget,
    )

    return DebateAcceptedResponse(status="deliberating", debateId=payload.debate_id)
