import logging

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from models import DebateRequest, IgniteRequest
from services.ignite_background import run_crew_and_webhook, run_simple_debate_and_webhook

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


class DebateAcceptedResponse(BaseModel):
    status: str
    debateId: str


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


def _run_debate_crew_and_webhook(
    debate_id: str,
    query: str,
    agent_count: int,
    model_mix: float,
) -> None:
    """Background worker: CrewAI debate + webhook delivery."""
    run_simple_debate_and_webhook(
        debate_id,
        query,
        agent_count=agent_count,
        model_mix=model_mix,
    )


@app.post("/debate", response_model=DebateAcceptedResponse)
def debate(
    payload: DebateRequest,
    background_tasks: BackgroundTasks,
) -> DebateAcceptedResponse:
    """
    Accept a debate job from shoal-web and return immediately.

    Expected JSON body:
    {
      "debate_id": "...",
      "query": "...",
      "agent_count": 50,
      "model_mix": 25
    }
    """
    logger.info(
        "Debate accepted: debate_id=%s agents=%s model_mix=%s query_len=%s",
        payload.debate_id,
        payload.agent_count,
        payload.model_mix,
        len(payload.query),
    )

    background_tasks.add_task(
        _run_debate_crew_and_webhook,
        payload.debate_id,
        payload.query,
        payload.agent_count,
        payload.model_mix,
    )

    return DebateAcceptedResponse(
        status="deliberating",
        debateId=payload.debate_id,
    )
