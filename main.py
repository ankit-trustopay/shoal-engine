import logging

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from models import DebateRequest, IgniteRequest
from services.debate_crew import fallback_debate_result
from services.ignite_background import run_crew_and_webhook, run_simple_debate_and_webhook
from services.metrics import compute_swarm_credits
from services.webhook_notify import notify_debate_completion

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Shoal AI Engine", version="0.2.0")

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
    """Accept ignite job; CrewAI runs in background and POSTs results via webhook."""
    resolved_model = payload.model or payload.model_tier

    logger.info(
        "Ignite accepted: swarm=%s agents=%s model=%r",
        payload.swarmId,
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
    """
    Background worker with a hard guarantee: webhook always receives debate JSON.
    """
    try:
        run_simple_debate_and_webhook(
            debate_id,
            query,
            agent_count=agent_count,
            model_mix=model_mix,
        )
    except Exception as exc:
        logger.exception(
            "Debate background task failed for %s; sending fallback webhook",
            debate_id,
        )
        fallback = fallback_debate_result(str(exc))
        cost = float(compute_swarm_credits(max(3, agent_count)))
        notify_debate_completion(
            debate_id,
            verdict=fallback["verdict"],
            confidence=int(fallback["confidence"]),
            agents=list(fallback["agents"]),
            runtime=1,
            cost=cost,
            agent_count=max(3, agent_count),
        )


@app.post("/debate", response_model=DebateAcceptedResponse)
def debate(
    payload: DebateRequest,
    background_tasks: BackgroundTasks,
) -> DebateAcceptedResponse:
    """
    Accept a debate job from shoal-web and return immediately.

    Body: { debate_id, query, agent_count, model_mix }
    """
    logger.info(
        "Debate accepted: debate_id=%s agents=%s model_mix=%s",
        payload.debate_id,
        payload.agent_count,
        payload.model_mix,
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
