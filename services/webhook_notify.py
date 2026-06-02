"""Notify the Shoal web app when a swarm ignite completes or fails."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 15

# Keys read by shoal-web/lib/parse-engine-webhook.ts (flat body or nested reportData).
_IGNITE_FIELD_KEYS = (
    "messages",
    "confidence",
    "votesFor",
    "votesAgainst",
    "votesNeutral",
    "runtime",
    "cost",
    "evidence",
    "agentProfiles",
    "debateTranscript",
    "recommendedActions",
    "minorityDissent",
    "model",
    "swarmSize",
    "agentCount",
    "response",
)


def _resolve_webhook_url() -> str | None:
    for env_key in ("WEBHOOK_URL", "SHOAL_WEBHOOK_URL"):
        direct = os.getenv(env_key, "").strip()
        if direct:
            return direct.rstrip("/")

    base = os.getenv("SHOAL_WEB_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/api/webhooks/engine"

    base = os.getenv("SHOAL_WEB_APP_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/api/webhooks/engine"

    return None


def _webhook_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    secret = os.getenv("ENGINE_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["x-engine-webhook-secret"] = secret
    else:
        print(
            "[webhook] WARNING: ENGINE_WEBHOOK_SECRET is not set; "
            "shoal-web will reject the request in production",
        )
    return headers


def _json_safe(value: Any) -> Any:
    """Coerce payload values to JSON-serializable primitives for requests.post(json=...)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def notify_debate_completion(
    debate_id: str,
    *,
    verdict: str,
    confidence: int,
    agents: list[dict[str, str]],
    runtime: int,
    cost: float,
    agent_count: int,
) -> bool:
    """
    POST canonical debate completion JSON to shoal-web /api/webhooks/engine.
    """
    url = _resolve_webhook_url()
    if not url:
        logger.error("Webhook URL not configured; cannot deliver debate %s", debate_id)
        return False

    safe_verdict = (verdict or "").strip()
    if not safe_verdict:
        safe_verdict = (
            "Deliberation completed without a parsed verdict string. "
            "Check engine logs for raw agent output."
        )
    safe_confidence = int(max(0, min(100, confidence)))
    safe_agents = [
        {
            "name": str(agent.get("name") or f"Agent {index + 1}"),
            "position": str(agent.get("position") or "No position recorded."),
        }
        for index, agent in enumerate(agents)
        if isinstance(agent, dict)
    ]

    body: dict[str, Any] = {
        "debate_id": debate_id,
        "status": "completed",
        "verdict": safe_verdict,
        "confidence": safe_confidence,
        "agents": safe_agents,
        "runtime": int(max(1, runtime)),
        "cost": float(cost),
        "agentCount": int(agent_count),
    }
    return _post_webhook(url, _json_safe(body), debate_id, "debate-complete")


def format_success_webhook_body(
    swarm_id: str,
    ignite_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Build POST body for shoal-web /api/webhooks/engine success callbacks.

    Next.js accepts either:
      - flat ignite fields + swarmId, or
      - { swarmId, reportData: { messages, confidence, ... } }
    """
    report_data: dict[str, Any] = {}
    for key in _IGNITE_FIELD_KEYS:
        if key in ignite_fields and ignite_fields[key] is not None:
            report_data[key] = _json_safe(ignite_fields[key])

    # Guarantees looksLikeIgnitePayload() in parse-engine-webhook.ts
    if "confidence" not in report_data:
        report_data["confidence"] = 0
    if "messages" not in report_data:
        report_data["messages"] = []
    if "evidence" not in report_data:
        report_data["evidence"] = []
    if "agentProfiles" not in report_data:
        report_data["agentProfiles"] = []
    if "debateTranscript" not in report_data:
        report_data["debateTranscript"] = []

    body: dict[str, Any] = {
        "swarmId": swarm_id,
        "reportData": report_data,
    }
    # Flat duplicate so either parser branch succeeds.
    body.update(report_data)

    # Additional compatibility payload (explicitly requested by QA/ops):
    # { debate_id, status, verdict, agents, confidence }
    # Keep swarmId/reportData as the canonical fields for shoal-web persistence.
    body.update(
        {
            "debate_id": swarm_id,
            "status": "completed",
            "verdict": report_data.get("consensus")
            or report_data.get("response")
            or ignite_fields.get("response")
            or "",
            "agents": report_data.get("agentProfiles") or [],
            "confidence": report_data.get("confidence") or 0,
        }
    )
    return _json_safe(body)


def _post_webhook(url: str, payload: dict[str, Any], swarm_id: str, label: str) -> bool:
    headers = _webhook_headers()
    safe_payload = _json_safe(payload)

    print(f"[webhook] POST {url} ({label}) swarmId={swarm_id}")
    print(
        "[webhook] request payload:",
        json.dumps(safe_payload, indent=2, default=str)[:8000],
    )
    print(
        "[webhook] x-engine-webhook-secret present:",
        bool(headers.get("x-engine-webhook-secret")),
    )

    try:
        response = requests.post(
            url,
            json=safe_payload,
            headers=headers,
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        print(f"[webhook] POST failed ({label}) swarmId={swarm_id}: {exc}")
        logger.exception(
            "Failed to POST swarm %s webhook for %s: %s",
            label,
            swarm_id,
            exc,
        )
        return False

    print(
        f"[webhook] response status={response.status_code} "
        f"body={response.text[:2000]}"
    )

    if response.status_code >= 400:
        logger.error(
            "Swarm %s webhook rejected for %s: status=%s body=%s",
            label,
            swarm_id,
            response.status_code,
            response.text[:500],
        )
        return False

    logger.info("Swarm %s webhook delivered for %s", label, swarm_id)
    return True


def notify_swarm_success(swarm_id: str, ignite_fields: dict[str, Any]) -> bool:
    """
    POST completed ignite payload to Next.js /api/webhooks/engine.
    """
    url = _resolve_webhook_url()
    if not url:
        print(
            f"[webhook] ERROR: no WEBHOOK_URL for swarm {swarm_id}; "
            "set WEBHOOK_URL, SHOAL_WEBHOOK_URL, or SHOAL_WEB_APP_URL",
        )
        logger.error(
            "WEBHOOK_URL / SHOAL_WEBHOOK_URL / SHOAL_WEB_APP_URL not configured; "
            "cannot deliver results for swarm %s",
            swarm_id,
        )
        return False

    body = format_success_webhook_body(swarm_id, ignite_fields)
    return _post_webhook(url, body, swarm_id, "success")


def notify_swarm_failure(swarm_id: str, error: str) -> bool:
    """
    POST failure payload to the Next.js engine webhook so the DB marks FAILED.
    """
    url = _resolve_webhook_url()
    if not url:
        print(f"[webhook] ERROR: no WEBHOOK_URL for failure swarm {swarm_id}")
        logger.error(
            "WEBHOOK_URL / SHOAL_WEBHOOK_URL / SHOAL_WEB_APP_URL not configured; "
            "cannot notify failure for swarm %s",
            swarm_id,
        )
        return False

    payload = {
        "swarmId": swarm_id,
        "status": "FAILED",
        "error": (error or "Unknown engine error")[:2000],
    }

    return _post_webhook(url, payload, swarm_id, "failure")
