"""Notify the Shoal web app when a swarm ignite fails fatally."""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 15


def _resolve_webhook_url() -> str | None:
    direct = os.getenv("SHOAL_WEBHOOK_URL", "").strip()
    if direct:
        return direct.rstrip("/")

    base = os.getenv("SHOAL_WEB_APP_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/api/webhooks/engine"

    return None


def notify_swarm_failure(swarm_id: str, error: str) -> bool:
    """
    POST failure payload to the Next.js engine webhook so the DB marks FAILED.
    Returns True when the webhook accepted the payload.
    """
    url = _resolve_webhook_url()
    if not url:
        logger.error(
            "SHOAL_WEBHOOK_URL / SHOAL_WEB_APP_URL not configured; "
            "cannot notify failure for swarm %s",
            swarm_id,
        )
        return False

    headers: dict[str, str] = {"Content-Type": "application/json"}
    secret = os.getenv("ENGINE_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["x-engine-webhook-secret"] = secret

    payload = {
        "swarmId": swarm_id,
        "status": "FAILED",
        "error": (error or "Unknown engine error")[:2000],
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.exception(
            "Failed to POST swarm failure webhook for %s: %s",
            swarm_id,
            exc,
        )
        return False

    if response.status_code >= 400:
        logger.error(
            "Swarm failure webhook rejected for %s: status=%s body=%s",
            swarm_id,
            response.status_code,
            response.text[:500],
        )
        return False

    logger.info("Swarm failure webhook delivered for %s", swarm_id)
    return True
