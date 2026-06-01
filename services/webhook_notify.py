"""Notify the Shoal web app when a swarm ignite completes or fails."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 15


def _resolve_webhook_url() -> str | None:
    for env_key in ("WEBHOOK_URL", "SHOAL_WEBHOOK_URL"):
        direct = os.getenv(env_key, "").strip()
        if direct:
            return direct.rstrip("/")

    base = os.getenv("SHOAL_WEB_APP_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/api/webhooks/engine"

    return None


def _webhook_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    secret = os.getenv("ENGINE_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["x-engine-webhook-secret"] = secret
    return headers


def _post_webhook(url: str, payload: dict[str, Any], swarm_id: str, label: str) -> bool:
    try:
        response = requests.post(
            url,
            json=payload,
            headers=_webhook_headers(),
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.exception(
            "Failed to POST swarm %s webhook for %s: %s",
            label,
            swarm_id,
            exc,
        )
        return False

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


def notify_swarm_success(swarm_id: str, result_payload: dict[str, Any]) -> bool:
    """
    POST completed ignite payload to Next.js /api/webhooks/engine.
    Expects flat ignite fields plus swarmId (see parse-engine-webhook.ts).
    """
    url = _resolve_webhook_url()
    if not url:
        logger.error(
            "WEBHOOK_URL / SHOAL_WEBHOOK_URL / SHOAL_WEB_APP_URL not configured; "
            "cannot deliver results for swarm %s",
            swarm_id,
        )
        return False

    return _post_webhook(url, result_payload, swarm_id, "success")


def notify_swarm_failure(swarm_id: str, error: str) -> bool:
    """
    POST failure payload to the Next.js engine webhook so the DB marks FAILED.
    Returns True when the webhook accepted the payload.
    """
    url = _resolve_webhook_url()
    if not url:
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
