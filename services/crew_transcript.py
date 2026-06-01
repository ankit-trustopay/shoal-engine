"""Extract debateTranscript rows from CrewAI task outputs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from services.dynamic_personas import DynamicPersona
from services.swarm_types import DebateTranscriptEntry, format_debate_timestamp

logger = logging.getLogger(__name__)


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_turn_payload(raw: str) -> dict[str, Any] | None:
    cleaned = _strip_json_fence(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return payload if isinstance(payload, dict) else None


def _task_output_text(output: Any) -> str:
    if output is None:
        return ""

    for attr in ("raw", "output", "result"):
        value = getattr(output, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(output, str):
        return output.strip()

    return str(output).strip()


def extract_debate_turn_text(raw: str) -> str:
    payload = _parse_turn_payload(raw)
    if payload:
        for key in ("debateTurn", "debate_turn", "text", "argument", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return raw.strip()


def build_transcript_from_task_outputs(
    personas: list[DynamicPersona],
    task_outputs: list[Any],
    *,
    synthesis_task_count: int = 1,
) -> list[DebateTranscriptEntry]:
    """
    Map worker task outputs (all but the final CEO synthesis) to debateTranscript entries.
    """
    worker_outputs = task_outputs[: max(0, len(task_outputs) - synthesis_task_count)]
    entries: list[DebateTranscriptEntry] = []

    for index, output in enumerate(worker_outputs):
        persona = personas[index] if index < len(personas) else personas[-1]
        raw = _task_output_text(output)
        if not raw:
            continue

        text = extract_debate_turn_text(raw)
        if not text:
            continue

        entries.append(
            DebateTranscriptEntry(
                agentName=str(persona.get("name") or persona.get("role") or "Agent"),
                role=str(persona.get("role") or "Analyst"),
                text=text,
                timestamp=format_debate_timestamp(index),
            ),
        )

    if entries:
        return entries

    logger.warning("No task outputs parsed; building transcript from crew log strings")
    for index, output in enumerate(worker_outputs):
        persona = personas[index] if index < len(personas) else personas[-1]
        raw = _task_output_text(output)
        if not raw:
            continue
        entries.append(
            DebateTranscriptEntry(
                agentName=str(persona.get("name") or "Agent"),
                role=str(persona.get("role") or "Analyst"),
                text=raw[:4000],
                timestamp=format_debate_timestamp(index),
            ),
        )

    return entries
