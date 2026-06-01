"""Shared swarm orchestration types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.llm import ManagerSynthesis
from services.scraper import EvidenceItem

SECONDS_PER_DEBATE_TURN = 12


def format_debate_timestamp(turn_index: int) -> str:
    """Synthetic war-room clock for transcript entries (00:00, 00:12, …)."""
    total_seconds = turn_index * SECONDS_PER_DEBATE_TURN
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class DebateTranscriptEntry:
    agentName: str
    role: str
    text: str
    timestamp: str


@dataclass
class SwarmIgniteResult:
    messages: list[dict[str, str]]
    debate_transcript: list[DebateTranscriptEntry]
    evidence: list[EvidenceItem]
    agent_profiles: list[dict[str, Any]]
    manager_synthesis: ManagerSynthesis
    executed_agent_count: int
