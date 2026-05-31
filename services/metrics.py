"""Swarm outcome metrics: confidence, vote extrapolation, and credit costing."""

from __future__ import annotations

import re
from typing import Literal

DEFAULT_SWARM_SIZE = 1000

AgentSentiment = Literal["For", "Against", "Neutral"]

_VALID_SENTIMENTS = frozenset({"For", "Against", "Neutral"})

_POSITIVE_TERMS = (
    "recommend",
    "strong",
    "favorable",
    "benefit",
    "excellent",
    "confident",
    "support",
    "worth",
    "positive",
    "advantage",
    "clear winner",
    "proceed",
    "yes",
)

_NEGATIVE_TERMS = (
    "avoid",
    "risk",
    "concern",
    "caution",
    "weak",
    "poor",
    "against",
    "not recommend",
    "unfavorable",
    "negative",
    "flaw",
    "recall",
    "warning",
    "no",
)


def normalize_sentiment(value: str | None) -> AgentSentiment | None:
    if not value:
        return None
    cleaned = value.strip().capitalize()
    if cleaned in _VALID_SENTIMENTS:
        return cleaned  # type: ignore[return-value]
    lowered = value.strip().lower()
    if lowered in ("for", "pro", "support", "supporting", "yes"):
        return "For"
    if lowered in ("against", "con", "oppose", "opposing", "no", "reject"):
        return "Against"
    if lowered in ("neutral", "undecided", "abstain", "mixed"):
        return "Neutral"
    return None


def score_text_sentiment(text: str) -> float:
    """Rough sentiment in [-1.0, 1.0] from keyword hits."""
    lowered = text.lower()
    if not lowered.strip():
        return 0.0

    pos = sum(1 for term in _POSITIVE_TERMS if term in lowered)
    neg = sum(1 for term in _NEGATIVE_TERMS if term in lowered)

    if pos == 0 and neg == 0:
        if re.search(r"\b(should|will|best option|consensus)\b", lowered):
            return 0.25
        return 0.0

    raw = (pos - neg) / (pos + neg)
    return max(-1.0, min(1.0, raw))


def compute_confidence_from_synthesis(
    sentiments: list[AgentSentiment],
    manager_confidence: int | None = None,
    evidence_quality_score: int | None = None,
) -> int:
    """
    Blend manager confidence with sentiment alignment and evidence quality.

    The manager must weigh both how aligned agents are and how well they cited
    verifiable data from the research context.
    """
    if manager_confidence is not None:
        base = max(75, min(95, int(manager_confidence)))
    elif sentiments:
        total = len(sentiments)
        for_count = sum(1 for item in sentiments if item == "For")
        against_count = sum(1 for item in sentiments if item == "Against")
        dominant = max(for_count, against_count, total - for_count - against_count)
        agreement_ratio = dominant / total
        base = max(75, min(95, 75 + int(round(agreement_ratio * 20))))
    else:
        base = 75

    if evidence_quality_score is None:
        return base

    quality = max(0, min(100, int(evidence_quality_score)))
    blended = int(round(0.55 * base + 0.45 * max(75, quality)))

    if quality < 60:
        blended = min(blended, 82)
    if quality < 45:
        blended = min(blended, 78)

    return max(75, min(95, blended))


def compute_confidence_from_sentiments(
    sentiments: list[AgentSentiment],
    manager_confidence: int | None = None,
) -> int:
    """Backward-compatible confidence helper."""
    return compute_confidence_from_synthesis(sentiments, manager_confidence, None)


def compute_extrapolated_votes_from_sentiments(
    sentiments: list[AgentSentiment],
    swarm_size: int = DEFAULT_SWARM_SIZE,
) -> tuple[int, int, int]:
    """
    Extrapolate swarm-scale votes from the five-agent sentiment ratio.

    Example: 4 For, 1 Against, 0 Neutral at swarm_size=1000 -> 800 / 200 / 0.
    """
    total = max(1, int(swarm_size))
    panel_size = max(1, len(sentiments))

    for_count = sum(1 for item in sentiments if item == "For")
    against_count = sum(1 for item in sentiments if item == "Against")

    votes_for = (total * for_count) // panel_size
    votes_against = (total * against_count) // panel_size
    votes_neutral = total - votes_for - votes_against

    return votes_for, votes_against, votes_neutral


def compute_confidence(manager_text: str) -> int:
    """Legacy keyword-based confidence for plain-text manager output."""
    sentiment = score_text_sentiment(manager_text)
    score = 75 + int(round((sentiment + 1.0) * 10))
    return max(75, min(95, score))


def compute_extrapolated_votes(
    confidence: int,
    manager_text: str,
    leader_texts: list[str],
    swarm_size: int = DEFAULT_SWARM_SIZE,
) -> tuple[int, int, int]:
    """Legacy vote extrapolation anchored to confidence % — prefer sentiment-based path."""
    total = max(1, int(swarm_size))
    conf = max(0, min(100, confidence))

    votes_for = int(round(total * conf / 100))
    votes_for = max(0, min(total, votes_for))
    remainder = total - votes_for

    manager_sent = score_text_sentiment(manager_text)
    leader_scores = [score_text_sentiment(text) for text in leader_texts if text.strip()]
    leader_sent = sum(leader_scores) / len(leader_scores) if leader_scores else 0.0
    blend = 0.35 * manager_sent + 0.65 * leader_sent

    against_share = max(0.12, min(0.88, 0.5 - (blend * 0.38)))
    votes_against = int(round(remainder * against_share))
    votes_neutral = remainder - votes_against

    if votes_for + votes_against + votes_neutral != total:
        votes_neutral = total - votes_for - votes_against

    return votes_for, votes_against, votes_neutral


def compute_swarm_credits(swarm_size: int) -> float:
    """1 virtual human = 1 credit (100 agents = 100.0, 1,000 agents = 1,000.0)."""
    size = max(1, int(swarm_size))
    return float(size)


def compute_vote_distribution(
    manager_text: str,
    total: int = DEFAULT_SWARM_SIZE,
) -> tuple[int, int, int]:
    confidence = compute_confidence(manager_text)
    return compute_extrapolated_votes(confidence, manager_text, [], total)
