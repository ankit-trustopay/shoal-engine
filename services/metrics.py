"""Swarm outcome metrics: confidence, vote extrapolation, and credit costing."""

from __future__ import annotations

import re

DEFAULT_SWARM_SIZE = 1000

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


def score_manager_sentiment(manager_text: str) -> float:
    return score_text_sentiment(manager_text)


def aggregate_leader_sentiment(leader_texts: list[str]) -> float:
    if not leader_texts:
        return 0.0
    scores = [score_text_sentiment(text) for text in leader_texts if text.strip()]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_confidence(manager_text: str) -> int:
    """Realistic confidence score between 75 and 95."""
    sentiment = score_manager_sentiment(manager_text)
    score = 75 + int(round((sentiment + 1.0) * 10))
    return max(75, min(95, score))


def _split_remainder(remainder: int, blend: float) -> tuple[int, int]:
    """Allocate leftover votes between Against and Neutral from leader sentiment."""
    if remainder <= 0:
        return 0, 0

    # Higher blend (bullish) -> fewer Against, more Neutral
    against_share = 0.5 - (blend * 0.38)
    against_share = max(0.12, min(0.88, against_share))

    votes_against = int(round(remainder * against_share))
    votes_neutral = remainder - votes_against
    return votes_against, votes_neutral


def compute_extrapolated_votes(
    confidence: int,
    manager_text: str,
    leader_texts: list[str],
    swarm_size: int = DEFAULT_SWARM_SIZE,
) -> tuple[int, int, int]:
    """
    Extrapolate votes across swarm_size agents.
    votesFor is anchored to confidence %; remainder splits by archetype sentiment.
    """
    total = max(1, int(swarm_size))
    conf = max(0, min(100, confidence))

    votes_for = int(round(total * conf / 100))
    votes_for = max(0, min(total, votes_for))
    remainder = total - votes_for

    manager_sent = score_manager_sentiment(manager_text)
    leader_sent = aggregate_leader_sentiment(leader_texts)
    blend = 0.35 * manager_sent + 0.65 * leader_sent

    votes_against, votes_neutral = _split_remainder(remainder, blend)

    # Guarantee exact sum
    current = votes_for + votes_against + votes_neutral
    if current != total:
        votes_neutral = total - votes_for - votes_against

    return votes_for, votes_against, votes_neutral


def compute_swarm_credits(swarm_size: int) -> float:
    """1 virtual human = 1 credit (100 agents = 100.0, 1,000 agents = 1,000.0)."""
    size = max(1, int(swarm_size))
    return float(size)


# Backward-compatible alias
def compute_vote_distribution(
    manager_text: str,
    total: int = DEFAULT_SWARM_SIZE,
) -> tuple[int, int, int]:
    confidence = compute_confidence(manager_text)
    return compute_extrapolated_votes(confidence, manager_text, [], total)
