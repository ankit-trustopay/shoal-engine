"""Swarm outcome metrics derived from the Manager consensus."""

from __future__ import annotations

import re

TOTAL_AGENTS = 50

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


def score_manager_sentiment(manager_text: str) -> float:
    """
    Rough sentiment in [-1.0, 1.0] from keyword hits on the Manager verdict.
    """
    text = manager_text.lower()
    if not text.strip():
        return 0.0

    pos = sum(1 for term in _POSITIVE_TERMS if term in text)
    neg = sum(1 for term in _NEGATIVE_TERMS if term in text)

    if pos == 0 and neg == 0:
        # Mild boost when the manager sounds decisive without keyword hits
        if re.search(r"\b(should|will|best option|consensus)\b", text):
            return 0.25
        return 0.0

    raw = (pos - neg) / (pos + neg)
    return max(-1.0, min(1.0, raw))


def compute_confidence(manager_text: str) -> int:
    """Realistic confidence score between 75 and 95."""
    sentiment = score_manager_sentiment(manager_text)
    score = 75 + int(round((sentiment + 1.0) * 10))
    return max(75, min(95, score))


def compute_vote_distribution(
    manager_text: str,
    total: int = TOTAL_AGENTS,
) -> tuple[int, int, int]:
    """
    Simulate a vote split across total agents, aligned with Manager sentiment.
    Default bullish split: 28 For, 10 Against, 12 Neutral.
    """
    sentiment = score_manager_sentiment(manager_text)

    if sentiment > 0.2:
        votes_for, votes_against, votes_neutral = 28, 10, 12
    elif sentiment < -0.2:
        votes_for, votes_against, votes_neutral = 12, 26, 12
    else:
        votes_for, votes_against, votes_neutral = 20, 14, 16

    current = votes_for + votes_against + votes_neutral
    if current != total:
        scale = total / current
        votes_for = int(round(votes_for * scale))
        votes_against = int(round(votes_against * scale))
        votes_neutral = total - votes_for - votes_against

    return votes_for, votes_against, votes_neutral


def compute_mock_cost(runtime_sec: int, message_count: int) -> float:
    """Mock API cost — typically ~0.04 for a standard swarm run."""
    base = 0.028 + message_count * 0.002
    runtime_component = runtime_sec * 0.0005
    return round(max(0.04, base + runtime_component), 2)
