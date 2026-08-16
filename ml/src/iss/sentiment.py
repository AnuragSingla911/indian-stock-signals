"""Lexicon-based headline sentiment scoring."""

from __future__ import annotations

import re

_POSITIVE = frozenset(
    {
        "beat",
        "beats",
        "growth",
        "profit",
        "profits",
        "gain",
        "gains",
        "rise",
        "rises",
        "rising",
        "surge",
        "surges",
        "record",
        "strong",
        "upgrade",
        "upgraded",
        "bullish",
        "outperform",
        "expansion",
        "dividend",
        "buyback",
        "deal",
        "win",
        "wins",
        "approval",
        "approved",
        "launch",
        "partnership",
        "recovery",
        "rebound",
    }
)
_NEGATIVE = frozenset(
    {
        "miss",
        "misses",
        "loss",
        "losses",
        "fall",
        "falls",
        "falling",
        "drop",
        "drops",
        "decline",
        "declines",
        "weak",
        "downgrade",
        "downgraded",
        "bearish",
        "underperform",
        "lawsuit",
        "fraud",
        "probe",
        "investigation",
        "cut",
        "cuts",
        "slump",
        "concern",
        "concerns",
        "warning",
        "delay",
        "delays",
        "default",
        "bankruptcy",
        "strike",
    }
)

_WORD_RE = re.compile(r"[a-z]+")


def score_sentiment(text: str) -> float:
    """Lexicon sentiment in [-1, 1]. Neutral text returns 0."""
    if not text:
        return 0.0
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)
