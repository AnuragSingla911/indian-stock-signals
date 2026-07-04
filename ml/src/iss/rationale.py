"""Deterministic, template-driven rationale generation from factor z-scores.

No LLM dependency -> reproducible, free, and auditable.
"""

from __future__ import annotations

import pandas as pd

from .scoring import FACTOR_LABELS

_POSITIVE_PHRASES = [
    (1.5, "very strong"),
    (0.8, "strong"),
    (0.3, "above-average"),
]
_NEGATIVE_PHRASES = [
    (-1.5, "very weak"),
    (-0.8, "weak"),
    (-0.3, "below-average"),
]


def _phrase(z: float) -> str | None:
    for thr, word in _POSITIVE_PHRASES:
        if z >= thr:
            return word
    for thr, word in _NEGATIVE_PHRASES:
        if z <= thr:
            return word
    return None


def stock_rationale(z_row: pd.Series, up_probability: float, composite_0_100: float) -> str:
    """Build a rationale citing the largest positive and any notable negative factors."""
    ranked = z_row.sort_values(ascending=False)
    positives: list[str] = []
    for factor, z in ranked.items():
        if len(positives) >= 3 or z < 0.3:
            break
        label = FACTOR_LABELS.get(str(factor), str(factor))
        word = _phrase(float(z))
        if word:
            positives.append(f"{word} {label}")

    negatives: list[str] = []
    for factor, z in ranked.sort_values().items():
        if len(negatives) >= 1 or z > -0.8:
            break
        label = FACTOR_LABELS.get(str(factor), str(factor))
        negatives.append(f"{_phrase(float(z))} {label}")

    if positives:
        head = "Ranks highly on " + ", ".join(positives)
    else:
        head = "Balanced factor profile with no standout weakness"

    parts = [head]
    parts.append(f"composite score {composite_0_100:.0f}/100")
    parts.append(f"model up-probability {up_probability * 100:.0f}%")
    text = "; ".join(parts) + "."
    if negatives:
        text += " Watch: " + ", ".join(negatives) + "."
    return text


def sector_rationale(display_name: str, momentum_3m: float | None, breadth: float | None) -> str:
    bits: list[str] = []
    if momentum_3m is not None:
        direction = "up" if momentum_3m >= 0 else "down"
        bits.append(f"3M sector momentum {direction} {abs(momentum_3m) * 100:.1f}%")
    if breadth is not None:
        bits.append(f"{breadth * 100:.0f}% of constituents above their 200-DMA")
    if not bits:
        return f"{display_name}: selected on relative factor strength."
    return f"{display_name}: " + "; ".join(bits) + "."
