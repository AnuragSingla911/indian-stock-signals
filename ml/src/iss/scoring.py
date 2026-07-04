"""Cross-sectional scoring: winsorize -> z-score -> factor groups -> composite."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mapping of factor group -> raw factor columns (higher raw value = better after sign fixes).
FACTOR_GROUPS: dict[str, list[str]] = {
    "momentum": ["mom_12_1", "ret_3m", "ret_6m"],
    "trend": ["dist_50dma", "dist_200dma", "golden_cross", "vol_trend"],
    "quality": ["roe", "margin", "earnings_growth"],
    "value": ["inv_pe", "inv_pb"],
    "lowvol": ["lowvol"],
}

# Human-readable labels for rationale generation.
FACTOR_LABELS: dict[str, str] = {
    "mom_12_1": "12-1 month momentum",
    "ret_3m": "3-month return",
    "ret_6m": "6-month return",
    "dist_50dma": "price vs 50-DMA",
    "dist_200dma": "price vs 200-DMA",
    "golden_cross": "50/200-DMA trend",
    "vol_trend": "volume trend",
    "roe": "return on equity",
    "margin": "profit margin",
    "earnings_growth": "earnings growth",
    "inv_pe": "valuation (P/E)",
    "inv_pb": "valuation (P/B)",
    "lowvol": "low volatility",
}


def winsorize_zscore(frame: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize each column to [lower, upper] quantiles then standardize to z-scores.

    Missing values are imputed to 0 (the cross-sectional mean) after standardization.
    """
    z = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        s = frame[col].astype(float)
        if s.notna().sum() < 2:
            z[col] = 0.0
            continue
        lo, hi = s.quantile(lower), s.quantile(upper)
        s = s.clip(lo, hi)
        mean, std = s.mean(), s.std(ddof=0)
        z[col] = 0.0 if std == 0 or np.isnan(std) else (s - mean) / std
        z[col] = z[col].fillna(0.0)
    return z


def group_scores(zframe: pd.DataFrame) -> pd.DataFrame:
    """Average the z-scores within each factor group."""
    out = pd.DataFrame(index=zframe.index)
    for group, cols in FACTOR_GROUPS.items():
        present = [c for c in cols if c in zframe.columns]
        out[group] = zframe[present].mean(axis=1) if present else 0.0
    return out


def factor_composite(groups: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = pd.Series(0.0, index=groups.index)
    for group, w in weights.items():
        if group in groups.columns:
            total = total + w * groups[group]
    return total


def to_0_100(series: pd.Series) -> pd.Series:
    """Rank-percentile transform to a 0-100 interpretable score."""
    if len(series) <= 1:
        return pd.Series(50.0, index=series.index)
    ranks = series.rank(method="average", pct=True)
    return (ranks * 100).round(1)
