"""Sector scoring and selection: index momentum + constituent breadth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .features import _pct_change
from .universe import Sector, Stock


@dataclass
class SectorScore:
    key: str
    display_name: str
    score: float  # 0-100
    momentum_3m: float | None
    breadth: float | None


def _zscore(values: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    if len(arr) < 2 or np.nanstd(arr) == 0:
        return {k: 0.0 for k in values}
    mean, std = np.nanmean(arr), np.nanstd(arr)
    return {k: float((v - mean) / std) for k, v in values.items()}


def score_sectors(
    sectors_meta: dict[str, Sector],
    by_sector: dict[str, list[Stock]],
    feature_frame: pd.DataFrame,
    index_prices: dict[str, pd.DataFrame],
) -> list[SectorScore]:
    momentum: dict[str, float] = {}
    breadth: dict[str, float] = {}

    for key, stocks in by_sector.items():
        meta = sectors_meta.get(key)
        symbols = [s.symbol for s in stocks if s.symbol in feature_frame.index]
        if not symbols:
            continue

        idx_ticker = meta.index_ticker if meta else None
        if idx_ticker and idx_ticker in index_prices and not index_prices[idx_ticker].empty:
            momentum[key] = _pct_change(index_prices[idx_ticker]["Close"], 63)
        else:
            momentum[key] = float(feature_frame.loc[symbols, "ret_3m"].median())

        above = (feature_frame.loc[symbols, "dist_200dma"] > 0).mean()
        breadth[key] = float(above)

    z_mom = _zscore(momentum)
    z_brd = _zscore(breadth)

    raw = {
        k: CONFIG.sector_momentum_w * z_mom.get(k, 0.0)
        + CONFIG.sector_breadth_w * z_brd.get(k, 0.0)
        for k in momentum
    }
    # Rank-percentile to 0-100 for display.
    s = pd.Series(raw)
    pct = (s.rank(pct=True) * 100).round(1) if len(s) > 1 else pd.Series(50.0, index=s.index)

    scores = [
        SectorScore(
            key=k,
            display_name=sectors_meta[k].display_name if k in sectors_meta else k,
            score=float(pct[k]),
            momentum_3m=momentum.get(k),
            breadth=breadth.get(k),
        )
        for k in raw
    ]
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores
