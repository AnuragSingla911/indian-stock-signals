"""Factor computation: technical (price/volume) and fundamental factors.

Only trailing windows are used for technical features (no look-ahead). Fundamentals are
current-snapshot best-effort values and feed the quality/value factor groups only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Technical features used by both the composite and the ML model.
TECH_FEATURES = [
    "mom_12_1",
    "ret_3m",
    "ret_6m",
    "dist_50dma",
    "dist_200dma",
    "golden_cross",
    "rsi14",
    "vol_trend",
    "lowvol",
]


def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < period + 1:
        return 50.0
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    if last_loss == 0 or np.isnan(last_loss):
        return 100.0 if last_gain and last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return float(100 - 100 / (1 + rs))


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    old = series.iloc[-1 - periods]
    if old == 0 or np.isnan(old):
        return 0.0
    return float(series.iloc[-1] / old - 1.0)


def technical_features(close: pd.Series, volume: pd.Series | None = None) -> dict[str, float]:
    """Compute technical factors from a price series ending at the evaluation point."""
    close = close.dropna()
    out: dict[str, float] = {}

    # Momentum: 12-1 (return from ~252d ago to ~21d ago), 3M, 6M.
    if len(close) > 252:
        p_252 = close.iloc[-252]
        p_21 = close.iloc[-21]
        out["mom_12_1"] = float(p_21 / p_252 - 1.0) if p_252 else 0.0
    else:
        out["mom_12_1"] = _pct_change(close, max(len(close) - 1, 1))
    out["ret_3m"] = _pct_change(close, 63)
    out["ret_6m"] = _pct_change(close, 126)

    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
    last = close.iloc[-1]
    out["dist_50dma"] = float(last / sma50 - 1.0) if sma50 else 0.0
    out["dist_200dma"] = float(last / sma200 - 1.0) if sma200 else 0.0
    out["golden_cross"] = 1.0 if sma50 >= sma200 else 0.0

    out["rsi14"] = rsi(close)

    daily = close.pct_change().dropna()
    window = daily.iloc[-63:] if len(daily) >= 63 else daily
    vol = float(window.std() * np.sqrt(252)) if len(window) > 1 else 0.0
    out["lowvol"] = -vol  # lower volatility scores higher

    if volume is not None and len(volume.dropna()) >= 60:
        v = volume.dropna()
        sma20 = v.rolling(20).mean().iloc[-1]
        sma60 = v.rolling(60).mean().iloc[-1]
        out["vol_trend"] = float(sma20 / sma60 - 1.0) if sma60 else 0.0
    else:
        out["vol_trend"] = 0.0

    return out


def technical_panel_dated(
    close: pd.Series, horizon: int, step: int = 21, min_history: int = 260
) -> list[tuple[dict[str, float], int, pd.Timestamp, float]]:
    """Build (features, label, as_of_date, forward_return) samples at historical points.

    Label = 1 if forward `horizon`-day return > 0. No look-ahead: features use only data up
    to t; label/return use strictly future prices. The as-of date enables walk-forward
    (time-ordered) evaluation.
    """
    close = close.dropna()
    samples: list[tuple[dict[str, float], int, pd.Timestamp, float]] = []
    n = len(close)
    if n < min_history + horizon + 1:
        return samples
    for t in range(min_history, n - horizon, step):
        window = close.iloc[: t + 1]
        feats = technical_features(window)
        fwd = float(close.iloc[t + horizon] / close.iloc[t] - 1.0)
        samples.append((feats, int(fwd > 0), pd.Timestamp(close.index[t]), fwd))
    return samples


def technical_panel(
    close: pd.Series, horizon: int, step: int = 21, min_history: int = 260
) -> list[tuple[dict[str, float], int]]:
    """Build (features, label) samples at historical points for ML training.

    Label = 1 if forward `horizon`-day return > 0. No look-ahead: features use only data up
    to t; label uses strictly future prices.
    """
    return [(f, y) for f, y, _, _ in technical_panel_dated(close, horizon, step, min_history)]


def build_feature_frame(
    prices: dict[str, pd.DataFrame], fundamentals: dict[str, dict]
) -> pd.DataFrame:
    """Assemble a per-symbol raw factor frame (technical + fundamental)."""
    rows: dict[str, dict[str, float]] = {}
    for sym, df in prices.items():
        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else None
        feats = technical_features(close, volume)

        f = fundamentals.get(sym, {})
        pe = f.get("pe")
        pb = f.get("pb")
        feats["inv_pe"] = float(1.0 / pe) if pe and pe > 0 else np.nan
        feats["inv_pb"] = float(1.0 / pb) if pb and pb > 0 else np.nan
        feats["roe"] = float(f["roe"]) if f.get("roe") is not None else np.nan
        feats["margin"] = float(f["margin"]) if f.get("margin") is not None else np.nan
        feats["earnings_growth"] = (
            float(f["earnings_growth"]) if f.get("earnings_growth") is not None else np.nan
        )
        rows[sym] = feats
    return pd.DataFrame.from_dict(rows, orient="index")
