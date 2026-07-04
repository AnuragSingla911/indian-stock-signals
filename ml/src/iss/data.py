"""Market data access: yfinance with disk cache and an offline/synthetic fallback.

The pipeline must always produce output, even with no network (CI, sandboxes). When live
fetching fails or ISS_OFFLINE=1, we generate deterministic synthetic OHLCV so the whole
system is reproducible and testable offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import CONFIG, PRICE_CACHE_DIR, offline

log = logging.getLogger("iss.data")

_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)


def _synthetic_history(symbol: str, days: int) -> pd.DataFrame:
    """Deterministic geometric-brownian-motion OHLCV keyed by symbol.

    Drift/vol vary per symbol so cross-sectional ranking is non-degenerate.
    """
    rng = np.random.default_rng(_seed(symbol))
    n = days
    # Per-symbol drift in [-0.0004, 0.0011] daily, vol in [0.010, 0.032].
    drift = (rng.random() - 0.35) * 0.0016
    vol = 0.010 + rng.random() * 0.022
    rets = rng.normal(drift, vol, n)
    start_price = 50 + rng.random() * 3500
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    vol_base = 1e5 + rng.random() * 5e6
    volume = (vol_base * (1 + np.abs(rng.normal(0, 0.3, n)))).astype("int64")

    end = datetime.utcnow().date()
    # Business-day index ending today.
    idx = pd.bdate_range(end=end, periods=n)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df


def _cache_path(symbol: str):
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("^", "_idx_").replace("&", "_and_").replace("-", "_")
    return PRICE_CACHE_DIR / f"{safe}.csv"


def _read_cache(symbol: str, max_age_hours: int = 18) -> pd.DataFrame | None:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    age_h = (datetime.utcnow().timestamp() - p.stat().st_mtime) / 3600
    if age_h > max_age_hours:
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:  # noqa: BLE001 - cache is best-effort
        return None


def _write_cache(symbol: str, df: pd.DataFrame) -> None:
    try:
        df.to_csv(_cache_path(symbol))
    except Exception:  # noqa: BLE001
        pass


def _to_yahoo(symbol: str) -> str:
    """NSE equity symbols get a .NS suffix; index tickers (starting with ^) are used as-is."""
    return symbol if symbol.startswith("^") else f"{symbol}.NS"


def _fetch_yf(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    period_start = (datetime.utcnow() - timedelta(days=int(days * 1.6) + 30)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    ymap = {sym: _to_yahoo(sym) for sym in symbols}
    data = yf.download(
        tickers=" ".join(ymap.values()),
        start=period_start,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    for sym in symbols:
        ysym = ymap[sym]
        try:
            if len(symbols) == 1:
                sub = data
            else:
                sub = data[ysym]
            sub = sub.dropna(how="all")
            cols = [c for c in _PRICE_COLS if c in sub.columns]
            if not cols or sub[cols].dropna().empty:
                continue
            out[sym] = sub[cols].dropna()
        except Exception:  # noqa: BLE001
            continue
    return out


def get_price_history(symbols: list[str], days: int | None = None) -> dict[str, pd.DataFrame]:
    """Return {symbol: OHLCV DataFrame}. Uses cache, then live, then synthetic fallback."""
    days = days or CONFIG.lookback_days
    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    if not offline():
        for sym in symbols:
            cached = _read_cache(sym)
            if cached is not None:
                result[sym] = cached
            else:
                missing.append(sym)
        if missing:
            try:
                fetched = _fetch_yf(missing, days)
                for sym, df in fetched.items():
                    _write_cache(sym, df)
                    result[sym] = df
            except Exception as e:  # noqa: BLE001
                log.warning("yfinance fetch failed (%s); using synthetic fallback", e)
    else:
        missing = list(symbols)

    for sym in symbols:
        if sym not in result or result[sym].empty:
            result[sym] = _synthetic_history(sym, days)
    return result


def get_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Best-effort fundamentals via yfinance; synthetic fallback offline/on failure."""
    out: dict[str, dict] = {}
    if not offline():
        try:
            import yfinance as yf

            for sym in symbols:
                try:
                    info = yf.Ticker(f"{sym}.NS").info
                    out[sym] = {
                        "pe": info.get("trailingPE"),
                        "pb": info.get("priceToBook"),
                        "roe": info.get("returnOnEquity"),
                        "margin": info.get("profitMargins"),
                        "earnings_growth": info.get("earningsGrowth"),
                        "market_cap": info.get("marketCap"),
                    }
                except Exception:  # noqa: BLE001
                    out[sym] = {}
        except Exception as e:  # noqa: BLE001
            log.warning("fundamentals fetch failed (%s); synthetic fallback", e)

    for sym in symbols:
        if not out.get(sym) or all(v is None for v in out[sym].values()):
            rng = np.random.default_rng(_seed(sym) ^ 0x5EED)
            out[sym] = {
                "pe": float(8 + rng.random() * 45),
                "pb": float(0.8 + rng.random() * 9),
                "roe": float(0.03 + rng.random() * 0.30),
                "margin": float(0.02 + rng.random() * 0.28),
                "earnings_growth": float(-0.1 + rng.random() * 0.4),
                "market_cap": float((5 + rng.random() * 500) * 1e9),
            }
    return out


def dump_sample_prices(symbols: list[str], path) -> None:
    """Write synthetic sample prices to a JSON file for committed offline fixtures."""
    payload = {}
    for sym in symbols:
        df = _synthetic_history(sym, CONFIG.lookback_days)
        payload[sym] = {
            "index": [d.strftime("%Y-%m-%d") for d in df.index],
            "close": df["Close"].round(4).tolist(),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
