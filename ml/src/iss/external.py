"""External context: macro indicators and news sentiment.

Fetches macro series (Nifty, USD/INR, Brent) and per-symbol news via yfinance.
Sentiment is lexicon-based (no LLM) for reproducibility and offline CI support.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import CONFIG, offline
from .news_archive import ArchivedArticle, NewsArchive, get_news_archive
from .news_types import NewsItem, StockNews
from .sentiment import score_sentiment

log = logging.getLogger("iss.external")

_MACRO_TICKERS = {
    "nifty": "^NSEI",
    "usd_inr": "USDINR=X",
    "brent": "BZ=F",
}


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)


@dataclass
class MacroContext:
    nifty_1m_return: float
    usd_inr_1m_change: float
    brent_1m_return: float
    summary: str
    as_of: str


def _pct_return(close: pd.Series, days: int = 21) -> float:
    close = close.dropna()
    if len(close) <= days:
        return 0.0
    old = close.iloc[-1 - days]
    if old == 0 or np.isnan(old):
        return 0.0
    return float(close.iloc[-1] / old - 1.0)


def _fetch_macro_live() -> MacroContext:
    import yfinance as yf

    series: dict[str, pd.Series] = {}
    for key, ticker in _MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                series[key] = hist["Close"]
        except Exception:  # noqa: BLE001
            continue

    nifty_ret = _pct_return(series.get("nifty", pd.Series(dtype=float)))
    inr_chg = _pct_return(series.get("usd_inr", pd.Series(dtype=float)))
    oil_ret = _pct_return(series.get("brent", pd.Series(dtype=float)))

    parts: list[str] = []
    if nifty_ret >= 0.03:
        parts.append(f"Nifty up {nifty_ret * 100:.1f}% over 1M (risk-on)")
    elif nifty_ret <= -0.03:
        parts.append(f"Nifty down {abs(nifty_ret) * 100:.1f}% over 1M (cautious tape)")
    else:
        parts.append(f"Nifty flat ({nifty_ret * 100:+.1f}% over 1M)")

    if inr_chg >= 0.01:
        parts.append(f"INR weakening ({inr_chg * 100:+.1f}% USD/INR 1M)")
    elif inr_chg <= -0.01:
        parts.append(f"INR strengthening ({inr_chg * 100:+.1f}% USD/INR 1M)")

    if oil_ret >= 0.05:
        parts.append(f"Brent up {oil_ret * 100:.1f}% (import-cost headwind)")
    elif oil_ret <= -0.05:
        parts.append(f"Brent down {abs(oil_ret) * 100:.1f}% (margin tailwind)")

    summary = "; ".join(parts) if parts else "Macro context unavailable."
    return MacroContext(
        nifty_1m_return=nifty_ret,
        usd_inr_1m_change=inr_chg,
        brent_1m_return=oil_ret,
        summary=summary,
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _synthetic_macro() -> MacroContext:
    rng = np.random.default_rng(42)
    nifty_ret = float(rng.normal(0.01, 0.03))
    inr_chg = float(rng.normal(0.005, 0.01))
    oil_ret = float(rng.normal(0.0, 0.04))
    return MacroContext(
        nifty_1m_return=nifty_ret,
        usd_inr_1m_change=inr_chg,
        brent_1m_return=oil_ret,
        summary=(
            f"Offline macro snapshot: Nifty {nifty_ret * 100:+.1f}% 1M; "
            f"USD/INR {inr_chg * 100:+.1f}%; Brent {oil_ret * 100:+.1f}%."
        ),
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def get_macro_context() -> MacroContext:
    if offline():
        return _synthetic_macro()
    try:
        return _fetch_macro_live()
    except Exception as e:  # noqa: BLE001
        log.warning("macro fetch failed (%s); synthetic fallback", e)
        return _synthetic_macro()


def _parse_news_item(raw: dict) -> NewsItem | None:
    title = raw.get("title") or raw.get("headline") or ""
    if not title:
        return None
    ts = raw.get("providerPublishTime") or raw.get("pubDate")
    if isinstance(ts, (int, float)):
        published = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    else:
        published = str(ts)[:10] if ts else ""
    return NewsItem(
        title=title.strip(),
        publisher=str(raw.get("publisher") or raw.get("source") or "Unknown"),
        url=str(raw.get("link") or raw.get("url") or ""),
        published_at=published,
        sentiment=score_sentiment(title),
    )


def _fetch_news_live(symbols: list[str], max_headlines: int) -> dict[str, StockNews]:
    import yfinance as yf

    out: dict[str, StockNews] = {}
    for sym in symbols:
        try:
            raw_items = yf.Ticker(f"{sym}.NS").news or []
        except Exception:  # noqa: BLE001
            raw_items = []
        headlines: list[NewsItem] = []
        for raw in raw_items[:max_headlines]:
            item = _parse_news_item(raw)
            if item:
                headlines.append(item)
        if headlines:
            avg_sent = float(np.mean([h.sentiment for h in headlines]))
        else:
            avg_sent = 0.0
        out[sym] = StockNews(
            symbol=sym,
            sentiment=avg_sent,
            article_count=len(headlines),
            headlines=headlines,
        )
    return out


def _synthetic_news(symbols: list[str], max_headlines: int) -> dict[str, StockNews]:
    templates_pos = [
        "{sym} reports strong quarterly growth",
        "Analysts upgrade {sym} on margin expansion",
        "{sym} wins major contract",
    ]
    templates_neg = [
        "{sym} faces margin pressure",
        "Regulatory probe weighs on {sym}",
        "{sym} misses earnings estimates",
    ]
    out: dict[str, StockNews] = {}
    for sym in symbols:
        rng = np.random.default_rng(_seed(sym) ^ 0x5EED)
        n = int(rng.integers(1, max_headlines + 1))
        headlines: list[NewsItem] = []
        for _ in range(n):
            pos = rng.random() > 0.45
            tpl = rng.choice(templates_pos if pos else templates_neg)
            title = tpl.format(sym=sym)
            headlines.append(
                NewsItem(
                    title=title,
                    publisher="SyntheticWire",
                    url="",
                    published_at="2026-01-01",
                    sentiment=score_sentiment(title),
                )
            )
        out[sym] = StockNews(
            symbol=sym,
            sentiment=float(np.mean([h.sentiment for h in headlines])),
            article_count=len(headlines),
            headlines=headlines,
        )
    return out


def get_news_features(
    symbols: list[str],
    max_headlines: int | None = None,
    prices: dict[str, pd.DataFrame] | None = None,
    archive: NewsArchive | None = None,
) -> dict[str, StockNews]:
    """Return per-symbol news sentiment and recent headlines.

    Uses the on-disk news archive when available; falls back to live yfinance fetch,
    then deterministic synthetic headlines in offline mode.
    """
    max_headlines = max_headlines or CONFIG.news_max_headlines
    archive = archive or get_news_archive()

    out: dict[str, StockNews] = {}
    for sym in symbols:
        close = prices[sym]["Close"] if prices and sym in prices else None
        sn = archive.to_stock_news(sym, close=close)
        if sn.article_count > 0:
            out[sym] = sn

    missing = [s for s in symbols if s not in out]
    if missing and not offline():
        try:
            live = _fetch_news_live(missing, max_headlines)
            for sym, sn in live.items():
                out[sym] = sn
                archive.merge(
                    sym,
                    [
                        ArchivedArticle(
                            title=h.title,
                            published_at=h.published_at,
                            publisher=h.publisher,
                            url=h.url,
                            sentiment=h.sentiment,
                        )
                        for h in sn.headlines
                    ],
                )
            missing = [s for s in missing if s not in out]
        except Exception as e:  # noqa: BLE001
            log.warning("news fetch failed (%s); synthetic fallback", e)

    if missing:
        synthetic = _synthetic_news(missing, max_headlines)
        out.update(synthetic)
    return out


def build_insights(
    macro: MacroContext,
    news: dict[str, StockNews],
    pick_symbols: list[str],
    max_highlights: int = 8,
) -> dict:
    """Assemble market-level insights for the predictions payload."""
    highlights: list[dict] = []
    for sym in pick_symbols:
        sn = news.get(sym)
        if not sn or not sn.headlines:
            continue
        top = max(sn.headlines, key=lambda h: abs(h.sentiment))
        highlights.append(
            {
                "symbol": sym,
                "headline": top.title,
                "sentiment": round(top.sentiment, 2),
                "publisher": top.publisher,
                "url": top.url,
                "published_at": top.published_at,
            }
        )
    highlights.sort(key=lambda h: abs(h["sentiment"]), reverse=True)

    return {
        "macro": {
            "nifty_1m_return": round(macro.nifty_1m_return, 4),
            "usd_inr_1m_change": round(macro.usd_inr_1m_change, 4),
            "brent_1m_return": round(macro.brent_1m_return, 4),
            "summary": macro.summary,
            "as_of": macro.as_of,
        },
        "stock_highlights": highlights[:max_highlights],
    }
