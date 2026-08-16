"""Historical news archive for backtestable sentiment features.

Articles are stored on disk under ``ml/data/news_archive/{symbol}.json`` and can be
refreshed from Finnhub (``FINNHUB_API_KEY``) or bootstrapped deterministically in
offline mode from trailing price action (no look-ahead).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, NEWS_ARCHIVE_DIR, offline
from .news_types import NewsItem, StockNews
from .sentiment import score_sentiment

log = logging.getLogger("iss.news_archive")

_POS_TEMPLATES = (
    "{sym} reports strong quarterly growth",
    "Analysts upgrade {sym} on margin expansion",
    "{sym} wins major contract",
)
_NEG_TEMPLATES = (
    "{sym} faces margin pressure",
    "Regulatory probe weighs on {sym}",
    "{sym} misses earnings estimates",
)


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)


def finnhub_api_key() -> str | None:
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    return key or None


@dataclass
class ArchivedArticle:
    title: str
    published_at: str  # YYYY-MM-DD
    publisher: str
    url: str
    sentiment: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> ArchivedArticle:
        title = str(raw.get("title", "")).strip()
        published = str(raw.get("published_at", ""))[:10]
        return cls(
            title=title,
            published_at=published,
            publisher=str(raw.get("publisher") or raw.get("source") or "Unknown"),
            url=str(raw.get("url") or raw.get("link") or ""),
            sentiment=float(raw.get("sentiment", score_sentiment(title))),
        )


class NewsArchive:
    """Disk-backed news archive with optional Finnhub refresh."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or NEWS_ARCHIVE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, list[ArchivedArticle]] = {}

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace("&", "_and_").replace("-", "_")
        return self.root / f"{safe}.json"

    def load(self, symbol: str) -> list[ArchivedArticle]:
        if symbol in self._memory:
            return self._memory[symbol]
        path = self._path(symbol)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            articles = [ArchivedArticle.from_dict(a) for a in raw.get("articles", [])]
            self._memory[symbol] = articles
            return articles
        except Exception as e:  # noqa: BLE001
            log.warning("failed to load archive for %s (%s)", symbol, e)
            return []

    def save(self, symbol: str, articles: list[ArchivedArticle]) -> None:
        payload = {
            "symbol": symbol,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "articles": [a.to_dict() for a in articles],
        }
        path = self._path(symbol)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._memory[symbol] = articles

    def merge(self, symbol: str, incoming: list[ArchivedArticle]) -> list[ArchivedArticle]:
        existing = self.load(symbol)
        seen = {(a.title, a.published_at) for a in existing}
        merged = list(existing)
        for article in incoming:
            key = (article.title, article.published_at)
            if key not in seen and article.title:
                merged.append(article)
                seen.add(key)
        merged.sort(key=lambda a: a.published_at)
        self.save(symbol, merged)
        return merged

    def sentiment_on_date(
        self,
        symbol: str,
        as_of: pd.Timestamp,
        lookback_days: int | None = None,
        close: pd.Series | None = None,
    ) -> tuple[float, float]:
        """Average sentiment and log article count in the trailing window ending at as_of."""
        lookback = lookback_days if lookback_days is not None else CONFIG.news_lookback_days
        articles = self._articles_for_symbol(symbol, close)
        as_of_date = pd.Timestamp(as_of).normalize()
        start = as_of_date - pd.Timedelta(days=lookback)
        window = [
            a
            for a in articles
            if a.published_at and start <= pd.Timestamp(a.published_at) <= as_of_date
        ]
        if not window:
            return 0.0, 0.0
        sentiments = [a.sentiment for a in window]
        return float(np.mean(sentiments)), float(np.log1p(len(window)))

    def _articles_for_symbol(self, symbol: str, close: pd.Series | None) -> list[ArchivedArticle]:
        articles = self.load(symbol)
        if articles:
            return articles
        if offline() and close is not None:
            bootstrapped = bootstrap_synthetic_archive(symbol, close)
            self._memory[symbol] = bootstrapped
            return bootstrapped
        return []

    def to_stock_news(self, symbol: str, close: pd.Series | None = None) -> StockNews:
        """Current snapshot for inference / insights."""
        as_of = (
            pd.Timestamp(close.index[-1])
            if close is not None and len(close)
            else pd.Timestamp.now()
        )
        articles = self._articles_for_symbol(symbol, close)
        lookback = CONFIG.news_lookback_days
        start = as_of.normalize() - pd.Timedelta(days=lookback)
        recent = [
            a
            for a in articles
            if a.published_at and start <= pd.Timestamp(a.published_at) <= as_of.normalize()
        ]
        recent.sort(key=lambda a: a.published_at, reverse=True)
        headlines = [
            NewsItem(
                title=a.title,
                publisher=a.publisher,
                url=a.url,
                published_at=a.published_at,
                sentiment=a.sentiment,
            )
            for a in recent[: CONFIG.news_max_headlines]
        ]
        sentiment, _ = self.sentiment_on_date(symbol, as_of, lookback, close)
        return StockNews(
            symbol=symbol,
            sentiment=sentiment,
            article_count=len(recent),
            headlines=headlines,
        )

    def fetch_finnhub(
        self, symbol: str, from_date: str, to_date: str, api_key: str | None = None
    ) -> list[ArchivedArticle]:
        token = api_key or finnhub_api_key()
        if not token:
            return []
        params = urllib.parse.urlencode(
            {
                "symbol": f"{symbol}.NS",
                "from": from_date,
                "to": to_date,
                "token": token,
            }
        )
        url = f"https://finnhub.io/api/v1/company-news?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                raw_items = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            log.warning("Finnhub fetch failed for %s (%s)", symbol, e)
            return []

        articles: list[ArchivedArticle] = []
        for raw in raw_items:
            headline = str(raw.get("headline", "")).strip()
            if not headline:
                continue
            ts = raw.get("datetime")
            if isinstance(ts, (int, float)):
                published = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                published = str(ts)[:10] if ts else ""
            articles.append(
                ArchivedArticle(
                    title=headline,
                    published_at=published,
                    publisher=str(raw.get("source") or "Finnhub"),
                    url=str(raw.get("url") or ""),
                    sentiment=score_sentiment(headline),
                )
            )
        return articles

    def refresh_symbol(
        self,
        symbol: str,
        days_back: int | None = None,
        api_key: str | None = None,
        throttle_sec: float = 1.05,
    ) -> int:
        """Fetch recent history from Finnhub and merge into the on-disk archive."""
        days_back = days_back or CONFIG.news_archive_days
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=days_back)
        incoming = self.fetch_finnhub(
            symbol,
            from_date.isoformat(),
            to_date.isoformat(),
            api_key=api_key,
        )
        if throttle_sec > 0:
            time.sleep(throttle_sec)
        if not incoming:
            return 0
        before = len(self.load(symbol))
        merged = self.merge(symbol, incoming)
        return len(merged) - before

    def refresh_all(
        self,
        symbols: list[str],
        days_back: int | None = None,
        api_key: str | None = None,
    ) -> dict[str, int]:
        added: dict[str, int] = {}
        for sym in symbols:
            added[sym] = self.refresh_symbol(sym, days_back=days_back, api_key=api_key)
        return added


def bootstrap_synthetic_archive(
    symbol: str, close: pd.Series, step: int | None = None
) -> list[ArchivedArticle]:
    """Deterministic pseudo-news keyed on trailing price action (offline / no API)."""
    step = step if step is not None else CONFIG.train_step
    close = close.dropna()
    articles: list[ArchivedArticle] = []
    if len(close) < 260:
        return articles

    for t in range(260, len(close), step):
        as_of = pd.Timestamp(close.index[t])
        window = close.iloc[max(0, t - 10) : t + 1]
        ret = float(window.iloc[-1] / window.iloc[0] - 1.0) if len(window) > 1 else 0.0
        rng = np.random.default_rng(_seed(symbol) ^ int(as_of.strftime("%Y%m%d")))
        positive = ret > 0.01 or (abs(ret) <= 0.01 and rng.random() > 0.45)
        tpl = rng.choice(_POS_TEMPLATES if positive else _NEG_TEMPLATES)
        title = tpl.format(sym=symbol)
        articles.append(
            ArchivedArticle(
                title=title,
                published_at=as_of.strftime("%Y-%m-%d"),
                publisher="SyntheticArchive",
                url="",
                sentiment=score_sentiment(title),
            )
        )
    return articles


def get_news_archive() -> NewsArchive:
    return NewsArchive()
