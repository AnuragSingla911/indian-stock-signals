"""Shared datatypes for news headlines and per-symbol snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NewsItem:
    title: str
    publisher: str
    url: str
    published_at: str
    sentiment: float


@dataclass
class StockNews:
    symbol: str
    sentiment: float
    article_count: int
    headlines: list[NewsItem] = field(default_factory=list)
