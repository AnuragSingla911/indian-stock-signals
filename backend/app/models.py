"""Pydantic models describing the predictions payload / API responses."""

from __future__ import annotations

from pydantic import BaseModel


class ChartLinks(BaseModel):
    tradingview: str
    yahoo: str


class MacroInsight(BaseModel):
    nifty_1m_return: float
    usd_inr_1m_change: float
    brent_1m_return: float
    summary: str
    as_of: str


class StockHighlight(BaseModel):
    symbol: str
    headline: str
    sentiment: float
    publisher: str
    url: str = ""
    published_at: str = ""


class Insights(BaseModel):
    macro: MacroInsight
    stock_highlights: list[StockHighlight]


class StockPick(BaseModel):
    symbol: str
    yahoo_symbol: str
    name: str
    composite_score: float
    up_probability: float
    factors: dict[str, float]
    rationale: str
    chart_links: ChartLinks
    news_sentiment: float = 0.0
    news_headline: str | None = None


class SectorBlock(BaseModel):
    sector: str
    display_name: str
    sector_score: float
    sector_rationale: str
    stocks: list[StockPick]


class Predictions(BaseModel):
    generated_at: str
    horizon_days: int
    universe_size: int
    model_trained: bool
    model_name: str = ""
    model_samples: int
    disclaimer: str
    insights: Insights | None = None
    sectors: list[SectorBlock]


class Meta(BaseModel):
    generated_at: str
    horizon_days: int
    universe_size: int
    model_trained: bool
    model_name: str = ""
    disclaimer: str


class SectorSummary(BaseModel):
    sector: str
    display_name: str
    sector_score: float
    sector_rationale: str
    num_stocks: int
