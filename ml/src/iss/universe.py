"""Load the curated ticker universe and sector metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass

from .config import SECTORS_CSV, UNIVERSE_CSV


@dataclass(frozen=True)
class Stock:
    symbol: str  # NSE symbol, e.g. "TCS"
    sector: str
    name: str

    @property
    def yahoo_symbol(self) -> str:
        return f"{self.symbol}.NS"


@dataclass(frozen=True)
class Sector:
    key: str
    index_ticker: str | None
    display_name: str


def load_universe() -> list[Stock]:
    stocks: list[Stock] = []
    with open(UNIVERSE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stocks.append(
                Stock(
                    symbol=row["symbol"].strip(),
                    sector=row["sector"].strip(),
                    name=row["name"].strip(),
                )
            )
    return stocks


def load_sectors() -> dict[str, Sector]:
    sectors: dict[str, Sector] = {}
    with open(SECTORS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["sector"].strip()
            idx = row["index_ticker"].strip() or None
            sectors[key] = Sector(
                key=key, index_ticker=idx, display_name=row["display_name"].strip()
            )
    return sectors


def stocks_by_sector(stocks: list[Stock]) -> dict[str, list[Stock]]:
    out: dict[str, list[Stock]] = {}
    for s in stocks:
        out.setdefault(s.sector, []).append(s)
    return out
