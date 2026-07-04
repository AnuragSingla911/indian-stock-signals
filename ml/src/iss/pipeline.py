"""End-to-end pipeline: fetch -> features -> score -> select -> rationale -> predictions.json."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from urllib.parse import quote

import pandas as pd

from .config import CONFIG, DISCLAIMER, PREDICTIONS_PATH
from .data import get_fundamentals, get_price_history
from .features import build_feature_frame
from .model import predict_up_proba, train_model
from .rationale import sector_rationale, stock_rationale
from .scoring import (
    FACTOR_LABELS,
    factor_composite,
    group_scores,
    to_0_100,
    winsorize_zscore,
)
from .sectors import score_sectors
from .universe import load_sectors, load_universe, stocks_by_sector

log = logging.getLogger("iss.pipeline")


def _chart_links(symbol: str) -> dict[str, str]:
    tv = symbol.replace("&", "_").replace("-", "_")
    return {
        "tradingview": f"https://www.tradingview.com/chart/?symbol=NSE:{quote(tv)}",
        "yahoo": f"https://finance.yahoo.com/quote/{quote(symbol)}.NS",
    }


def run() -> dict:
    stocks = load_universe()
    sectors_meta = load_sectors()
    by_sector = stocks_by_sector(stocks)
    symbols = [s.symbol for s in stocks]
    lookup = {s.symbol: s for s in stocks}

    log.info("Fetching price history for %d symbols", len(symbols))
    prices = get_price_history(symbols)
    fundamentals = get_fundamentals(symbols)

    feature_frame = build_feature_frame(prices, fundamentals)
    zframe = winsorize_zscore(feature_frame)
    groups = group_scores(zframe)
    z_total = factor_composite(groups, CONFIG.weights)

    model = train_model(prices, horizon=CONFIG.horizon_days)
    up_proba = predict_up_proba(model, feature_frame)

    blended = (
        CONFIG.w_factor * z_total
        + CONFIG.w_ml * (up_proba - 0.5) * CONFIG.ml_scale
    )
    composite = to_0_100(blended)

    # Sector scoring.
    index_tickers = [m.index_ticker for m in sectors_meta.values() if m.index_ticker]
    index_prices = get_price_history(index_tickers) if index_tickers else {}
    sector_scores = score_sectors(sectors_meta, by_sector, feature_frame, index_prices)
    top_sectors = sector_scores[: CONFIG.top_sectors]

    label_cols = [c for c in FACTOR_LABELS if c in zframe.columns]

    payload_sectors = []
    for sc in top_sectors:
        sector_symbols = [
            s.symbol for s in by_sector.get(sc.key, []) if s.symbol in composite.index
        ]
        ranked = composite.loc[sector_symbols].sort_values(ascending=False)
        picks = ranked.index[: CONFIG.stocks_per_sector]

        stock_entries = []
        for sym in picks:
            g = groups.loc[sym]
            stock_entries.append(
                {
                    "symbol": sym,
                    "yahoo_symbol": f"{sym}.NS",
                    "name": lookup[sym].name,
                    "composite_score": float(round(composite[sym], 1)),
                    "up_probability": float(round(up_proba[sym], 3)),
                    "factors": {k: float(round(g[k], 2)) for k in groups.columns},
                    "rationale": stock_rationale(
                        cast(pd.Series, zframe.loc[sym, label_cols]),
                        float(up_proba[sym]),
                        float(composite[sym]),
                    ),
                    "chart_links": _chart_links(sym),
                }
            )

        payload_sectors.append(
            {
                "sector": sc.key,
                "display_name": sc.display_name,
                "sector_score": float(round(sc.score, 1)),
                "sector_rationale": sector_rationale(sc.display_name, sc.momentum_3m, sc.breadth),
                "stocks": stock_entries,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_days": CONFIG.horizon_days,
        "universe_size": len(symbols),
        "model_trained": model.trained,
        "model_samples": model.n_samples,
        "disclaimer": DISCLAIMER,
        "sectors": payload_sectors,
    }
    return payload


def write_predictions(payload: dict, path=PREDICTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian Stock Signals pipeline")
    parser.add_argument("--out", default=str(PREDICTIONS_PATH), help="output predictions.json path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    payload = run()
    write_predictions(payload, Path(args.out))

    n_stocks = sum(len(s["stocks"]) for s in payload["sectors"])
    print(
        f"Generated {len(payload['sectors'])} sectors / {n_stocks} stocks "
        f"(model_trained={payload['model_trained']}) -> {args.out}"
    )
    for s in payload["sectors"]:
        names = ", ".join(x["symbol"] for x in s["stocks"])
        print(f"  [{s['sector_score']:>5}] {s['display_name']}: {names}")


if __name__ == "__main__":
    main()
