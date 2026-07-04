# Product Research — Indian Stock Signals

> **Disclaimer:** This document and the product it describes are for **educational and
> informational purposes only**. Nothing here is investment advice, a recommendation, or a
> solicitation to buy or sell any security. Equity markets are risky; past performance and
> model signals do not guarantee future results. Consult a SEBI-registered adviser before
> investing.

## 1. Problem & Vision

Retail investors in India face an overwhelming universe (~2,000+ liquid NSE/BSE stocks across
~15 sectors) and fragmented, often low-quality "tip" content. They want a **short, explainable
shortlist** with reasoning they can verify, not a black box.

**Vision:** A transparent, data-driven screening app that surfaces the **top 5 sectors** and
**5 stocks per sector (25 total)** ranked by a composite quant score, each with a plain-English
**rationale** and a **link to an external interactive chart** for further study.

We deliberately frame output as **probabilistic signals / rankings**, not price forecasts. The
honest, defensible claim is: "these names currently score highly on momentum, quality, value and
trend factors" — not "these will go up."

## 2. Feasibility of "prediction"

Academic consensus (Efficient Market Hypothesis, weak/semi-strong form) says short-horizon price
prediction from public data is extremely hard; realized out-of-sample accuracy for direction is
typically 50–55%. What *does* have robust, published evidence of cross-sectional explanatory power
is **factor investing**: momentum, quality, low-volatility, value, and size factors (Fama-French,
Carhart, AQR). Our model therefore ranks stocks by a **multi-factor composite** plus a lightweight
ML classifier that estimates the probability of positive forward return over a chosen horizon
(e.g., 21 trading days). This is defensible, reproducible, and explainable.

## 3. Data sources (all free / public)

| Source | Access | Data | Notes |
|---|---|---|---|
| Yahoo Finance (`query1.finance.yahoo.com` v8 chart API, via `yfinance`) | Free, no key | Historical OHLCV, corporate actions, some fundamentals | Indian tickers use `.NS` (NSE) / `.BO` (BSE) suffix. Verified reachable from this environment. |
| NSE sectoral indices (Nifty IT, Bank, FMCG, Auto, Pharma, Energy, Metal, Realty, etc.) | Via Yahoo index tickers (e.g. `^CNXIT`, `^NSEBANK`) | Sector index OHLCV for sector momentum | Used to rank sectors. |
| yfinance `.info` / `.fast_info` | Free | Market cap, P/E, P/B, ROE, margins (best-effort) | Fundamentals coverage is patchy; model degrades gracefully when a field is missing. |
| Static seed universe (curated CSV in repo) | Bundled | Ticker → sector mapping for a liquid large/mid-cap universe | Avoids scraping fragile endpoints; ~150–200 names across sectors. |

**External chart links** (per stock, opened in a new tab): TradingView
(`https://www.tradingview.com/chart/?symbol=NSE:<SYMBOL>`) and Yahoo Finance
(`https://finance.yahoo.com/quote/<SYMBOL>.NS`).

### Data risk & mitigations
- **Rate limiting / blocking from Yahoo:** batch requests, cache to disk, add retry/backoff, and
  ship a committed sample dataset so the app and CI work offline.
- **Missing fundamentals:** factor scores are computed on available fields; missing values are
  imputed to the cross-sectional median and down-weighted.
- **Survivorship / look-ahead bias:** features use only trailing windows; labels use strictly
  forward returns during model training/backtest.

## 4. Sector taxonomy (NSE-aligned)

Top-level sectors we score and rank: **IT, Banking/Financials, FMCG, Auto, Pharma/Healthcare,
Energy/Oil&Gas, Metal, Realty, Infrastructure, Consumer Durables**. The app selects the **top 5**
by a sector-momentum + breadth composite each run.

## 5. Competitive landscape

| Product | What it does | Gap we address |
|---|---|---|
| Screener.in | Powerful fundamental screener | No opinionated ranked shortlist or ML signal; steep learning curve |
| Tickertape / Trendlyne | Scores, screeners, "SMART" scores | Paywalled depth; not open/transparent methodology |
| TradingView | Charting + community | No India-specific ranked pick list |
| Broker "research" | Analyst calls | Opaque, conflicted, not reproducible |

**Our differentiation:** fully **open, reproducible methodology**; concise 25-name output grouped
by sector; every pick has a transparent factor-based rationale and a deep-link to study further.

## 6. Target users & core use case

- **Primary:** self-directed retail investor doing weekly idea generation.
- **Secondary:** finance students / hobbyists learning factor investing.

**Golden path:** open app → see 5 sector cards → each lists 5 ranked stocks with score + rationale
→ click a stock's chart link to study externally.

## 7. Success metrics (product)

- Time-to-shortlist < 5s on a cached run.
- Every pick has a non-empty, factor-grounded rationale and a working external chart link.
- Backtest: composite top-quintile beats equal-weight universe on information ratio over the
  sample period (sanity check, not a performance promise).

## 8. Risks & compliance

- **Not investment advice** — disclaimers on every screen and in the API payload.
- No order placement, no PII, no accounts in v1 → minimal regulatory surface.
- Clearly label data freshness and the model horizon.

## 9. Scope decision (v1)

In: batch pipeline, factor + ML ranking, 25-pick JSON, REST API, React UI, external chart links,
disclaimers, Docker deploy, CI.
Out (future): live intraday data, user accounts/watchlists, backtest UI, alerts, more exchanges.
