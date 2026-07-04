# Product Requirements Document (PRD) — Indian Stock Signals

> **Disclaimer:** Educational tool only. Not investment advice. See `01-product-research.md`.

## 1. Overview

Indian Stock Signals is a web app that presents a ranked shortlist of **25 Indian equities — the
top 5 stocks in each of the top 5 sectors** — using a transparent, multi-factor + ML scoring
model built on free public market data. Each pick includes a plain-English rationale and a link to
an external interactive chart.

## 2. Goals / Non-goals

**Goals**
- G1: Rank sectors and select the top 5.
- G2: Within each selected sector, rank constituents and select the top 5.
- G3: Produce a factor-grounded rationale string per stock.
- G4: Expose results via a REST API and a responsive React UI.
- G5: Provide per-stock external chart deep-links (TradingView + Yahoo).
- G6: Be fully reproducible and runnable offline via committed sample data.

**Non-goals (v1)**: brokerage/order placement, user auth, real-time streaming, portfolio
tracking, mobile apps, multi-country support.

## 3. Personas

- **Riya, 29, self-directed investor** — wants a weekly, explainable shortlist to research.
- **Arjun, 21, finance student** — wants to learn how factor ranking works, transparently.

## 4. User stories & acceptance criteria

### US-1 — View top sectors
As a user, I see the 5 top-ranked sectors.
- **AC:** UI shows exactly 5 sector cards, each with sector name, a sector score, and a one-line
  sector rationale (e.g., strongest 3-month index momentum + breadth).

### US-2 — View 5 stocks per sector
As a user, within each sector I see 5 ranked stocks.
- **AC:** Each sector card lists exactly 5 stocks ordered by composite score (desc), showing
  symbol, company name, composite score (0–100), and forward-return probability.

### US-3 — Understand the rationale
As a user, I understand *why* each stock was picked.
- **AC:** Each stock shows a rationale citing its top contributing factors (e.g., "Strong 12-1
  momentum (top decile), high ROE, reasonable P/E, price above 200-DMA").

### US-4 — Study externally
As a user, I can open an external chart for any stock.
- **AC:** Each stock has a "Chart" link → TradingView (`NSE:<symbol>`) and a Yahoo link, opening in
  a new tab.

### US-5 — Know the data freshness & disclaimer
- **AC:** UI shows "data as of <date>", the model horizon, and a visible not-investment-advice
  disclaimer. The same metadata is present in the API payload.

### US-6 — Refresh data (operator)
As an operator, I can regenerate rankings by running the pipeline.
- **AC:** A single command (`make pipeline` / CLI) fetches data, scores, and writes
  `predictions.json`; the API serves the latest file. Graceful fallback to bundled sample data on
  network failure.

## 5. Functional requirements

- FR-1: Pipeline computes technical factors (momentum 12-1, 3M/6M returns, volatility, RSI,
  distance from 50/200-DMA, volume trend).
- FR-2: Pipeline computes fundamental factors when available (P/E, P/B, ROE, profit margin,
  earnings growth), imputing missing to median.
- FR-3: Composite score = weighted, cross-sectionally z-scored blend of factor groups, mapped to
  0–100.
- FR-4: ML classifier estimates P(forward 21-day return > 0) per stock; blended into the score.
- FR-5: Sector score = index momentum + constituent breadth (share of constituents above 200-DMA).
- FR-6: Select top 5 sectors, then top 5 stocks per sector → 25 picks.
- FR-7: Rationale generator produces a human-readable explanation from each pick's factor
  contributions.
- FR-8: API endpoints: `GET /api/health`, `GET /api/meta`, `GET /api/predictions`,
  `GET /api/sectors`, `GET /api/sectors/{sector}`.
- FR-9: UI renders sectors → stocks with scores, rationale, chart links, disclaimer, freshness.

## 6. Non-functional requirements

- NFR-1 (Perf): API serves cached JSON < 200ms; UI first paint < 2s on cached run.
- NFR-2 (Reliability): pipeline retries + backoff; offline fallback to sample data; CI runs
  offline.
- NFR-3 (Transparency): scoring weights and formulas documented and configurable.
- NFR-4 (Portability): `docker compose up` runs API + UI locally.
- NFR-5 (Quality): unit tests for factor math, scoring, rationale; lint + typecheck in CI.
- NFR-6 (Compliance): disclaimer surfaced in UI and API; no PII stored.

## 7. Data contract (`predictions.json`)

```jsonc
{
  "generated_at": "2026-07-04T18:00:00Z",
  "horizon_days": 21,
  "universe_size": 180,
  "disclaimer": "Educational only. Not investment advice.",
  "sectors": [
    {
      "sector": "IT",
      "sector_score": 78.4,
      "sector_rationale": "Nifty IT up 9.2% over 3M; 72% of constituents above 200-DMA.",
      "stocks": [
        {
          "symbol": "TCS",
          "yahoo_symbol": "TCS.NS",
          "name": "Tata Consultancy Services",
          "composite_score": 82.1,
          "up_probability": 0.61,
          "factors": { "momentum": 1.8, "quality": 1.2, "value": -0.3, "trend": 1.1 },
          "rationale": "Strong 12-1 momentum (top decile), high ROE, price above 200-DMA; valuation slightly rich.",
          "chart_links": {
            "tradingview": "https://www.tradingview.com/chart/?symbol=NSE:TCS",
            "yahoo": "https://finance.yahoo.com/quote/TCS.NS"
          }
        }
        // ... 5 total
      ]
    }
    // ... 5 total
  ]
}
```

## 8. Release plan

- **M1:** docs (research, PRD, tech spec) + BMAD setup.
- **M2:** data pipeline + factor/ML scoring + sample dataset.
- **M3:** API + UI.
- **M4:** tests, CI, Docker, deploy docs.

## 9. Open questions / assumptions

- Universe is a curated bundled CSV (~180 liquid names) to avoid fragile scraping; expandable.
- Fundamentals are best-effort from Yahoo; model degrades gracefully.
- Horizon default 21 trading days; configurable.
