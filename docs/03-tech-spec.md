# Technical Specification — Indian Stock Signals

> **Disclaimer:** Educational tool only. Not investment advice.

## 1. Architecture

```
                    ┌────────────────────────────────────────────┐
                    │                 ml/ (Python)                │
  Yahoo Finance ───▶│  data.py  → features.py → scoring.py →      │
  (yfinance, v8)    │  model.py → rationale.py → pipeline.py      │
  sector indices    │        writes backend/app/data/predictions.json
                    └────────────────────────────────────────────┘
                                        │  (JSON on disk, cached)
                                        ▼
                    ┌────────────────────────────────────────────┐
                    │            backend/ (FastAPI)               │
                    │  GET /api/health /meta /predictions         │
                    │      /sectors /sectors/{sector}             │
                    └────────────────────────────────────────────┘
                                        │  HTTP/JSON
                                        ▼
                    ┌────────────────────────────────────────────┐
                    │         frontend/ (React + Vite)            │
                    │  Sector cards → stock rows → chart links    │
                    └────────────────────────────────────────────┘
```

**Why this split:** ML + data are Python-native (pandas/scikit-learn/yfinance); FastAPI keeps the
backend in the same language as the model for zero serialization friction; React/Vite for a fast,
modern SPA. The pipeline is decoupled (batch) from serving (stateless API reads a JSON artifact),
which makes the API trivially cacheable and the whole app runnable offline from committed sample
data.

## 2. Repository layout

```
indian-stock-signals/
├── docs/                     # research, PRD, tech spec, methodology
├── ml/
│   ├── src/iss/
│   │   ├── config.py         # weights, horizon, paths, universe path
│   │   ├── universe.py       # load ticker→sector CSV
│   │   ├── data.py           # yfinance fetch + disk cache + offline fallback
│   │   ├── features.py       # technical + fundamental factor computation
│   │   ├── scoring.py        # cross-sectional z-scores → composite 0–100
│   │   ├── model.py          # sklearn classifier: P(fwd return > 0)
│   │   ├── rationale.py      # factor contributions → English rationale
│   │   ├── sectors.py        # sector scoring + selection
│   │   └── pipeline.py       # orchestration → predictions.json
│   ├── data/universe.csv     # curated symbol,sector,name universe (~180)
│   ├── data/sample_prices/   # committed sample OHLCV for offline/CI
│   └── tests/                # pytest unit tests
├── backend/
│   ├── app/main.py           # FastAPI app + routes
│   ├── app/data/predictions.json   # served artifact (committed sample)
│   └── tests/                # API tests (pytest + httpx)
├── frontend/
│   ├── src/ (React components), vite config, api client
├── docker-compose.yml, backend/Dockerfile, frontend/Dockerfile
├── Makefile
└── .github/workflows/ci.yml
```

## 3. ML methodology

### 3.1 Universe
Curated CSV of ~180 liquid large/mid-cap NSE names mapped to sectors. Loaded via `universe.py`.

### 3.2 Features (per stock, using trailing windows only)
- **Momentum:** 12-1 month return (skip most recent month), 3M return, 6M return.
- **Trend:** last close vs 50-DMA and 200-DMA (%), 50/200-DMA crossover flag.
- **Volatility (inverse):** annualized 3M daily-return stdev (lower is better → sign-flipped).
- **RSI(14)**, **volume trend** (20d vs 60d average).
- **Quality (fundamental):** ROE, profit margin, earnings growth (if available).
- **Value (fundamental):** inverse P/E, inverse P/B (cheaper is better).

### 3.3 Cross-sectional scoring
Each raw factor is winsorized (1/99 pct) and z-scored across the current universe. Missing values
imputed to 0 (the cross-sectional mean). Factors are grouped (momentum, trend, quality, value,
lowvol) and combined with configurable weights (`config.py`) into a factor composite `z_total`.

### 3.4 ML signal
A `GradientBoostingClassifier` (scikit-learn) is trained on historical cross-sections: features =
the same factors computed at time *t*; label = 1 if forward `horizon_days` return > 0. Training
uses a time-series split (no look-ahead). Output `up_probability = P(label=1)`. If training data is
insufficient (offline/CI), the model falls back to a deterministic logistic map of `z_total` so the
pipeline always produces a value. Model artifact cached to `ml/data/model.joblib`.

### 3.5 Composite
`composite_raw = w_factor * z_total + w_ml * (up_probability - 0.5) * k`. Mapped to 0–100 via a
rank-percentile transform across the universe for interpretability.

### 3.6 Sector scoring & selection
`sector_score` = 0.6 * (sector index 3M momentum, z-scored across sectors) + 0.4 * breadth (share
of the sector's universe constituents above their 200-DMA). Top 5 sectors selected. Within each,
top 5 stocks by composite.

### 3.7 Rationale generation
Deterministic template driven by each pick's largest positive factor z-scores and any notable
negatives, e.g.: "Strong 12-1 momentum (top decile), above 200-DMA, high ROE; valuation slightly
rich (P/E above median)." No LLM dependency → reproducible and free.

## 4. Backend API (FastAPI)

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{status:"ok"}` |
| GET | `/api/meta` | generated_at, horizon_days, universe_size, disclaimer |
| GET | `/api/predictions` | full `predictions.json` |
| GET | `/api/sectors` | list of sectors with scores (no nested stocks) |
| GET | `/api/sectors/{sector}` | one sector with its 5 stocks |

- CORS enabled for the frontend origin.
- Reads `PREDICTIONS_PATH` (env) → defaults to `backend/app/data/predictions.json`.
- Pydantic models validate the payload shape on load.

## 5. Frontend (React + Vite)

- `api.js` — fetch wrapper (base URL from `VITE_API_BASE`, default `http://localhost:8000`).
- Components: `App`, `Disclaimer`, `MetaBar` (freshness/horizon), `SectorCard`, `StockRow`,
  `ScoreBadge`, `ChartLinks`.
- Responsive grid of 5 sector cards; each expands to 5 stock rows with score badges, rationale,
  and external chart links (open in new tab, `rel="noopener noreferrer"`).
- Loading/error states; fully static-buildable (`npm run build`).

## 6. Configuration

`ml/src/iss/config.py`: factor weights, `HORIZON_DAYS`, cache dir, universe path, offline flag.
Env: `PREDICTIONS_PATH`, `VITE_API_BASE`, `ISS_OFFLINE=1` to force sample data.

## 7. Testing strategy

- **ml/tests:** factor math (known inputs), z-score/winsorize, scoring monotonicity, rationale
  non-empty, sector selection returns 5×5, pipeline end-to-end on sample data producing valid JSON.
- **backend/tests:** each endpoint returns 200 and schema-valid payloads (TestClient).
- **frontend:** component render smoke test (Vitest + Testing Library) + build check.
- All tests run **offline** using committed sample data. CI = lint + typecheck + tests + builds.

## 8. Deployment

- `backend/Dockerfile` (uvicorn), `frontend/Dockerfile` (nginx serving built assets),
  `docker-compose.yml` wiring both. `Makefile` targets: `setup`, `pipeline`, `api`, `web`, `test`,
  `lint`, `docker`.
- Deploy notes for Render/Railway (backend) + Netlify/Vercel/static (frontend) in README.
- Pipeline can be scheduled (cron / GitHub Action) to refresh `predictions.json`.

## 9. CI (GitHub Actions)

`ci.yml`: Python (ruff + mypy + pytest for ml & backend) and Node (eslint + `npm run build` +
vitest) jobs. Runs on push/PR. Offline (no external network needed) via sample data.

## 10. Security & compliance

- No secrets required for core functionality (public data). No PII. Disclaimer everywhere.
- Outbound calls limited to Yahoo endpoints during pipeline runs only.
- Dependencies pinned; `pip-audit`/`npm audit` advisory (non-blocking) noted in README.
