# 🇮🇳 Indian Stock Signals

Transparent, data-driven screening app that ranks the **top 5 Indian market sectors** and the
**top 5 stocks in each (25 total)** using a multi-factor + machine-learning model built on free
public market data. Every pick comes with a plain-English **rationale** and a **link to an external
interactive chart** to study further.

> ## ⚠️ Not investment advice
> This project is for **educational and informational purposes only**. It is **not** investment
> advice, a recommendation, or a solicitation to buy or sell any security. No model can reliably
> predict prices; outputs are **probabilistic factor signals**, not forecasts. Markets are risky.
> Consult a SEBI-registered adviser before investing.

## What it does

- Ingests historical OHLCV + best-effort fundamentals for a curated universe (~96 liquid NSE names
  across 10 sectors) via Yahoo Finance (`yfinance`, `.NS` tickers).
- Computes cross-sectional **factors**: momentum (12-1, 3M, 6M), trend (50/200-DMA, golden cross),
  quality (ROE, margin, earnings growth), value (P/E, P/B), and low-volatility.
- Blends the factor composite with a **gradient-boosted classifier** that estimates the probability
  of a positive forward return over a configurable horizon (default 21 trading days).
- Scores sectors (index momentum + constituent breadth), selects the **top 5**, and within each the
  **top 5 stocks** → a 25-name shortlist with rationale and TradingView/Yahoo chart links.
- Serves it via a FastAPI REST API and a React (Vite) UI.

Runs **fully offline** (deterministic synthetic data) for CI/sandboxes, and on **live data** when
the network is available.

## Architecture

```
ml/ (Python)  --pipeline-->  backend/app/data/predictions.json
                                   |
backend/ (FastAPI)  --REST-->  frontend/ (React + Vite)
```

See [`docs/`](docs/) for the full **product research**, **PRD**, and **tech spec**.

## Quick start

```bash
# 1) ML pipeline: generate predictions.json
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
iss-pipeline -v                 # live data
# ISS_OFFLINE=1 iss-pipeline -v # offline / deterministic

# 2) Backend API (:8000)
cd ../backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000

# 3) Frontend (:3000)
cd ../frontend
npm install
npm run dev
```

Or use the **Makefile**: `make setup && make pipeline && make api` (in one shell) and `make web`
(in another).

### Docker

```bash
docker compose up --build
# frontend -> http://localhost:3000 , api -> http://localhost:8000
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/meta` | freshness, horizon, universe size, disclaimer |
| GET | `/api/predictions` | full 5×5 payload |
| GET | `/api/sectors` | sector summaries |
| GET | `/api/sectors/{sector}` | one sector + its 5 stocks |

## Tests / lint

```bash
make test    # ml (pytest) + backend (pytest) + frontend (vitest)
make lint    # ruff + mypy + eslint
make build   # frontend production build
```

CI (`.github/workflows/ci.yml`) runs all of the above offline on every push/PR.

## Refreshing the data

`predictions.json` is a snapshot. Re-run `make pipeline` (or schedule the offline/online pipeline
via cron / a GitHub Action) to refresh rankings. The API picks up file changes automatically (cache
keyed by mtime).

## Deployment

- **Backend:** any container host (Render, Railway, Fly.io, ECS) using `backend/Dockerfile`.
- **Frontend:** static hosting (Netlify, Vercel, Cloudflare Pages, S3+CDN) from `npm run build`,
  or the provided nginx image. Set `VITE_API_BASE` to your API URL at build time.

## Methodology & honesty

The ML model uses **only trailing features** (no look-ahead) and forward-return labels during
training; fundamentals feed the quality/value factors only. Scores are rank-percentiles for
interpretability. This is a **screening/ranking** tool grounded in published factor research — not a
price oracle. Read `docs/03-tech-spec.md` for details.

## BMAD

This repo is set up with the [BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) agent
framework (`_bmad/`, skills in `.agents/skills/`) for structured, agent-assisted planning and
development. Generated planning/implementation artifacts go to `_bmad-output/` (git-ignored).

## License

MIT — see [LICENSE](LICENSE).
