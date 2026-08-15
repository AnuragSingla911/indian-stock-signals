# iss — Indian Stock Signals (ML pipeline)

Factor + ML ranking pipeline for Indian equities. **Educational only — not investment advice.**

## Install

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the pipeline

```bash
iss-pipeline -v                 # live data (yfinance), writes backend predictions.json
ISS_OFFLINE=1 iss-pipeline -v   # offline, deterministic synthetic data (CI-safe)
```

Output: `backend/app/data/predictions.json` — top 5 sectors, 5 stocks each, with scores,
rationale, news insights, and external chart links.

### News archive (backtestable sentiment)

Historical headlines are stored under `ml/data/news_archive/` and used for both ML training
and inference. Refresh from Finnhub (free tier):

```bash
export FINNHUB_API_KEY=your_key   # https://finnhub.io
iss-news-archive -v               # fetches ~2y of headlines for the universe
```

Without an API key, offline mode bootstraps deterministic pseudo-news from trailing price
action so CI and local runs stay reproducible.

## Test / lint / typecheck

```bash
pytest
ruff check src tests
mypy src
```

See `../docs/03-tech-spec.md` for methodology.
