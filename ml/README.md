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
rationale, and external chart links.

## Test / lint / typecheck

```bash
pytest
ruff check src tests
mypy src
```

See `../docs/03-tech-spec.md` for methodology.
