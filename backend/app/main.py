"""Indian Stock Signals REST API.

Serves the batch-generated predictions.json artifact. Stateless and cache-friendly.
Educational only — not investment advice.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Meta, Predictions, SectorBlock, SectorSummary

DEFAULT_PREDICTIONS = Path(__file__).resolve().parent / "data" / "predictions.json"
PREDICTIONS_PATH = Path(os.environ.get("PREDICTIONS_PATH", DEFAULT_PREDICTIONS))

app = FastAPI(
    title="Indian Stock Signals API",
    version="0.1.0",
    description="Educational factor + ML ranking of Indian equities. NOT investment advice.",
)

_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _load_raw(path_str: str, mtime: float) -> dict:
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


def load_predictions() -> Predictions:
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Predictions not available yet. Run the ML pipeline to generate them.",
        )
    # Cache keyed by mtime so edits to the file are picked up without a restart.
    raw = _load_raw(str(PREDICTIONS_PATH), PREDICTIONS_PATH.stat().st_mtime)
    return Predictions.model_validate(raw)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta", response_model=Meta)
def meta() -> Meta:
    p = load_predictions()
    return Meta(
        generated_at=p.generated_at,
        horizon_days=p.horizon_days,
        universe_size=p.universe_size,
        model_trained=p.model_trained,
        model_name=p.model_name,
        disclaimer=p.disclaimer,
    )


@app.get("/api/predictions", response_model=Predictions)
def predictions() -> Predictions:
    return load_predictions()


@app.get("/api/sectors", response_model=list[SectorSummary])
def sectors() -> list[SectorSummary]:
    p = load_predictions()
    return [
        SectorSummary(
            sector=s.sector,
            display_name=s.display_name,
            sector_score=s.sector_score,
            sector_rationale=s.sector_rationale,
            num_stocks=len(s.stocks),
        )
        for s in p.sectors
    ]


@app.get("/api/sectors/{sector}", response_model=SectorBlock)
def sector_detail(sector: str) -> SectorBlock:
    p = load_predictions()
    for s in p.sectors:
        if s.sector.lower() == sector.lower():
            return s
    raise HTTPException(status_code=404, detail=f"Sector '{sector}' not found")
