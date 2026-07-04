"""Central configuration: paths, weights, horizons.

All tunables live here so the scoring methodology is transparent and reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ML_DIR = Path(__file__).resolve().parents[2]  # .../ml
REPO_DIR = ML_DIR.parent
DATA_DIR = ML_DIR / "data"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
SECTORS_CSV = DATA_DIR / "sectors.csv"
PRICE_CACHE_DIR = DATA_DIR / "price_cache"
SAMPLE_PRICE_DIR = DATA_DIR / "sample_prices"
MODEL_PATH = DATA_DIR / "model.joblib"

# Where the API reads predictions from.
PREDICTIONS_PATH = Path(
    os.environ.get("PREDICTIONS_PATH", REPO_DIR / "backend" / "app" / "data" / "predictions.json")
)

DISCLAIMER = (
    "Educational and informational use only. This is NOT investment advice, a recommendation, "
    "or a solicitation to buy or sell any security. Markets are risky and model signals do not "
    "guarantee future results. Consult a SEBI-registered adviser before investing."
)


def offline() -> bool:
    """Force offline mode (use committed sample data) via ISS_OFFLINE=1."""
    return os.environ.get("ISS_OFFLINE", "0").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Config:
    horizon_days: int = 21
    lookback_days: int = 1500  # ~6y of trading history to request (more ML training data)
    top_sectors: int = 5
    stocks_per_sector: int = 5

    # Sampling stride (trading days) for building ML training/eval windows. A smaller step
    # yields more (overlapping) samples from the same history.
    train_step: int = 10

    # Factor-group weights for the cross-sectional composite (sum need not be 1).
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.30,
            "trend": 0.20,
            "quality": 0.20,
            "value": 0.15,
            "lowvol": 0.15,
        }
    )

    # Blend between factor composite and ML probability signal.
    w_factor: float = 0.7
    w_ml: float = 0.3
    ml_scale: float = 4.0  # maps (prob - 0.5) into z-like range

    # Sector score blend.
    sector_momentum_w: float = 0.6
    sector_breadth_w: float = 0.4


CONFIG = Config()
