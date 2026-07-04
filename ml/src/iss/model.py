"""ML signal: probability that a stock's forward return over the horizon is positive.

A GradientBoostingClassifier is trained on pooled historical cross-sections using only
technical features (no look-ahead). When training data is insufficient (e.g. offline/CI),
a deterministic logistic fallback keeps the pipeline fully functional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import TECH_FEATURES, technical_panel

log = logging.getLogger("iss.model")


@dataclass
class SignalModel:
    estimator: object | None = None  # sklearn estimator or None (fallback)
    n_samples: int = 0

    @property
    def trained(self) -> bool:
        return self.estimator is not None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_model(prices: dict[str, pd.DataFrame], horizon: int, step: int = 21) -> SignalModel:
    rows: list[list[float]] = []
    labels: list[int] = []
    for df in prices.values():
        for feats, label in technical_panel(df["Close"], horizon=horizon, step=step):
            rows.append([float(feats.get(k, 0.0)) for k in TECH_FEATURES])
            labels.append(label)

    if len(labels) < 60 or len(set(labels)) < 2:
        log.info("Insufficient training data (%d samples); using logistic fallback", len(labels))
        return SignalModel(estimator=None, n_samples=len(labels))

    try:
        from sklearn.ensemble import GradientBoostingClassifier

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)
        clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        )
        clf.fit(X, y)
        return SignalModel(estimator=clf, n_samples=len(labels))
    except Exception as e:  # noqa: BLE001
        log.warning("model training failed (%s); using fallback", e)
        return SignalModel(estimator=None, n_samples=len(labels))


def predict_up_proba(model: SignalModel, feature_frame: pd.DataFrame) -> pd.Series:
    X = feature_frame.reindex(columns=TECH_FEATURES).astype(float).fillna(0.0)
    if model.trained:
        proba = model.estimator.predict_proba(X.values)[:, 1]  # type: ignore[attr-defined]
        return pd.Series(proba, index=feature_frame.index)

    # Deterministic fallback: logistic of momentum + long-term trend.
    raw = 5.0 * (0.6 * X["mom_12_1"] + 0.4 * X["dist_200dma"])
    return pd.Series(_sigmoid(np.asarray(raw.values, dtype=float)), index=feature_frame.index)
