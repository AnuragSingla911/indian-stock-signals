"""Out-of-sample model evaluation via walk-forward (time-ordered) validation.

Pools historical (features, label, date, forward-return) samples across the universe, sorts
them by as-of date, then trains on the past and tests on the next block repeatedly. This gives
an honest read on how the model would have performed on unseen future data — never a fit on the
same rows it was trained on. Educational only, not investment advice.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import get_price_history
from .features import ML_FEATURES, TECH_FEATURES, technical_panel_dated
from .news_archive import get_news_archive
from .universe import load_universe

log = logging.getLogger("iss.evaluate")


@dataclass
class EvalMetrics:
    n_samples: int
    n_test: int
    n_folds: int
    horizon_days: int
    feature_set: str
    base_rate: float  # fraction of up (positive forward return) labels
    accuracy: float
    auc: float
    brier: float
    log_loss: float
    information_coefficient: float  # Spearman rank corr(pred, realized forward return)


def _collect(
    prices: dict[str, pd.DataFrame],
    horizon: int,
    step: int,
    use_news: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = get_news_archive() if use_news else None
    feature_cols = ML_FEATURES if use_news else TECH_FEATURES
    rows: list[list[float]] = []
    labels: list[int] = []
    dates: list[pd.Timestamp] = []
    fwds: list[float] = []
    for sym, df in prices.items():
        for feats, label, date, fwd in technical_panel_dated(
            df["Close"],
            horizon=horizon,
            step=step,
            symbol=sym if use_news else None,
            archive=archive,
        ):
            rows.append([float(feats.get(k, 0.0)) for k in feature_cols])
            labels.append(label)
            dates.append(date)
            fwds.append(fwd)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    fwd_arr = np.asarray(fwds, dtype=float)
    order = np.argsort(pd.DatetimeIndex(dates).values)
    return X[order], y[order], fwd_arr[order]


def _walk_forward(
    X: np.ndarray, y: np.ndarray, fwd: np.ndarray, n_folds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if n < 2 * (n_folds + 1):
        raise ValueError(f"Not enough samples ({n}) for {n_folds}-fold walk-forward evaluation")

    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    preds = np.full(n, np.nan)
    for i in range(1, n_folds):
        tr_end, te_end = bounds[i], bounds[i + 1]
        y_tr = y[:tr_end]
        if len(set(y_tr.tolist())) < 2 or te_end <= tr_end:
            continue
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_depth=3,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(X[:tr_end], y_tr)
        preds[tr_end:te_end] = clf.predict_proba(X[tr_end:te_end])[:, 1]

    mask = ~np.isnan(preds)
    return y[mask], preds[mask], fwd[mask]


def evaluate(
    n_folds: int = 5,
    horizon: int | None = None,
    step: int | None = None,
    use_news: bool = True,
    prices: dict[str, pd.DataFrame] | None = None,
) -> EvalMetrics:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    horizon = horizon or CONFIG.horizon_days
    step = step if step is not None else CONFIG.train_step
    if prices is None:
        symbols = [s.symbol for s in load_universe()]
        log.info("Fetching price history for %d symbols", len(symbols))
        prices = get_price_history(symbols)
    X, y, fwd = _collect(prices, horizon, step, use_news=use_news)

    y_t, p_t, fwd_t = _walk_forward(X, y, fwd, n_folds)
    if len(y_t) == 0:
        raise ValueError("No out-of-sample predictions were produced")

    has_both = len(set(y_t.tolist())) > 1
    ic = pd.Series(p_t).corr(pd.Series(fwd_t), method="spearman")
    feature_set = "technical+news" if use_news else "technical_only"
    return EvalMetrics(
        n_samples=len(y),
        n_test=int(len(y_t)),
        n_folds=n_folds,
        horizon_days=horizon,
        feature_set=feature_set,
        base_rate=float(y.mean()),
        accuracy=float(accuracy_score(y_t, (p_t > 0.5).astype(int))),
        auc=float(roc_auc_score(y_t, p_t)) if has_both else float("nan"),
        brier=float(brier_score_loss(y_t, p_t)),
        log_loss=float(log_loss(y_t, p_t, labels=[0, 1])),
        information_coefficient=float(ic) if pd.notna(ic) else float("nan"),
    )


def compare_models(
    n_folds: int = 5, horizon: int | None = None, step: int | None = None
) -> dict:
    """Run walk-forward eval for technical-only vs technical+news and return deltas."""
    symbols = [s.symbol for s in load_universe()]
    log.info("Fetching price history once for %d symbols", len(symbols))
    prices = get_price_history(symbols)
    baseline = evaluate(n_folds=n_folds, horizon=horizon, step=step, use_news=False, prices=prices)
    enriched = evaluate(n_folds=n_folds, horizon=horizon, step=step, use_news=True, prices=prices)
    b, e = asdict(baseline), asdict(enriched)
    delta = {
        k: round(e[k] - b[k], 4)
        for k in ("accuracy", "auc", "brier", "log_loss", "information_coefficient")
    }
    # Lower is better for brier/log_loss — flip sign so positive delta = improvement.
    delta["brier"] = round(b["brier"] - e["brier"], 4)
    delta["log_loss"] = round(b["log_loss"] - e["log_loss"], 4)
    return {
        "baseline_technical_only": b,
        "with_news": e,
        "delta_improvement": delta,
    }


def _print_metrics(m: EvalMetrics) -> None:
    print(f"  feature set:              {m.feature_set}")
    print(f"  samples (total / tested): {m.n_samples} / {m.n_test}  over {m.n_folds} folds")
    print(f"  horizon:                  {m.horizon_days} trading days")
    print(f"  base rate (P[up]):        {m.base_rate:.3f}")
    print(f"  accuracy:                 {m.accuracy:.3f}")
    print(f"  ROC AUC:                  {m.auc:.3f}")
    print(f"  Brier score (lower=better): {m.brier:.3f}")
    print(f"  log loss:                 {m.log_loss:.3f}")
    print(f"  information coefficient:  {m.information_coefficient:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation of the ISS ML model")
    parser.add_argument("--folds", type=int, default=5, help="number of time-ordered folds")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare technical-only vs technical+news models",
    )
    parser.add_argument(
        "--tech-only",
        action="store_true",
        help="evaluate using technical features only (no news)",
    )
    parser.add_argument("--out", default=None, help="optional path to write metrics JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.compare:
        result = compare_models(n_folds=args.folds)
        print("Walk-forward model comparison (out-of-sample)")
        print("\nBaseline (technical only):")
        for k, v in result["baseline_technical_only"].items():
            print(f"  {k}: {v}")
        print("\nEnriched (technical + news):")
        for k, v in result["with_news"].items():
            print(f"  {k}: {v}")
        print("\nDelta (positive = news model better):")
        for k, v in result["delta_improvement"].items():
            print(f"  {k}: {v:+.4f}")
        payload = result
    else:
        m = evaluate(n_folds=args.folds, use_news=not args.tech_only)
        print("Walk-forward evaluation (out-of-sample)")
        _print_metrics(m)
        payload = asdict(m)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
