import numpy as np
import pandas as pd

from iss.features import build_feature_frame, rsi, technical_features, technical_panel


def _series(values):
    idx = pd.bdate_range(end="2024-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_rsi_bounds_and_uptrend():
    up = _series(np.linspace(100, 200, 300))
    assert 0 <= rsi(up) <= 100
    # Strong, uninterrupted uptrend -> high RSI.
    assert rsi(up) > 70


def test_technical_features_keys_and_momentum_sign():
    prices = _series(np.linspace(100, 300, 400))  # steady uptrend
    feats = technical_features(prices)
    for key in ["mom_12_1", "ret_3m", "ret_6m", "dist_200dma", "rsi14", "lowvol"]:
        assert key in feats
    assert feats["ret_3m"] > 0
    assert feats["dist_200dma"] > 0  # price above its long-term average
    assert feats["golden_cross"] == 1.0


def test_downtrend_negative_momentum():
    prices = _series(np.linspace(300, 100, 400))
    feats = technical_features(prices)
    assert feats["ret_3m"] < 0
    assert feats["dist_200dma"] < 0


def test_technical_panel_produces_labeled_samples():
    rng = np.random.default_rng(0)
    prices = _series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 500))))
    samples = technical_panel(prices, horizon=21, step=21)
    assert len(samples) > 0
    feats, label = samples[0]
    assert label in (0, 1)
    assert "mom_12_1" in feats


def test_build_feature_frame_shape():
    prices = {
        "AAA": pd.DataFrame({"Close": np.linspace(10, 20, 300), "Volume": np.arange(300) + 1}),
        "BBB": pd.DataFrame({"Close": np.linspace(20, 10, 300), "Volume": np.arange(300) + 1}),
    }
    funds = {
        "AAA": {"pe": 20, "pb": 3, "roe": 0.2, "margin": 0.1, "earnings_growth": 0.15},
        "BBB": {},
    }
    frame = build_feature_frame(prices, funds)
    assert set(frame.index) == {"AAA", "BBB"}
    assert "inv_pe" in frame.columns
    assert frame.loc["AAA", "inv_pe"] == 1 / 20
