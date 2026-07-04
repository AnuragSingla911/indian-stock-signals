import numpy as np
import pandas as pd

from iss.config import CONFIG
from iss.scoring import (
    factor_composite,
    group_scores,
    to_0_100,
    winsorize_zscore,
)


def test_winsorize_zscore_standardizes():
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4, 100]})  # includes an outlier
    z = winsorize_zscore(frame)
    assert abs(z["a"].mean()) < 1e-6
    # Outlier winsorized -> not extreme.
    assert z["a"].max() < 3


def test_winsorize_handles_all_nan():
    frame = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    z = winsorize_zscore(frame)
    assert (z["a"] == 0).all()


def test_group_and_composite():
    zframe = pd.DataFrame(
        {
            "mom_12_1": [1.0, -1.0],
            "ret_3m": [1.0, -1.0],
            "ret_6m": [1.0, -1.0],
            "dist_50dma": [0.5, -0.5],
            "dist_200dma": [0.5, -0.5],
            "golden_cross": [1.0, 0.0],
            "vol_trend": [0.0, 0.0],
            "roe": [1.0, -1.0],
            "margin": [1.0, -1.0],
            "earnings_growth": [1.0, -1.0],
            "inv_pe": [1.0, -1.0],
            "inv_pb": [1.0, -1.0],
            "lowvol": [1.0, -1.0],
        },
        index=["good", "bad"],
    )
    groups = group_scores(zframe)
    assert set(groups.columns) == set(CONFIG.weights.keys())
    total = factor_composite(groups, CONFIG.weights)
    assert total["good"] > total["bad"]


def test_to_0_100_monotonic():
    s = pd.Series([0.1, 0.5, 0.9, -0.3], index=list("abcd"))
    out = to_0_100(s)
    assert out["c"] == out.max()
    assert out["d"] == out.min()
    assert out.between(0, 100).all()
