"""Tests for historical news archive."""

import json

import numpy as np
import pandas as pd

from iss.news_archive import (
    ArchivedArticle,
    NewsArchive,
    bootstrap_synthetic_archive,
)
from iss.sentiment import score_sentiment


def test_score_sentiment_positive_negative():
    assert score_sentiment("Company reports strong profit growth") > 0
    assert score_sentiment("Stock falls on weak earnings miss") < 0


def test_sentiment_on_date_no_look_ahead(tmp_path):
    archive = NewsArchive(root=tmp_path)
    articles = [
        ArchivedArticle("good news beat estimates", "2024-01-10", "Wire", "", 0.5),
        ArchivedArticle("future headline miss", "2024-02-01", "Wire", "", -1.0),
    ]
    archive.save("AAA", articles)

    close = pd.Series(
        np.linspace(100, 110, 40),
        index=pd.bdate_range("2024-01-01", periods=40),
    )
    sent, vol = archive.sentiment_on_date("AAA", pd.Timestamp("2024-01-15"), close=close)
    assert sent == 0.5
    assert vol > 0


def test_bootstrap_synthetic_archive_is_deterministic():
    close = pd.Series(
        np.linspace(50, 150, 400),
        index=pd.bdate_range("2020-01-01", periods=400),
    )
    a = bootstrap_synthetic_archive("XYZ", close, step=21)
    b = bootstrap_synthetic_archive("XYZ", close, step=21)
    assert len(a) > 0
    assert [x.title for x in a] == [x.title for x in b]


def test_merge_dedupes(tmp_path):
    archive = NewsArchive(root=tmp_path)
    first = [ArchivedArticle("headline", "2024-01-01", "Wire", "", 0.2)]
    archive.merge("BBB", first)
    archive.merge("BBB", first)
    assert len(archive.load("BBB")) == 1
    path = tmp_path / "BBB.json"
    raw = json.loads(path.read_text())
    assert raw["symbol"] == "BBB"
