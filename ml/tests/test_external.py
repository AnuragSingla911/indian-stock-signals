"""Tests for external news/macro enrichment."""

from iss.external import MacroContext, NewsItem, StockNews, build_insights, score_sentiment


def test_score_sentiment_positive():
    assert score_sentiment("Company reports strong profit growth and record earnings") > 0


def test_score_sentiment_negative():
    assert score_sentiment("Stock falls on weak earnings miss and downgrade") < 0


def test_score_sentiment_neutral():
    assert score_sentiment("Company announces board meeting date") == 0.0


def test_build_insights_highlights_picks():
    macro = MacroContext(
        nifty_1m_return=0.02,
        usd_inr_1m_change=0.01,
        brent_1m_return=-0.03,
        summary="Test macro",
        as_of="2026-01-01T00:00:00Z",
    )
    news = {
        "AAA": StockNews(
            symbol="AAA",
            sentiment=0.5,
            article_count=2,
            headlines=[
                NewsItem("AAA beats estimates", "Wire", "", "2026-01-01", 0.5),
                NewsItem("AAA launches product", "Wire", "", "2026-01-01", 0.3),
            ],
        ),
        "BBB": StockNews(
            symbol="BBB",
            sentiment=-0.6,
            article_count=1,
            headlines=[
                NewsItem("BBB faces probe", "Wire", "", "2026-01-01", -0.6),
            ],
        ),
    }
    insights = build_insights(macro, news, ["AAA", "BBB"], max_highlights=5)
    assert "macro" in insights
    assert insights["macro"]["summary"] == "Test macro"
    assert len(insights["stock_highlights"]) == 2
    assert insights["stock_highlights"][0]["symbol"] in {"AAA", "BBB"}
