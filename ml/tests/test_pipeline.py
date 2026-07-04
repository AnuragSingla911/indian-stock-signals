from iss.config import CONFIG
from iss.pipeline import run


def test_pipeline_offline_end_to_end():
    payload = run()
    assert payload["universe_size"] > 0
    assert payload["disclaimer"]
    sectors = payload["sectors"]
    assert len(sectors) == CONFIG.top_sectors

    seen_symbols = set()
    for sec in sectors:
        assert sec["sector"]
        assert sec["sector_rationale"]
        assert 0 <= sec["sector_score"] <= 100
        assert len(sec["stocks"]) == CONFIG.stocks_per_sector
        for st in sec["stocks"]:
            assert st["symbol"]
            assert st["name"]
            assert 0 <= st["composite_score"] <= 100
            assert 0.0 <= st["up_probability"] <= 1.0
            assert st["rationale"]
            assert st["chart_links"]["tradingview"].startswith("https://")
            assert st["chart_links"]["yahoo"].startswith("https://")
            assert set(st["factors"].keys()) == set(CONFIG.weights.keys())
            seen_symbols.add(st["symbol"])

    # 25 distinct picks.
    assert len(seen_symbols) == CONFIG.top_sectors * CONFIG.stocks_per_sector


def test_pipeline_stocks_sorted_by_score():
    payload = run()
    for sec in payload["sectors"]:
        scores = [s["composite_score"] for s in sec["stocks"]]
        assert scores == sorted(scores, reverse=True)
