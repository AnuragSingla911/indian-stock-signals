from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_days"] > 0
    assert body["disclaimer"]


def test_predictions_shape():
    r = client.get("/api/predictions")
    assert r.status_code == 200
    body = r.json()
    assert len(body["sectors"]) == 5
    for sec in body["sectors"]:
        assert len(sec["stocks"]) == 5
        for st in sec["stocks"]:
            assert st["chart_links"]["tradingview"].startswith("https://")
            assert 0 <= st["composite_score"] <= 100


def test_sectors_list():
    r = client.get("/api/sectors")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 5
    assert all(s["num_stocks"] == 5 for s in body)


def test_sector_detail_and_404():
    first = client.get("/api/sectors").json()[0]["sector"]
    r = client.get(f"/api/sectors/{first}")
    assert r.status_code == 200
    assert r.json()["sector"] == first

    assert client.get("/api/sectors/NOPE").status_code == 404
