import pytest
from api.routers import stocks, stock_rankings, stock_portfolio, smid_growth_portfolio


@pytest.mark.parametrize("module", [stock_rankings, stock_portfolio, smid_growth_portfolio])
def test_caches_release_expired_unrequested_payloads_and_bound_capacity(monkeypatch, module):
    monkeypatch.setattr(module, "_CACHE", {})
    monkeypatch.setattr(module, "MAX_CACHE_ENTRIES", 2)
    monkeypatch.setattr(module.time, "time", lambda: 100)
    def put(key, value):
        if module is smid_growth_portfolio:
            return module._cache_set(key, value)
        return module._cache_set(key, value, module.CACHE_TTL_SECONDS)
    payload = {"holdings": [{"ticker": "AIR", "weight": 0.1}], "gain": 0.12}
    put("a", payload)
    put("b", {"other": True})
    assert module._cache_get("a") is payload
    put("c", {})
    assert module._cache_get("b") is None
    assert module._cache_get("a") == payload
    monkeypatch.setattr(module.time, "time", lambda: 100 + module.CACHE_TTL_SECONDS)
    put("fresh", {})
    assert set(module._CACHE) == {"fresh"}


def test_stock_cache_retains_explicit_ttls_and_evicts_lru(monkeypatch):
    monkeypatch.setattr(stocks, "_cache", {})
    monkeypatch.setattr(stocks, "MAX_CACHE_ENTRIES", 2)
    monkeypatch.setattr(stocks.time, "time", lambda: 100)
    stocks._set("history", [1, 2], ttl=900)
    stocks._set("search", ["AIR"], ttl=86400)
    assert stocks._get("history", ttl=900) == [1, 2]
    stocks._set("new", 42, ttl=900)
    assert stocks._get("search", ttl=86400) is None
    stocks._set("search", ["AIR"], ttl=86400)
    monkeypatch.setattr(stocks.time, "time", lambda: 1000)
    assert stocks._get("search", ttl=86400) == ["AIR"]
    assert set(stocks._cache) == {"search"}
    monkeypatch.setattr(stocks.time, "time", lambda: 86500)
    assert stocks._get("search", ttl=86400) is None
    assert not stocks._cache
