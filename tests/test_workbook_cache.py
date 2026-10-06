from api.routers import stock_intelligence as module


def test_cache_prunes_unrequested_expired_entries(monkeypatch):
    monkeypatch.setattr(module, "_CACHE", {})
    monkeypatch.setattr(module.time, "time", lambda: 100)
    module._cache_set("expired", {"large": "payload"}, 5)
    module._cache_set("fresh", 42, 50)
    monkeypatch.setattr(module.time, "time", lambda: 106)
    assert module._cache_get("fresh") == 42
    assert "expired" not in module._CACHE


def test_cache_capacity_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(module, "_CACHE", {})
    monkeypatch.setattr(module, "MAX_CACHE_ENTRIES", 2)
    module._cache_set("a", 1, 60)
    module._cache_set("b", 2, 60)
    assert module._cache_get("a") == 1
    module._cache_set("c", 3, 60)
    assert module._cache_get("b") is None
    assert module._cache_get("a") == 1
    assert module._cache_get("c") == 3


def test_replacing_entry_does_not_evict_another(monkeypatch):
    monkeypatch.setattr(module, "_CACHE", {})
    monkeypatch.setattr(module, "MAX_CACHE_ENTRIES", 2)
    module._cache_set("a", 1, 60)
    module._cache_set("b", 2, 60)
    module._cache_set("a", 3, 60)
    assert module._cache_get("a") == 3
    assert module._cache_get("b") == 2
