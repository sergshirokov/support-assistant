from __future__ import annotations

from pathlib import Path

from cache import InMemoryQueryCacheStore, JsonFileQueryCacheStore


def test_inmemory_cache_get_set_clear() -> None:
    store = InMemoryQueryCacheStore()
    assert store.get("s1", "faq", "h1") is None
    assert store.count_entries() == 0

    store.set(
        "s1",
        "faq",
        "h1",
        answer="A1",
    )
    entry = store.get("s1", "faq", "h1")
    assert entry is not None
    assert entry.answer == "A1"
    assert store.count_entries("s1") == 1

    store.clear_session("s1")
    assert store.get("s1", "faq", "h1") is None
    assert store.count_entries() == 0


def test_json_cache_get_set_clear(tmp_path: Path) -> None:
    store = JsonFileQueryCacheStore(tmp_path / "query_cache.json")
    assert store.get("s1", None, "h1") is None
    assert store.count_entries() == 0

    store.set(
        "s1",
        None,
        "h1",
        answer="A1",
    )
    entry = store.get("s1", None, "h1")
    assert entry is not None
    assert entry.answer == "A1"
    assert store.count_entries("s1") == 1

    store.clear_session("s1")
    assert store.get("s1", None, "h1") is None
    assert store.count_entries() == 0


def test_json_cache_rotates_by_session_limit(tmp_path: Path) -> None:
    store = JsonFileQueryCacheStore(tmp_path / "query_cache.json", max_entries_per_session=2)
    store.set("s1", None, "h1", answer="A1")
    store.set("s1", None, "h2", answer="A2")
    store.set("s1", None, "h3", answer="A3")

    assert store.count_entries("s1") == 2
    assert store.get("s1", None, "h1") is None
    assert store.get("s1", None, "h3") is not None
