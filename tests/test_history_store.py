from __future__ import annotations

from pathlib import Path

from dialogue import InMemoryHistoryStore, JsonFileHistoryStore


def test_inmemory_history_append_get_clear() -> None:
    store = InMemoryHistoryStore()
    store.append("s1", "user", "u1")
    store.append("s1", "assistant", "a1")

    assert store.get("s1") == [("user", "u1"), ("assistant", "a1")]
    assert store.get("s1", limit=1) == [("assistant", "a1")]

    store.clear("s1")
    assert store.get("s1") == []


def test_json_history_append_get_clear(tmp_path: Path) -> None:
    store = JsonFileHistoryStore(tmp_path / "history.json")
    store.append("s1", "user", "u1")
    store.append("s1", "assistant", "a1")

    assert store.get("s1") == [("user", "u1"), ("assistant", "a1")]
    assert store.get("s1", limit=1) == [("assistant", "a1")]

    store.clear("s1")
    assert store.get("s1") == []


def test_json_history_rotates_by_max_messages(tmp_path: Path) -> None:
    store = JsonFileHistoryStore(tmp_path / "history.json", max_messages_per_session=2)
    store.append("s1", "user", "u1")
    store.append("s1", "assistant", "a1")
    store.append("s1", "user", "u2")

    assert store.get("s1") == [("assistant", "a1"), ("user", "u2")]
