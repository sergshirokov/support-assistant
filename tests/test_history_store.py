from __future__ import annotations

from dialogue import InMemoryHistoryStore


def test_inmemory_history_append_get_clear() -> None:
    store = InMemoryHistoryStore()
    store.append("s1", "user", "u1")
    store.append("s1", "assistant", "a1")

    assert store.get("s1") == [("user", "u1"), ("assistant", "a1")]
    assert store.get("s1", limit=1) == [("assistant", "a1")]

    store.clear("s1")
    assert store.get("s1") == []
