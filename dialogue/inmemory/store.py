from __future__ import annotations

from dialogue.history_base import BaseHistoryStore


class InMemoryHistoryStore(BaseHistoryStore):
    """In-memory хранилище истории (для MVP и unit-тестов)."""

    def __init__(self) -> None:
        self._history_by_session: dict[str, list[tuple[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        self._history_by_session.setdefault(session_id, []).append((role, content))

    def get(self, session_id: str, *, limit: int | None = None) -> list[tuple[str, str]]:
        history = self._history_by_session.get(session_id, [])
        if limit is None:
            return list(history)
        if limit <= 0:
            return []
        return list(history[-limit:])

    def clear(self, session_id: str) -> None:
        self._history_by_session.pop(session_id, None)
