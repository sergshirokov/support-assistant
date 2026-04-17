from __future__ import annotations

import logging

from dialogue.history_base import BaseHistoryStore

logger = logging.getLogger(__name__)


class InMemoryHistoryStore(BaseHistoryStore):
    """In-memory хранилище истории (для MVP и unit-тестов)."""

    def __init__(self, *, max_messages_per_session: int | None = None) -> None:
        self._history_by_session: dict[str, list[tuple[str, str]]] = {}
        self._max_messages_per_session = max_messages_per_session

    def append(self, session_id: str, role: str, content: str) -> None:
        history = self._history_by_session.setdefault(session_id, [])
        history.append((role, content))
        if self._max_messages_per_session is not None and self._max_messages_per_session >= 0:
            overflow = len(history) - self._max_messages_per_session
            if overflow > 0:
                del history[:overflow]
                logger.info(
                    "history rotated: session_id=%s dropped_messages=%d limit=%d",
                    session_id,
                    overflow,
                    self._max_messages_per_session,
                )
        logger.debug("history append: session_id=%s role=%s content_len=%d", session_id, role, len(content))

    def get(self, session_id: str, *, limit: int | None = None) -> list[tuple[str, str]]:
        history = self._history_by_session.get(session_id, [])
        if limit is None:
            return list(history)
        if limit <= 0:
            return []
        return list(history[-limit:])

    def clear(self, session_id: str) -> None:
        self._history_by_session.pop(session_id, None)
        logger.info("history session cleared: session_id=%s", session_id)
