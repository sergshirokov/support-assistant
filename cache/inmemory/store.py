from __future__ import annotations

import logging

from cache.cache_base import BaseQueryCacheStore, CacheEntry

logger = logging.getLogger(__name__)


class InMemoryQueryCacheStore(BaseQueryCacheStore):
    """In-memory L1-кэш для unit-тестов и простых сценариев."""

    def __init__(self, *, max_entries_per_session: int | None = None) -> None:
        self._data: dict[str, CacheEntry] = {}
        self._max_entries_per_session = max_entries_per_session

    def get(self, session_id: str, source: str | None, query_hash: str) -> CacheEntry | None:
        return self._data.get(self._key(session_id, source, query_hash))

    def set(
        self,
        session_id: str,
        source: str | None,
        query_hash: str,
        *,
        answer: str,
    ) -> None:
        self._data[self._key(session_id, source, query_hash)] = CacheEntry(
            answer=answer,
        )
        self._trim_session_if_needed(session_id)
        logger.debug("cache set: session_id=%s source=%s", session_id, source or "all")

    def clear_session(self, session_id: str) -> None:
        prefix = f"{session_id}|"
        self._data = {k: v for k, v in self._data.items() if not k.startswith(prefix)}
        logger.info("cache session cleared: session_id=%s", session_id)

    def count_entries(self, session_id: str | None = None) -> int:
        if session_id is None:
            return len(self._data)
        prefix = f"{session_id}|"
        return sum(1 for key in self._data if key.startswith(prefix))

    @staticmethod
    def _key(session_id: str, source: str | None, query_hash: str) -> str:
        return f"{session_id}|{source or '__all__'}|{query_hash}"

    def _trim_session_if_needed(self, session_id: str) -> None:
        if self._max_entries_per_session is None:
            return
        prefix = f"{session_id}|"
        session_keys = [key for key in self._data if key.startswith(prefix)]
        overflow = len(session_keys) - self._max_entries_per_session
        if overflow <= 0:
            return
        for key in session_keys[:overflow]:
            self._data.pop(key, None)
        logger.info(
            "cache rotated: session_id=%s dropped_entries=%d limit=%d",
            session_id,
            overflow,
            self._max_entries_per_session,
        )
