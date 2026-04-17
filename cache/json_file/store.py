from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cache.cache_base import BaseQueryCacheStore, CacheEntry

logger = logging.getLogger(__name__)


class JsonFileQueryCacheStore(BaseQueryCacheStore):
    """Файловое JSON-хранилище L1-кэша для CLI MVP."""

    def __init__(self, path: str | Path, *, max_entries_per_session: int | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries_per_session = max_entries_per_session

    def get(self, session_id: str, source: str | None, query_hash: str) -> CacheEntry | None:
        data = self._read_data()
        row = data.get(self._key(session_id, source, query_hash))
        if not isinstance(row, dict):
            return None
        answer = row.get("answer")
        if not isinstance(answer, str):
            return None
        return CacheEntry(answer=answer)

    def set(
        self,
        session_id: str,
        source: str | None,
        query_hash: str,
        *,
        answer: str,
    ) -> None:
        data = self._read_data()
        data[self._key(session_id, source, query_hash)] = {
            "answer": answer,
        }
        self._trim_session_if_needed(data, session_id)
        self._write_data(data)
        logger.debug("json cache set: session_id=%s source=%s", session_id, source or "all")

    def clear_session(self, session_id: str) -> None:
        data = self._read_data()
        prefix = f"{session_id}|"
        filtered = {k: v for k, v in data.items() if not k.startswith(prefix)}
        self._write_data(filtered)
        logger.info("json cache session cleared: session_id=%s", session_id)

    def count_entries(self, session_id: str | None = None) -> int:
        data = self._read_data()
        if session_id is None:
            return len(data)
        prefix = f"{session_id}|"
        return sum(1 for key in data if key.startswith(prefix))

    def _read_data(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text(encoding="utf-8")
            if not raw.strip():
                return {}
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("json cache file has invalid root type: path=%s", self._path)
            return {}
        except json.JSONDecodeError:
            logger.warning("json cache file is corrupted, resetting in memory view: path=%s", self._path)
            return {}

    def _write_data(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _key(session_id: str, source: str | None, query_hash: str) -> str:
        return f"{session_id}|{source or '__all__'}|{query_hash}"

    def _trim_session_if_needed(self, data: dict[str, dict[str, Any]], session_id: str) -> None:
        if self._max_entries_per_session is None:
            return
        prefix = f"{session_id}|"
        session_keys = [key for key in data if key.startswith(prefix)]
        overflow = len(session_keys) - self._max_entries_per_session
        if overflow <= 0:
            return
        for key in session_keys[:overflow]:
            data.pop(key, None)
        logger.info(
            "json cache rotated: session_id=%s dropped_entries=%d limit=%d",
            session_id,
            overflow,
            self._max_entries_per_session,
        )
