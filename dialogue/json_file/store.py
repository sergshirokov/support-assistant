from __future__ import annotations

import json
import logging
from pathlib import Path

from dialogue.history_base import BaseHistoryStore

logger = logging.getLogger(__name__)


class JsonFileHistoryStore(BaseHistoryStore):
    """Файловое JSON-хранилище истории по session_id для CLI/dev."""

    def __init__(self, path: str | Path, *, max_messages_per_session: int | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_messages_per_session = max_messages_per_session

    def append(self, session_id: str, role: str, content: str) -> None:
        data = self._read_data()
        session = data.setdefault(session_id, [])
        session.append([role, content])
        if self._max_messages_per_session is not None and self._max_messages_per_session >= 0:
            overflow = len(session) - self._max_messages_per_session
            if overflow > 0:
                del session[:overflow]
                logger.info(
                    "json history rotated: session_id=%s dropped_messages=%d limit=%d",
                    session_id,
                    overflow,
                    self._max_messages_per_session,
                )
        self._write_data(data)
        logger.debug("json history append: session_id=%s role=%s", session_id, role)

    def get(self, session_id: str, *, limit: int | None = None) -> list[tuple[str, str]]:
        data = self._read_data()
        raw_items = data.get(session_id, [])
        history: list[tuple[str, str]] = [
            (str(item[0]), str(item[1])) for item in raw_items if isinstance(item, list) and len(item) == 2
        ]
        if limit is None:
            return history
        if limit <= 0:
            return []
        return history[-limit:]

    def clear(self, session_id: str) -> None:
        data = self._read_data()
        data.pop(session_id, None)
        self._write_data(data)
        logger.info("json history session cleared: session_id=%s", session_id)

    def _read_data(self) -> dict[str, list[list[str]]]:
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text(encoding="utf-8")
            if not raw.strip():
                return {}
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("json history file has invalid root type: path=%s", self._path)
            return {}
        except json.JSONDecodeError:
            logger.warning("json history file is corrupted, resetting in memory view: path=%s", self._path)
            return {}

    def _write_data(self, data: dict[str, list[list[str]]]) -> None:
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
