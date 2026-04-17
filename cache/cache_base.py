from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheEntry:
    """Кэшированный ответ query-пайплайна."""

    answer: str


class BaseQueryCacheStore(ABC):
    """Контракт L1-кэша ответов по (session_id, source, query_hash)."""

    @abstractmethod
    def get(self, session_id: str, source: str | None, query_hash: str) -> CacheEntry | None:
        """Получить кэшированный ответ, если есть."""

    @abstractmethod
    def set(
        self,
        session_id: str,
        source: str | None,
        query_hash: str,
        *,
        answer: str,
    ) -> None:
        """Сохранить кэшированный ответ."""

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Очистить кэш конкретной сессии."""

    @abstractmethod
    def count_entries(self, session_id: str | None = None) -> int:
        """Получить количество кэш-записей (всех или по сессии)."""
