from __future__ import annotations

from abc import ABC, abstractmethod


class BaseHistoryStore(ABC):
    """Контракт хранилища истории диалога по session_id."""

    @abstractmethod
    def append(self, session_id: str, role: str, content: str) -> None:
        """Добавить сообщение в историю сессии."""

    @abstractmethod
    def get(self, session_id: str, *, limit: int | None = None) -> list[tuple[str, str]]:
        """Получить историю (опционально ограниченную последними ``limit`` сообщениями)."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Очистить историю сессии."""
