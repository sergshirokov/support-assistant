from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseVectorStorage(ABC):
    """Контракт векторного хранилища; реализации — Qdrant, Weaviate и т.д.

    Payload — плоский словарь метаданных; для фильтрации по источнику используется
    поле ``source`` (см. ``search(..., source=...)``). Параметр ``top_k`` — число ближайших соседей.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Имя коллекции / класса / индекса в конкретном бэкенде."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Ожидаемая размерность векторов для этой коллекции."""

    @abstractmethod
    def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
        ids: Sequence[str | int] | None = None,
    ) -> list[str | int]:
        """Записать или обновить точки; порядок id соответствует входным векторам."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        source: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Векторный поиск; при ``source`` — только объекты с ``payload['source'] == source``.

        Элемент результата: ``{"id", "score", "payload"}`` — общий контракт для вызывающего кода.
        """
