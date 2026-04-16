from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class BaseEmbedder(ABC):
    """Базовый класс для провайдеров эмбеддингов (GigaChat, OpenAI и т.д.)."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Размерность вектора для коллекции Qdrant и проверок."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Строит эмбеддинги для списка текстов в том же порядке."""
