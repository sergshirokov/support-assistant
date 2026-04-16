from __future__ import annotations

from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Базовый класс для разбиения текста на чанки."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Разбить текст на фрагменты (порядок сохраняется)."""
