from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseChatModel(ABC):
    """Базовый интерфейс chat-модели для генерации ответа по сообщениям."""

    @abstractmethod
    def generate(self, messages: Sequence[Any]) -> Any:
        """Сгенерировать ответ по последовательности сообщений (формат сообщений задаёт интеграция)."""
