from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreprocessResult:
    """Результат предобработки документа перед ingestion."""

    text: str
    source: str
    extra_payload: dict[str, Any]


class BasePreprocessor(ABC):
    """Контракт предобработчика документов для ingestion-пайплайна."""

    @abstractmethod
    def preprocess(
        self,
        text: str,
        *,
        source: str | None,
        extra_payload: dict[str, Any] | None,
    ) -> PreprocessResult:
        """Подготовить source и дополнительные metadata для последующей записи в storage."""


class PassThroughPreprocessor(BasePreprocessor):
    """Предобработчик по умолчанию: использует переданный source и payload без изменений."""

    def preprocess(
        self,
        text: str,
        *,
        source: str | None,
        extra_payload: dict[str, Any] | None,
    ) -> PreprocessResult:
        if source is None or not source.strip():
            raise ValueError("source обязателен, если не задан кастомный preprocessor")
        return PreprocessResult(
            text=text,
            source=source,
            extra_payload=dict(extra_payload or {}),
        )
