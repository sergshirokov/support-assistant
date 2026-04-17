from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Настройки приложения: переменные окружения и при необходимости файл `.env` в корне проекта."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    qdrant_url: str | None = None
    qdrant_path: str | None = None
    qdrant_collection_name: str = "support_assistant"
    embedding_vector_size: int = 1024
    #: Если True — не сравниваем с ``embedding_vector_size``; после первого ``embed()`` размерность берётся из ответа API.
    embedding_vector_size_from_api: bool = False

    gigachat_credentials: str | None = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_embedding_model: str = "Embeddings"
    gigachat_chat_model: str = "GigaChat-2-Max"
    gigachat_verify_ssl_certs: bool = True
    gigachat_timeout: float = 60.0
    llm_temperature: float = 0.2
    llm_max_tokens: int | None = None
    dialog_history_limit: int = 10
    cache_session_limit: int = 50
    rag_system_prompt: str = (
        "Ты помощник службы поддержки. Отвечай по контексту из базы знаний. "
        "Если контекста недостаточно, так и скажи."
    )

    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_min_size: int = 50
    ingestion_data_dir: str = "data"

    #: Число ближайших соседей при векторном поиске (RAG retrieval).
    search_top_k: int = 5
    #: Минимальный score для Qdrant; ``None`` — порог не задаётся.
    search_score_threshold: float | None = None
    log_level: str = "INFO"
    log_format: str = "text"

    @field_validator("qdrant_url", "qdrant_path", "gigachat_credentials", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("search_score_threshold", mode="before")
    @classmethod
    def empty_str_search_score_threshold_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("search_top_k")
    @classmethod
    def search_top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("search_top_k должен быть >= 1")
        return v

    @field_validator("dialog_history_limit")
    @classmethod
    def dialog_history_limit_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dialog_history_limit должен быть >= 0")
        return v

    @field_validator("cache_session_limit")
    @classmethod
    def cache_session_limit_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("cache_session_limit должен быть >= 1")
        return v

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        value = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in allowed:
            raise ValueError(f"log_level должен быть одним из: {', '.join(sorted(allowed))}")
        return value

    @field_validator("log_format")
    @classmethod
    def normalize_log_format(cls, v: str) -> str:
        value = v.strip().lower()
        allowed = {"text", "json"}
        if value not in allowed:
            raise ValueError(f"log_format должен быть одним из: {', '.join(sorted(allowed))}")
        return value

    @model_validator(mode="after")
    def chunk_limits_consistent(self) -> Settings:
        if self.chunk_size < 1:
            raise ValueError("chunk_size должен быть >= 1")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть >= 0 и строго меньше chunk_size")
        if self.chunk_min_size < 1:
            raise ValueError("chunk_min_size должен быть >= 1")
        if self.chunk_min_size > self.chunk_size:
            raise ValueError("chunk_min_size не должен превышать chunk_size")
        return self

    def adaptive_chunker_kwargs(self) -> dict[str, Any]:
        """Аргументы для :class:`chunking.adaptive.AdaptiveChunker`."""
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.chunk_overlap,
            "min_chunk_size": self.chunk_min_size,
        }


@lru_cache
def get_settings() -> Settings:
    """Единый экземпляр настроек; в тестах вызывайте `get_settings.cache_clear()`."""
    return Settings()
