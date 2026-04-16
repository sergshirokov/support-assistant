from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import Settings, get_settings
from integrations.embedder_base import BaseEmbedder
from vector_storage.vector_storage_base import BaseVectorStorage

# Явный ``score_threshold=None`` в вызове — без порога; пропуск параметра — из Settings.
_UNSET = object()


@dataclass(frozen=True)
class QueryResult:
    """Результат поиска по запросу (без генерации ответа LLM)."""

    hits: list[dict[str, Any]]
    query: str


class QueryPipeline:
    """Оркестрация: текст запроса → эмбеддинг → векторный поиск в хранилище."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        storage: BaseVectorStorage,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._storage = storage
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        score_threshold: Any = _UNSET,
    ) -> QueryResult:
        """Построить эмбеддинг для ``query`` и выполнить ``search``.

        ``top_k`` и ``score_threshold`` по умолчанию берутся из :class:`~config.settings.Settings`.
        Чтобы не применять порог score, передайте ``score_threshold=None`` явно.
        """
        text = query.strip()
        if not text:
            return QueryResult(hits=[], query=query)

        k = self._settings.search_top_k if top_k is None else top_k
        if score_threshold is _UNSET:
            st: float | None = self._settings.search_score_threshold
        else:
            st = score_threshold

        vectors = self._embedder.embed([text])
        if len(vectors) != 1:
            raise ValueError(
                f"Ожидался ровно один эмбеддинг для запроса, получено {len(vectors)}."
            )

        hits = self._storage.search(
            vectors[0],
            top_k=k,
            source=source,
            score_threshold=st,
        )
        return QueryResult(hits=hits, query=query)
