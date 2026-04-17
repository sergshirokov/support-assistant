from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunking.chunker_base import BaseChunker
from integrations.embedder_base import BaseEmbedder
from ingestion.preprocessor_base import BasePreprocessor, PassThroughPreprocessor
from vector_storage.vector_storage_base import BaseVectorStorage


@dataclass(frozen=True)
class IngestionResult:
    """Результат одной операции индексации."""

    point_ids: list[str | int]
    chunks: list[str]


class IngestionPipeline:
    """Оркестрация: текст → чанки → эмбеддинги → upsert в векторное хранилище."""

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        storage: BaseVectorStorage,
        *,
        preprocessor: BasePreprocessor | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._storage = storage
        self._preprocessor = preprocessor or PassThroughPreprocessor()

    def ingest(
        self,
        text: str,
        source: str | None = None,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Разбить ``text`` на чанки, получить векторы и записать точки с ``payload['source']``."""
        prepared = self._preprocessor.preprocess(
            text,
            source=source,
            extra_payload=extra_payload,
        )

        chunks = self._chunker.chunk(prepared.text)
        if not chunks:
            return IngestionResult(point_ids=[], chunks=[])

        vectors = self._embedder.embed(chunks)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Число эмбеддингов ({len(vectors)}) не совпадает с числом чанков ({len(chunks)})."
            )

        extra = prepared.extra_payload
        payloads: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            pl: dict[str, Any] = {
                "source": prepared.source,
                "text": chunk,
                "chunk_index": i,
            }
            pl.update(extra)
            payloads.append(pl)

        point_ids = self._storage.upsert(vectors, payloads)
        return IngestionResult(point_ids=point_ids, chunks=chunks)
