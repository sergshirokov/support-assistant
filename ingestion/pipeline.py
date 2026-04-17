from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from chunking.chunker_base import BaseChunker
from integrations.embedder_base import BaseEmbedder
from ingestion.preprocessor_base import BasePreprocessor, PassThroughPreprocessor
from vector_storage.vector_storage_base import BaseVectorStorage

logger = logging.getLogger(__name__)


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
        logger.info(
            "ingestion started: text_len=%d source_provided=%s extra_payload_keys=%s",
            len(text),
            bool(source and source.strip()),
            sorted((extra_payload or {}).keys()),
        )
        prepared = self._preprocessor.preprocess(
            text,
            source=source,
            extra_payload=extra_payload,
        )
        logger.info(
            "preprocess finished: source=%s prepared_text_len=%d",
            prepared.source,
            len(prepared.text),
        )

        chunks = self._chunker.chunk(prepared.text)
        logger.info("chunking finished: chunks=%d source=%s", len(chunks), prepared.source)
        if not chunks:
            logger.warning("ingestion skipped: no chunks produced source=%s", prepared.source)
            return IngestionResult(point_ids=[], chunks=[])

        vectors = self._embedder.embed(chunks)
        if len(vectors) != len(chunks):
            logger.error(
                "embedding/chunk mismatch: embeddings=%d chunks=%d source=%s",
                len(vectors),
                len(chunks),
                prepared.source,
            )
            raise ValueError(
                f"Число эмбеддингов ({len(vectors)}) не совпадает с числом чанков ({len(chunks)})."
            )
        logger.info("embedding finished: embeddings=%d source=%s", len(vectors), prepared.source)

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
        logger.info(
            "ingestion finished: points=%d chunks=%d source=%s",
            len(point_ids),
            len(chunks),
            prepared.source,
        )
        return IngestionResult(point_ids=point_ids, chunks=chunks)
