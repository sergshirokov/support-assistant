from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from chunking.chunker_base import BaseChunker
from integrations.embedder_base import BaseEmbedder
from ingestion import BasePreprocessor, IngestionPipeline, IngestionResult, PreprocessResult
from vector_storage.vector_storage_base import BaseVectorStorage


def test_ingest_calls_chunk_embed_upsert_in_order() -> None:
    chunker = create_autospec(BaseChunker, instance=True)
    chunker.chunk.return_value = ["alpha", "beta"]

    embedder = create_autospec(BaseEmbedder, instance=True)
    embedder.embed.return_value = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    storage = create_autospec(BaseVectorStorage, instance=True)
    storage.upsert.return_value = ["id-1", "id-2"]

    pipeline = IngestionPipeline(chunker, embedder, storage)
    result = pipeline.ingest("ignored", source="src-a")

    chunker.chunk.assert_called_once_with("ignored")
    embedder.embed.assert_called_once_with(["alpha", "beta"])
    storage.upsert.assert_called_once()
    call_kw = storage.upsert.call_args
    vectors, payloads = call_kw[0][0], call_kw[0][1]
    assert len(vectors) == 2
    assert payloads == [
        {"source": "src-a", "text": "alpha", "chunk_index": 0},
        {"source": "src-a", "text": "beta", "chunk_index": 1},
    ]
    assert result.point_ids == ["id-1", "id-2"]
    assert result.chunks == ["alpha", "beta"]


def test_ingest_empty_chunks_skips_embed_and_upsert() -> None:
    chunker = create_autospec(BaseChunker, instance=True)
    chunker.chunk.return_value = []

    embedder = create_autospec(BaseEmbedder, instance=True)
    storage = create_autospec(BaseVectorStorage, instance=True)

    pipeline = IngestionPipeline(chunker, embedder, storage)
    result = pipeline.ingest("   ", source="x")

    embedder.embed.assert_not_called()
    storage.upsert.assert_not_called()
    assert result == IngestionResult(point_ids=[], chunks=[])


def test_ingest_merges_extra_payload() -> None:
    chunker = MagicMock()
    chunker.chunk.return_value = ["one"]

    embedder = MagicMock()
    embedder.embed.return_value = [[0.25] * 4]

    storage = MagicMock()
    storage.upsert.return_value = [99]

    pipeline = IngestionPipeline(chunker, embedder, storage)
    pipeline.ingest("t", source="s", extra_payload={"doc_id": "d1", "chunk_index": 42})

    _, payloads = storage.upsert.call_args[0]
    assert payloads[0]["source"] == "s"
    assert payloads[0]["text"] == "one"
    assert payloads[0]["chunk_index"] == 42
    assert payloads[0]["doc_id"] == "d1"


def test_ingest_raises_when_embedding_count_mismatch() -> None:
    chunker = MagicMock()
    chunker.chunk.return_value = ["a", "b"]

    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 4]

    storage = MagicMock()

    pipeline = IngestionPipeline(chunker, embedder, storage)
    with pytest.raises(ValueError, match="Число эмбеддингов"):
        pipeline.ingest("x", source="s")

    storage.upsert.assert_not_called()


def test_ingest_calls_preprocessor_and_uses_preprocessed_source() -> None:
    chunker = MagicMock()
    chunker.chunk.return_value = ["c1"]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    storage = MagicMock()
    storage.upsert.return_value = ["id"]

    preprocessor = create_autospec(BasePreprocessor, instance=True)
    preprocessor.preprocess.return_value = PreprocessResult(
        text="content only",
        source="doc title",
        extra_payload={"file_name": "doc1.txt"},
    )

    pipeline = IngestionPipeline(
        chunker,
        embedder,
        storage,
        preprocessor=preprocessor,
    )
    pipeline.ingest("body", source=None)

    preprocessor.preprocess.assert_called_once_with(
        "body",
        source=None,
        extra_payload=None,
    )
    chunker.chunk.assert_called_once_with("content only")
    payloads = storage.upsert.call_args[0][1]
    assert payloads[0]["source"] == "doc title"
    assert payloads[0]["file_name"] == "doc1.txt"
