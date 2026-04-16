"""Интеграция IngestionPipeline: GigaChat + локальный Qdrant (сеть, ключ).

Запуск::

    pytest -m integration --override-ini=addopts=
"""

from __future__ import annotations

import pytest

from chunking.paragraph.chunker import ParagraphChunker
from config import get_settings
from integrations.gigachat.embedder import GigaChatEmbedder
from ingestion import IngestionPipeline
from vector_storage import QdrantVectorStorage


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.integration
def test_ingest_then_search_by_source(tmp_path) -> None:
    settings = get_settings()
    if not settings.gigachat_credentials:
        pytest.skip("Задайте GIGACHAT_CREDENTIALS в .env или окружении")

    embedder = GigaChatEmbedder()
    if settings.embedding_vector_size_from_api:
        embedder.embed(["warmup"])

    storage = QdrantVectorStorage(
        collection_name="ingestion_integration",
        vector_size=embedder.vector_size,
        path=str(tmp_path / "qdrant_data"),
    )
    pipeline = IngestionPipeline(ParagraphChunker(), embedder, storage)

    body = "Уникальная фраза для поиска альфа.\n\nВторой абзац бета."
    source = "wiki/page1"
    result = pipeline.ingest(body, source=source)

    assert len(result.chunks) == 2
    assert len(result.point_ids) == 2

    query_vec = embedder.embed(["Уникальная фраза для поиска"])[0]
    hits = storage.search(query_vec, top_k=5, source=source)
    assert hits
    texts = {h["payload"].get("text", "") for h in hits}
    assert any("Уникальная фраза" in t for t in texts)
