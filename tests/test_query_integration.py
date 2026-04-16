"""Интеграция QueryPipeline: GigaChat + локальный Qdrant (сеть, ключ).

Запуск::

    pytest -m integration --override-ini=addopts=
"""

from __future__ import annotations

import pytest

from chunking.paragraph.chunker import ParagraphChunker
from config import get_settings
from integrations.gigachat.embedder import GigaChatEmbedder
from ingestion import IngestionPipeline
from query import QueryPipeline
from vector_storage import QdrantVectorStorage


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.integration
def test_ingest_then_retrieve_same_source(tmp_path) -> None:
    settings = get_settings()
    if not settings.gigachat_credentials:
        pytest.skip("Задайте GIGACHAT_CREDENTIALS в .env или окружении")

    embedder = GigaChatEmbedder()
    if settings.embedding_vector_size_from_api:
        embedder.embed(["warmup"])

    storage = QdrantVectorStorage(
        collection_name="query_integration",
        vector_size=embedder.vector_size,
        path=str(tmp_path / "qdrant_data"),
    )
    ingest = IngestionPipeline(ParagraphChunker(), embedder, storage)
    retrieve = QueryPipeline(embedder, storage)

    body = "Ответ на вопрос о погоде: солнечно.\n\nВторой блок про дождь."
    source = "docs/weather.md"
    ingest.ingest(body, source=source)

    result = retrieve.retrieve("погода солнечно", source=source)
    assert result.hits
    texts = {h["payload"].get("text", "") for h in result.hits}
    assert any("погода" in t or "солнечно" in t for t in texts)
