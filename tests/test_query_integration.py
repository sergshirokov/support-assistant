"""Интеграция QueryPipeline: GigaChat + локальный Qdrant (сеть, ключ).

Запуск::

    pytest -m integration --override-ini=addopts=
"""

from __future__ import annotations

import pytest

from chunking.paragraph.chunker import ParagraphChunker
from config import get_settings
from integrations.gigachat import GigaChatEmbedder, GigaChatLangChainChatModel
from ingestion import IngestionPipeline
from query import QueryPipeline
from vector_storage import QdrantVectorStorage


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.integration
def test_ingest_then_answer_with_history(tmp_path) -> None:
    settings = get_settings()
    if not settings.gigachat_credentials:
        pytest.skip("Задайте GIGACHAT_CREDENTIALS в .env или окружении")

    embedder = GigaChatEmbedder()
    if settings.embedding_vector_size_from_api:
        embedder.embed(["warmup"])

    storage = QdrantVectorStorage(
        collection_name="query_answer_integration",
        vector_size=embedder.vector_size,
        path=str(tmp_path / "qdrant_data"),
    )
    ingest = IngestionPipeline(ParagraphChunker(), embedder, storage)
    chat_model = GigaChatLangChainChatModel(settings=settings)
    query_pipeline = QueryPipeline(embedder, storage, chat_model=chat_model, settings=settings)

    body = (
        "Чтобы сбросить пароль, откройте страницу входа и нажмите 'Забыли пароль'.\n\n"
        "Письмо с инструкцией придет на зарегистрированный email."
    )
    ingest.ingest(body, source="docs/auth.md")

    retrieved = query_pipeline.retrieve("сбросить пароль", source="docs/auth.md")
    first = query_pipeline.answer("Как сбросить пароль?", session_id="demo", source="docs/auth.md")
    second = query_pipeline.answer("Куда придет инструкция?", session_id="demo", source="docs/auth.md")

    assert retrieved.hits
    assert first.hits
    assert isinstance(first.answer, str) and first.answer.strip()
    assert isinstance(second.answer, str) and second.answer.strip()
