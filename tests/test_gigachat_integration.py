"""Интеграционные тесты с реальным API GigaChat (сеть, ключ).

Запуск только явно::

    pytest -m integration --override-ini=addopts=

Нужны ``GIGACHAT_CREDENTIALS`` и корректный ``EMBEDDING_VECTOR_SIZE`` под выбранную
``GIGACHAT_EMBEDDING_MODEL`` (иначе embedder вернёт ошибку размерности).
"""

from __future__ import annotations

import math

import pytest

from config import get_settings
from integrations.gigachat.embedder import GigaChatEmbedder


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.integration
def test_gigachat_embed_single_text_real() -> None:
    if not get_settings().gigachat_credentials:
        pytest.skip("Задайте GIGACHAT_CREDENTIALS в .env или окружении")

    embedder = GigaChatEmbedder()
    vectors = embedder.embed(["Проверка эмбеддинга GigaChat."])

    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.vector_size
    assert all(math.isfinite(x) for x in vectors[0])


@pytest.mark.integration
def test_gigachat_embed_batch_real() -> None:
    if not get_settings().gigachat_credentials:
        pytest.skip("Задайте GIGACHAT_CREDENTIALS в .env или окружении")

    embedder = GigaChatEmbedder()
    vectors = embedder.embed(["Первый текст.", "Второй текст."])

    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == embedder.vector_size
        assert all(math.isfinite(x) for x in vec)
