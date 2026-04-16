from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integrations.embedder_base import BaseEmbedder
from integrations.gigachat.embedder import GigaChatEmbedder


def test_base_embedder_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseEmbedder()  # type: ignore[misc]


def test_gigachat_embedder_vector_size_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE_FROM_API", "false")
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "768")
    emb = GigaChatEmbedder()
    assert emb.vector_size == 768
    get_settings.cache_clear()


def test_gigachat_embedder_vector_size_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE_FROM_API", "false")
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "768")
    emb = GigaChatEmbedder(vector_size=512)
    assert emb.vector_size == 512
    get_settings.cache_clear()


def test_gigachat_embed_empty_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    emb = GigaChatEmbedder()
    assert emb.embed([]) == []
    get_settings.cache_clear()


def test_gigachat_embed_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")
    emb = GigaChatEmbedder()
    with pytest.raises(ValueError, match="gigachat_credentials"):
        emb.embed(["a"])
    get_settings.cache_clear()


def test_gigachat_embed_with_mock_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "4")

    mock_client = MagicMock()
    item0 = MagicMock()
    item0.embedding = [0.1, 0.2, 0.3, 0.4]
    item0.index = 1
    item1 = MagicMock()
    item1.embedding = [0.5, 0.6, 0.7, 0.8]
    item1.index = 0
    mock_resp = MagicMock()
    mock_resp.data = [item0, item1]
    mock_client.embeddings.return_value = mock_resp

    emb = GigaChatEmbedder(client=mock_client, vector_size=4)
    out = emb.embed(["b", "a"])

    assert len(out) == 2
    assert out[0] == [0.5, 0.6, 0.7, 0.8]
    assert out[1] == [0.1, 0.2, 0.3, 0.4]
    mock_client.embeddings.assert_called_once()
    get_settings.cache_clear()


def test_gigachat_embed_wrong_dimension_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "2")

    mock_client = MagicMock()
    item = MagicMock()
    item.embedding = [0.1, 0.2, 0.3]
    item.index = 0
    mock_resp = MagicMock()
    mock_resp.data = [item]
    mock_client.embeddings.return_value = mock_resp

    emb = GigaChatEmbedder(client=mock_client, vector_size=2)
    with pytest.raises(ValueError, match="переданным vector_size"):
        emb.embed(["x"])
    get_settings.cache_clear()


def test_vector_size_from_api_inferred_after_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE_FROM_API", "true")

    mock_client = MagicMock()
    item = MagicMock()
    item.embedding = [0.0] * 3
    item.index = 0
    mock_resp = MagicMock()
    mock_resp.data = [item]
    mock_client.embeddings.return_value = mock_resp

    emb = GigaChatEmbedder(client=mock_client)
    with pytest.raises(RuntimeError, match="первого вызова embed"):
        _ = emb.vector_size

    emb.embed(["x"])
    assert emb.vector_size == 3
    get_settings.cache_clear()
