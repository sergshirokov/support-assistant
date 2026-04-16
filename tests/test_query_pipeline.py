from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from config import get_settings
from integrations.embedder_base import BaseEmbedder
from query import QueryPipeline, QueryResult
from vector_storage.vector_storage_base import BaseVectorStorage


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_retrieve_uses_top_k_and_score_threshold_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARCH_TOP_K", "3")
    monkeypatch.setenv("SEARCH_SCORE_THRESHOLD", "0.5")
    get_settings.cache_clear()

    embedder = create_autospec(BaseEmbedder, instance=True)
    embedder.embed.return_value = [[0.0, 0.0, 1.0, 0.0]]

    storage = create_autospec(BaseVectorStorage, instance=True)
    storage.search.return_value = [
        {"id": 1, "score": 0.9, "payload": {"text": "a", "source": "s"}},
    ]

    pipeline = QueryPipeline(embedder, storage)
    result = pipeline.retrieve("  что такое тест?  ", source="s")

    embedder.embed.assert_called_once_with(["что такое тест?"])
    storage.search.assert_called_once_with(
        [0.0, 0.0, 1.0, 0.0],
        top_k=3,
        source="s",
        score_threshold=0.5,
    )
    assert result.query == "  что такое тест?  "
    assert result.hits == storage.search.return_value


def test_retrieve_explicit_score_threshold_none_disables_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARCH_SCORE_THRESHOLD", "0.5")
    get_settings.cache_clear()

    embedder = create_autospec(BaseEmbedder, instance=True)
    embedder.embed.return_value = [[0.0, 0.0, 1.0, 0.0]]
    storage = create_autospec(BaseVectorStorage, instance=True)
    storage.search.return_value = []

    pipeline = QueryPipeline(embedder, storage)
    pipeline.retrieve("x", score_threshold=None)

    assert storage.search.call_args.kwargs["score_threshold"] is None


def test_retrieve_empty_or_whitespace_no_embed_no_search() -> None:
    embedder = create_autospec(BaseEmbedder, instance=True)
    storage = create_autospec(BaseVectorStorage, instance=True)

    pipeline = QueryPipeline(embedder, storage)
    assert pipeline.retrieve("") == QueryResult(hits=[], query="")
    assert pipeline.retrieve("   \t\n") == QueryResult(hits=[], query="   \t\n")

    embedder.embed.assert_not_called()
    storage.search.assert_not_called()


def test_retrieve_raises_when_not_single_embedding() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 4, [1.0] * 4]
    storage = MagicMock()

    pipeline = QueryPipeline(embedder, storage)
    with pytest.raises(ValueError, match="ровно один эмбеддинг"):
        pipeline.retrieve("x")

    storage.search.assert_not_called()
