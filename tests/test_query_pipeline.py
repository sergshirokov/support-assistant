from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from config import get_settings
from dialogue import BaseHistoryStore
from integrations.chat_model_base import BaseChatModel
from integrations.embedder_base import BaseEmbedder
from query import BasePromptBuilder
from query import QueryAnswerResult, QueryPipeline, QueryResult
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


def test_answer_uses_retrieve_and_generates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIALOG_HISTORY_LIMIT", "2")
    get_settings.cache_clear()

    embedder = create_autospec(BaseEmbedder, instance=True)
    embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

    storage = create_autospec(BaseVectorStorage, instance=True)
    storage.search.return_value = [
        {"id": "1", "score": 0.9, "payload": {"source": "faq", "text": "Ответ из базы"}},
    ]

    llm = create_autospec(BaseChatModel, instance=True)
    llm_response = MagicMock()
    llm_response.content = "Сгенерированный ответ"
    llm.generate.return_value = llm_response

    pipeline = QueryPipeline(embedder, storage, chat_model=llm)
    out = pipeline.answer("Как сбросить пароль?", session_id="s1")

    assert out == QueryAnswerResult(
        query="Как сбросить пароль?",
        answer="Сгенерированный ответ",
        hits=storage.search.return_value,
        session_id="s1",
    )
    llm.generate.assert_called_once()


def test_answer_requires_chat_model() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0, 0.0, 0.0, 0.0]]
    storage = MagicMock()
    storage.search.return_value = []

    pipeline = QueryPipeline(embedder, storage)
    with pytest.raises(ValueError, match="не задан chat_model"):
        pipeline.answer("x", session_id="s")


def test_answer_keeps_history_per_session() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0, 0.0, 0.0, 0.0]]
    storage = MagicMock()
    storage.search.return_value = [{"id": "1", "score": 1.0, "payload": {"text": "ctx"}}]

    llm = MagicMock()
    resp1 = MagicMock()
    resp1.content = "A1"
    resp2 = MagicMock()
    resp2.content = "A2"
    llm.generate.side_effect = [resp1, resp2]

    history_store = create_autospec(BaseHistoryStore, instance=True)
    history_store.get.side_effect = [[], [("user", "Q1"), ("assistant", "A1")]]
    pipeline = QueryPipeline(
        embedder,
        storage,
        chat_model=llm,
        history_store=history_store,
    )
    pipeline.answer("Q1", session_id="s1")
    pipeline.answer("Q2", session_id="s1")

    first_messages = llm.generate.call_args_list[0].args[0]
    second_messages = llm.generate.call_args_list[1].args[0]
    assert len(second_messages) > len(first_messages)

    pipeline.clear_history("s1")
    history_store.clear.assert_called_once_with("s1")


def test_answer_uses_custom_prompt_builder() -> None:
    embedder = create_autospec(BaseEmbedder, instance=True)
    embedder.embed.return_value = [[0.0, 0.0, 0.0, 0.0]]
    storage = create_autospec(BaseVectorStorage, instance=True)
    storage.search.return_value = []

    llm = create_autospec(BaseChatModel, instance=True)
    llm_response = MagicMock()
    llm_response.content = "ok"
    llm.generate.return_value = llm_response

    history_store = create_autospec(BaseHistoryStore, instance=True)
    history_store.get.return_value = []
    prompt_builder = create_autospec(BasePromptBuilder, instance=True)
    prompt_builder.build_messages.return_value = ["mock-message"]

    pipeline = QueryPipeline(
        embedder,
        storage,
        chat_model=llm,
        history_store=history_store,
        prompt_builder=prompt_builder,
    )
    pipeline.answer("Q", session_id="s")

    prompt_builder.build_messages.assert_called_once()
    llm.generate.assert_called_once_with(["mock-message"])
