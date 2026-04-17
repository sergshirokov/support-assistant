from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import Settings, get_settings
from dialogue import BaseHistoryStore, InMemoryHistoryStore
from integrations.chat_model_base import BaseChatModel
from integrations.embedder_base import BaseEmbedder
from query.prompt_builder import BasePromptBuilder, RetrievalPromptBuilder
from vector_storage.vector_storage_base import BaseVectorStorage

# Явный ``score_threshold=None`` в вызове — без порога; пропуск параметра — из Settings.
_UNSET = object()


@dataclass(frozen=True)
class QueryResult:
    """Результат поиска по запросу (без генерации ответа LLM)."""

    hits: list[dict[str, Any]]
    query: str


@dataclass(frozen=True)
class QueryAnswerResult:
    """Результат RAG-ответа: retrieved контекст + сгенерированный ответ."""

    query: str
    answer: str
    hits: list[dict[str, Any]]
    session_id: str


class QueryPipeline:
    """Оркестрация query-time: retrieve (эмбеддинг+поиск) и генерация ответа с историей."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        storage: BaseVectorStorage,
        *,
        chat_model: BaseChatModel | None = None,
        history_store: BaseHistoryStore | None = None,
        prompt_builder: BasePromptBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._storage = storage
        self._chat_model = chat_model
        self._history_store = history_store or InMemoryHistoryStore()
        self._prompt_builder = prompt_builder or RetrievalPromptBuilder()
        self._settings = settings or get_settings()

    def _ensure_chat_model(self) -> BaseChatModel:
        if self._chat_model is not None:
            return self._chat_model
        raise ValueError(
            "Для генерации ответа не задан chat_model. "
            "Передайте интеграцию LLM в QueryPipeline(chat_model=...)."
        )

    def _get_history(self, session_id: str) -> list[tuple[str, str]]:
        limit = self._settings.dialog_history_limit
        return self._history_store.get(session_id, limit=limit)

    def _as_text(self, response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        score_threshold: Any = _UNSET,
    ) -> QueryResult:
        """Построить эмбеддинг для ``query`` и выполнить ``search``.

        ``top_k`` и ``score_threshold`` по умолчанию берутся из :class:`~config.settings.Settings`.
        Чтобы не применять порог score, передайте ``score_threshold=None`` явно.
        """
        text = query.strip()
        if not text:
            return QueryResult(hits=[], query=query)

        k = self._settings.search_top_k if top_k is None else top_k
        if score_threshold is _UNSET:
            st: float | None = self._settings.search_score_threshold
        else:
            st = score_threshold

        vectors = self._embedder.embed([text])
        if len(vectors) != 1:
            raise ValueError(
                f"Ожидался ровно один эмбеддинг для запроса, получено {len(vectors)}."
            )

        hits = self._storage.search(
            vectors[0],
            top_k=k,
            source=source,
            score_threshold=st,
        )
        return QueryResult(hits=hits, query=query)

    def answer(
        self,
        query: str,
        *,
        session_id: str,
        top_k: int | None = None,
        source: str | None = None,
        score_threshold: Any = _UNSET,
    ) -> QueryAnswerResult:
        """Выполнить retrieval и сгенерировать ответ LLM с учетом истории session_id."""
        text = query.strip()
        if not text:
            return QueryAnswerResult(query=query, answer="", hits=[], session_id=session_id)

        retrieve_result = self.retrieve(
            text,
            top_k=top_k,
            source=source,
            score_threshold=score_threshold,
        )
        messages = self._prompt_builder.build_messages(
            query=text,
            hits=retrieve_result.hits,
            history=self._get_history(session_id),
            system_prompt=self._settings.rag_system_prompt,
        )

        response = self._ensure_chat_model().generate(messages)
        answer = self._as_text(response)

        self._history_store.append(session_id, "user", text)
        self._history_store.append(session_id, "assistant", answer)
        return QueryAnswerResult(
            query=query,
            answer=answer,
            hits=retrieve_result.hits,
            session_id=session_id,
        )

    def clear_history(self, session_id: str) -> None:
        """Очистить историю сообщений для конкретной сессии."""
        self._history_store.clear(session_id)
