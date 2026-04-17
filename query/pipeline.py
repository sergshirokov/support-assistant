from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any

from cache import BaseQueryCacheStore, InMemoryQueryCacheStore
from config.settings import Settings, get_settings
from dialogue import BaseHistoryStore, InMemoryHistoryStore
from integrations.chat_model_base import BaseChatModel
from integrations.embedder_base import BaseEmbedder
from query.prompt_builder import BasePromptBuilder, RetrievalPromptBuilder
from vector_storage.vector_storage_base import BaseVectorStorage

logger = logging.getLogger(__name__)

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
        cache_store: BaseQueryCacheStore | None = None,
        prompt_builder: BasePromptBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._storage = storage
        self._chat_model = chat_model
        self._history_store = history_store or InMemoryHistoryStore()
        self._cache_store = cache_store or InMemoryQueryCacheStore()
        self._prompt_builder = prompt_builder or RetrievalPromptBuilder()
        self._settings = settings or get_settings()

    @staticmethod
    def _query_hash(query_text: str) -> str:
        return hashlib.sha256(query_text.encode("utf-8")).hexdigest()

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
        logger.info(
            "retrieve started: query_len=%d top_k=%s source=%s score_threshold=%s",
            len(text),
            top_k if top_k is not None else "default",
            source or "all",
            "default" if score_threshold is _UNSET else score_threshold,
        )
        if not text:
            logger.warning("retrieve skipped: empty query")
            return QueryResult(hits=[], query=query)

        k = self._settings.search_top_k if top_k is None else top_k
        if score_threshold is _UNSET:
            st: float | None = self._settings.search_score_threshold
        else:
            st = score_threshold

        vectors = self._embedder.embed([text])
        if len(vectors) != 1:
            logger.error("retrieve failed: expected one embedding got=%d", len(vectors))
            raise ValueError(
                f"Ожидался ровно один эмбеддинг для запроса, получено {len(vectors)}."
            )
        logger.info("query embedding ready")

        hits = self._storage.search(
            vectors[0],
            top_k=k,
            source=source,
            score_threshold=st,
        )
        logger.info("retrieve finished: hits=%d", len(hits))
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
        logger.info("answer started: session_id=%s query_len=%d", session_id, len(text))
        if not text:
            logger.warning("answer skipped: empty query session_id=%s", session_id)
            return QueryAnswerResult(query=query, answer="", hits=[], session_id=session_id)

        query_hash = self._query_hash(text)
        cached = self._cache_store.get(session_id, source, query_hash)
        if cached is not None:
            logger.info("cache hit: session_id=%s source=%s", session_id, source or "all")
            self._history_store.append(session_id, "user", text)
            self._history_store.append(session_id, "assistant", cached.answer)
            logger.info("history updated from cache: session_id=%s added_messages=2", session_id)
            return QueryAnswerResult(
                query=query,
                answer=cached.answer,
                hits=[],
                session_id=session_id,
            )
        logger.info("cache miss: session_id=%s source=%s", session_id, source or "all")

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
        logger.info(
            "messages prepared: session_id=%s messages=%d hits=%d",
            session_id,
            len(messages),
            len(retrieve_result.hits),
        )

        response = self._ensure_chat_model().generate(messages)
        answer = self._as_text(response)
        logger.info(
            "answer generated: session_id=%s answer_len=%d",
            session_id,
            len(answer),
        )

        self._history_store.append(session_id, "user", text)
        self._history_store.append(session_id, "assistant", answer)
        logger.info("history updated: session_id=%s added_messages=2", session_id)
        self._cache_store.set(
            session_id,
            source,
            query_hash,
            answer=answer,
        )
        logger.info("cache updated: session_id=%s source=%s", session_id, source or "all")
        return QueryAnswerResult(
            query=query,
            answer=answer,
            hits=retrieve_result.hits,
            session_id=session_id,
        )

    def clear_history(self, session_id: str) -> None:
        """Очистить историю сообщений для конкретной сессии."""
        self._history_store.clear(session_id)
        self._cache_store.clear_session(session_id)
        logger.info("history cleared: session_id=%s", session_id)
