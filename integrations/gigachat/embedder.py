from __future__ import annotations

import logging
from typing import Any, Sequence

from config import get_settings
from config.settings import Settings
from integrations.embedder_base import BaseEmbedder

logger = logging.getLogger(__name__)


class GigaChatEmbedder(BaseEmbedder):
    """Эмбеддинги через официальный SDK GigaChat (POST /embeddings).

    См. https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-embeddings

    Размерность вектора задаёт **модель эмбеддингов** в API; конфиг ``embedding_vector_size`` — для проверки
    и для Qdrant. Если включён ``embedding_vector_size_from_api``, ожидаемая длина берётся из первого ответа API.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        vector_size: int | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._explicit_vector_size = vector_size
        self._inferred_from_api: int | None = None
        self._client: Any | None = client

    @property
    def vector_size(self) -> int:
        if self._explicit_vector_size is not None:
            return self._explicit_vector_size
        if self._inferred_from_api is not None:
            return self._inferred_from_api
        if self._settings.embedding_vector_size_from_api:
            raise RuntimeError(
                "Размерность эмбеддинга станет известна после первого вызова embed(). "
                "Либо отключите embedding_vector_size_from_api и задайте embedding_vector_size."
            )
        return self._settings.embedding_vector_size

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        creds = self._settings.gigachat_credentials
        if not creds:
            logger.error("gigachat embedder init failed: credentials are missing")
            raise ValueError(
                "Не задан gigachat_credentials в окружении или .env — нужен ключ авторизации GigaChat."
            )

        from gigachat import GigaChat

        logger.info(
            "gigachat embedder client initialized: model=%s timeout=%s",
            self._settings.gigachat_embedding_model,
            self._settings.gigachat_timeout,
        )
        return GigaChat(
            credentials=creds,
            scope=self._settings.gigachat_scope,
            verify_ssl_certs=self._settings.gigachat_verify_ssl_certs,
            timeout=self._settings.gigachat_timeout,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            logger.warning("embed skipped: empty texts batch")
            return []

        logger.info(
            "embed started: batch_size=%d model=%s",
            len(texts),
            self._settings.gigachat_embedding_model,
        )
        client = self._get_client()
        result = client.embeddings(
            list(texts),
            model=self._settings.gigachat_embedding_model,
        )
        ordered = sorted(result.data, key=lambda e: e.index)
        vectors = [item.embedding for item in ordered]
        logger.info("embed finished: vectors=%d", len(vectors))

        if self._explicit_vector_size is not None:
            expected = self._explicit_vector_size
            for i, vec in enumerate(vectors):
                if len(vec) != expected:
                    logger.error(
                        "embed failed: explicit vector_size mismatch expected=%d got=%d",
                        expected,
                        len(vec),
                    )
                    raise ValueError(
                        f"Размерность эмбеддинга ({len(vec)}) для текста #{i} не совпадает "
                        f"с переданным vector_size={expected}."
                    )
            return vectors

        if self._settings.embedding_vector_size_from_api:
            dim = len(vectors[0])
            for i, vec in enumerate(vectors):
                if len(vec) != dim:
                    logger.error(
                        "embed failed: inconsistent vector dimensions in batch first=%d current=%d",
                        dim,
                        len(vec),
                    )
                    raise ValueError(
                        "В одном батче эмбеддинги разной длины: "
                        f"#{0} имеет {dim}, #{i} имеет {len(vec)}."
                    )
            self._inferred_from_api = dim
            logger.info("vector size inferred from api: dimension=%d", dim)
            return vectors

        expected = self._settings.embedding_vector_size
        for i, vec in enumerate(vectors):
            if len(vec) != expected:
                logger.error(
                    "embed failed: settings vector_size mismatch expected=%d got=%d",
                    expected,
                    len(vec),
                )
                raise ValueError(
                    f"Размерность эмбеддинга ({len(vec)}) для текста #{i} не совпадает "
                    f"с embedding_vector_size={expected}. Задайте в настройках фактическую размерность "
                    "модели или включите embedding_vector_size_from_api."
                )
        return vectors
