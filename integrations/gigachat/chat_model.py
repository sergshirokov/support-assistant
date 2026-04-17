from __future__ import annotations

import logging
from typing import Any, Sequence

from config import get_settings
from config.settings import Settings
from integrations.chat_model_base import BaseChatModel

logger = logging.getLogger(__name__)


class GigaChatLangChainChatModel(BaseChatModel):
    """LangChain-адаптер для чат-генерации через GigaChat."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        creds = self._settings.gigachat_credentials
        if not creds:
            logger.error("gigachat chat model init failed: credentials are missing")
            raise ValueError(
                "Не задан gigachat_credentials в окружении или .env — нужен ключ авторизации GigaChat."
            )

        from langchain_gigachat.chat_models import GigaChat

        logger.info(
            "gigachat chat client initialized: model=%s temperature=%s max_tokens=%s",
            self._settings.gigachat_chat_model,
            self._settings.llm_temperature,
            self._settings.llm_max_tokens,
        )
        return GigaChat(
            credentials=creds,
            scope=self._settings.gigachat_scope,
            verify_ssl_certs=self._settings.gigachat_verify_ssl_certs,
            timeout=self._settings.gigachat_timeout,
            model=self._settings.gigachat_chat_model,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )

    def generate(self, messages: Sequence[Any]) -> Any:
        logger.info("chat generation started: messages=%d", len(messages))
        client = self._get_client()
        response = client.invoke(list(messages))
        logger.info("chat generation finished")
        return response
