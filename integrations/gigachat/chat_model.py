from __future__ import annotations

from typing import Any, Sequence

from config import get_settings
from config.settings import Settings
from integrations.chat_model_base import BaseChatModel


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
            raise ValueError(
                "Не задан gigachat_credentials в окружении или .env — нужен ключ авторизации GigaChat."
            )

        from langchain_gigachat.chat_models import GigaChat

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
        client = self._get_client()
        return client.invoke(list(messages))
