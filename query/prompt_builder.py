from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class BasePromptBuilder(ABC):
    """Контракт сборки сообщений для chat-модели."""

    @abstractmethod
    def build_messages(
        self,
        *,
        query: str,
        hits: Sequence[dict[str, Any]],
        history: Sequence[tuple[str, str]],
        system_prompt: str,
    ) -> list[Any]:
        """Собрать список сообщений в формате LangChain."""


class RetrievalPromptBuilder(BasePromptBuilder):
    """Базовый prompt-builder для RAG: history + context + текущий вопрос."""

    @staticmethod
    def _render_context(hits: Sequence[dict[str, Any]]) -> str:
        if not hits:
            return "Контекст не найден."
        lines: list[str] = []
        for i, hit in enumerate(hits, start=1):
            payload = hit.get("payload") or {}
            source = payload.get("source", "unknown")
            text = payload.get("text", "")
            lines.append(f"[{i}] source={source}\n{text}")
        return "\n\n".join(lines)

    def build_messages(
        self,
        *,
        query: str,
        hits: Sequence[dict[str, Any]],
        history: Sequence[tuple[str, str]],
        system_prompt: str,
    ) -> list[Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list[Any] = [SystemMessage(content=system_prompt)]
        for role, content in history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        context = self._render_context(hits)
        prompt = (
            "Контекст из базы знаний:\n"
            f"{context}\n\n"
            "Вопрос пользователя:\n"
            f"{query}"
        )
        messages.append(HumanMessage(content=prompt))
        return messages
