from __future__ import annotations

from query.prompt_builder import RetrievalPromptBuilder


def test_retrieval_prompt_builder_includes_system_history_and_context() -> None:
    builder = RetrievalPromptBuilder()
    messages = builder.build_messages(
        query="Как сбросить пароль?",
        hits=[
            {
                "id": "1",
                "score": 0.9,
                "payload": {"source": "docs/auth.md", "text": "Нажмите 'Забыли пароль'"},
            }
        ],
        history=[("user", "Привет"), ("assistant", "Здравствуйте")],
        system_prompt="Ты помощник",
    )

    assert len(messages) == 4
    assert "Ты помощник" in messages[0].content
    assert messages[1].content == "Привет"
    assert messages[2].content == "Здравствуйте"
    assert "docs/auth.md" in messages[3].content
    assert "Как сбросить пароль?" in messages[3].content
