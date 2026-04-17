from __future__ import annotations

from ingestion import PassThroughPreprocessor, TitlePreprocessor


def test_title_preprocessor_uses_first_paragraph() -> None:
    p = TitlePreprocessor()
    out = p.preprocess(
        "Заголовок документа.\n\nТело документа.",
        source=None,
        extra_payload={"file_name": "doc1.txt"},
    )
    assert out.source == "Заголовок документа."
    assert out.text == "Тело документа."
    assert out.extra_payload["file_name"] == "doc1.txt"


def test_title_preprocessor_fallback_to_file_stem() -> None:
    p = TitlePreprocessor()
    out = p.preprocess(
        "   \n\n",
        source=None,
        extra_payload={"file_name": "manual.txt"},
    )
    assert out.source == "manual"
    assert out.text == ""


def test_passthrough_preprocessor_requires_source() -> None:
    p = PassThroughPreprocessor()
    try:
        p.preprocess("x", source=None, extra_payload=None)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "source обязателен" in str(exc)
