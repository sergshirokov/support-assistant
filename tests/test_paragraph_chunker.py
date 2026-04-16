from __future__ import annotations

import pytest

from chunking import BaseChunker, ParagraphChunker


def test_base_chunker_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseChunker()  # type: ignore[misc]


def test_empty_string() -> None:
    assert ParagraphChunker().chunk("") == []


def test_single_paragraph_unix_no_double_newline() -> None:
    text = "один абзац без разделителя"
    assert ParagraphChunker().chunk(text) == ["один абзац без разделителя"]


def test_two_paragraphs_unix_double_lf() -> None:
    text = "первый\n\nвторой"
    assert ParagraphChunker().chunk(text) == ["первый", "второй"]


def test_two_paragraphs_windows_double_crlf() -> None:
    text = "первый\r\n\r\nвторой"
    assert ParagraphChunker().chunk(text) == ["первый", "второй"]


def test_mixed_crlf_and_lf_between_paragraphs() -> None:
    text = "a\r\n\nb"
    assert ParagraphChunker().chunk(text) == ["a", "b"]


def test_multiple_blank_lines_collapsed_to_paragraph_boundary() -> None:
    text = "x\n\n\n\ny"
    assert ParagraphChunker().chunk(text) == ["x", "y"]


def test_strips_whitespace_around_paragraphs() -> None:
    text = "  hello  \n\n  world  "
    assert ParagraphChunker().chunk(text) == ["hello", "world"]


def test_paragraph_chunker_is_base_chunker() -> None:
    c = ParagraphChunker()
    assert isinstance(c, BaseChunker)
