from __future__ import annotations

import pytest

from chunking import AdaptiveChunker, BaseChunker


def test_adaptive_is_base_chunker() -> None:
    c = AdaptiveChunker(chunk_size=200, overlap=20, min_chunk_size=10)
    assert isinstance(c, BaseChunker)


def test_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        AdaptiveChunker(chunk_size=100, overlap=100)


def test_windows_paragraphs_then_respects_chunk_size() -> None:
    p1 = "a" * 120
    p2 = "b" * 120
    text = p1 + "\r\n\r\n" + p2
    chunks = AdaptiveChunker(chunk_size=200, overlap=30, min_chunk_size=10).chunk(text)
    assert len(chunks) >= 1
    assert all(len(c) <= 200 for c in chunks)


def test_short_chunk_merged_into_neighbor() -> None:
    text = "x" * 40 + "\n\n" + "y" * 40 + "\n\n" + "z"
    chunks = AdaptiveChunker(chunk_size=500, overlap=50, min_chunk_size=50).chunk(text)
    assert len(chunks) == 1


def test_long_paragraph_split_by_sentences() -> None:
    parts = ["Первое предложение.", " Второе.", " Третье длинное."] * 15
    text = "".join(parts)
    chunks = AdaptiveChunker(chunk_size=80, overlap=10, min_chunk_size=5).chunk(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 80 for c in chunks)


def test_hard_split_when_no_sentence_boundary() -> None:
    text = "w" * 300
    chunks = AdaptiveChunker(chunk_size=100, overlap=20, min_chunk_size=5).chunk(text)
    assert len(chunks) >= 2
    assert "".join(chunks).replace("\n\n", "") == "w" * 300 or "w" * 300 in "".join(chunks)
