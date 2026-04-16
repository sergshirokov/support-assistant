from __future__ import annotations

import re

from chunking.chunker_base import BaseChunker

# Двойной перевод строки: Unix (\n\n), Windows (\r\n\r\n) и смешанные варианты (\r\n\n и т.д.)
_PARAGRAPH_SPLIT = re.compile(r"(?:\r?\n){2,}")


def split_into_paragraphs(text: str) -> list[str]:
    """Абзацы по двойному переводу строки (LF / CRLF и смешанные варианты)."""
    if not text:
        return []
    parts = _PARAGRAPH_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


class ParagraphChunker(BaseChunker):
    """Чанки по пустым строкам между абзацами (два и более перевода строки подряд)."""

    def chunk(self, text: str) -> list[str]:
        return split_into_paragraphs(text)
