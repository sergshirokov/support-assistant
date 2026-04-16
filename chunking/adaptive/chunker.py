from __future__ import annotations

import re
from typing import Final

from chunking.chunker_base import BaseChunker
from chunking.paragraph.chunker import split_into_paragraphs

_SENTENCE_SPLIT = re.compile(r"([.!?]+\s+)")


class AdaptiveChunker(BaseChunker):
    """Абзацы (LF/CRLF) → упаковка по ``chunk_size`` с overlap; длинные — по предложениям; короткие — к соседям."""

    def __init__(
        self,
        *,
        chunk_size: int = 500,
        overlap: int = 100,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size должен быть >= 1")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap должен быть >= 0 и < chunk_size")
        if min_chunk_size < 1:
            raise ValueError("min_chunk_size должен быть >= 1")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        chunks = self._pack_paragraphs_greedy(split_into_paragraphs(text))
        chunks = self._split_any_oversized(chunks)
        chunks = self._merge_short_chunks(chunks)
        return chunks

    def _pack_paragraphs_greedy(self, paragraphs: list[str]) -> list[str]:
        """Жадно склеивает абзацы в чанки до ``chunk_size``; при переполнении — flush и overlap."""
        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= self._chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            elif current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._overlap_tail(current_chunk, self._overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            else:
                if len(paragraph) > self._chunk_size:
                    sentence_chunks = self._split_long_paragraph(paragraph)
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_any_oversized(self, chunks: list[str]) -> list[str]:
        """Если после упаковки кусок всё ещё длиннее лимита — режем по предложениям / жёстко."""
        out: list[str] = []
        for c in chunks:
            if len(c) <= self._chunk_size:
                out.append(c)
            else:
                out.extend(self._split_long_paragraph(c))
        return out

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        """Короткие чанки склеиваем с соседями, пока сумма <= ``chunk_size``; иначе — принудительно с предыдущим."""
        if not chunks:
            return []

        merged = [chunks[0]]
        for c in chunks[1:]:
            if len(c) >= self._min_chunk_size:
                merged.append(c)
                continue
            prev = merged[-1]
            if len(prev) + len(c) + 2 <= self._chunk_size:
                merged[-1] = prev + "\n\n" + c
            else:
                merged.append(c)

        i = 0
        while i < len(merged):
            if len(merged[i]) >= self._min_chunk_size or len(merged) == 1:
                i += 1
                continue
            if i > 0:
                merged[i - 1] = merged[i - 1] + "\n\n" + merged[i]
                del merged[i]
                if i <= 1:
                    i = 0
                else:
                    i -= 1
                continue
            if i + 1 < len(merged):
                merged[i] = merged[i] + "\n\n" + merged[i + 1]
                del merged[i + 1]
                continue
            break

        return merged

    def _overlap_tail(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text

        candidate = text[-overlap_size:]
        sentence_starts: Final = (". ", "! ", "? ", "\n")
        best_start = 0
        for delimiter in sentence_starts:
            pos = candidate.find(delimiter)
            if pos != -1 and pos + len(delimiter) > best_start:
                best_start = pos + len(delimiter)

        if best_start > 0:
            return candidate[best_start:].strip()
        return candidate.strip()

    def _split_sentences(self, paragraph: str) -> list[str]:
        parts = _SENTENCE_SPLIT.split(paragraph)
        sentences: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        sentences = self._split_sentences(paragraph)
        if not sentences:
            return self._hard_split(paragraph)

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if len(sentence) > self._chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._hard_split(sentence))
                continue

            if len(current) + len(sentence) + 1 <= self._chunk_size:
                current = current + " " + sentence if current else sentence
            else:
                if current:
                    chunks.append(current)
                    ov = self._overlap_tail(current, self._overlap)
                    current = (ov + " " if ov else "") + sentence
                else:
                    current = sentence

        if current:
            chunks.append(current)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]
        out: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self._chunk_size, n)
            out.append(text[start:end])
            if end == n:
                break
            start = max(0, end - self._overlap)
        return out
