from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ingestion.preprocessor_base import BasePreprocessor, PreprocessResult

logger = logging.getLogger(__name__)


class TitlePreprocessor(BasePreprocessor):
    """Извлекает заголовок из первого абзаца и сохраняет имя файла в metadata."""

    def preprocess(
        self,
        text: str,
        *,
        source: str | None,
        extra_payload: dict[str, Any] | None,
    ) -> PreprocessResult:
        payload = dict(extra_payload or {})
        file_name = str(payload.get("file_name", "")).strip()

        paragraphs = self._split_paragraphs(text)

        title = (source or "").strip()
        if not title and paragraphs:
            title = paragraphs[0]
        if not title:
            title = Path(file_name).stem if file_name else "untitled"

        # Не индексируем заголовок отдельным чанком: оставляем только контентные абзацы.
        body_text = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
        if not body_text:
            logger.warning("title preprocessor produced empty body: source=%s file=%s", title, file_name)
        logger.info(
            "title preprocessor finished: source=%s paragraphs=%d body_len=%d",
            title,
            len(paragraphs),
            len(body_text),
        )
        return PreprocessResult(text=body_text, source=title, extra_payload=payload)

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"(?:\r?\n){2,}", text) if p.strip()]
