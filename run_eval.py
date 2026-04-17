"""Офлайн-прогон eval-датасета: RAG-ответ, hits, latency → JSON в `artifacts/eval/`.

Требует проиндексированную коллекцию (как в CLI: `ingest` после запуска приложения).
История и L1-кэш — in-memory, не затрагивают файлы CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from cache import InMemoryQueryCacheStore
from config import configure_logging, get_settings
from dialogue import InMemoryHistoryStore
from integrations.gigachat import GigaChatEmbedder, GigaChatLangChainChatModel
from query import QueryPipeline
from vector_storage import QdrantVectorStorage

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET = _PROJECT_ROOT / "eval" / "eval_dataset.json"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "artifacts" / "eval"


def _build_eval_query_pipeline(settings: Any) -> tuple[QueryPipeline, QdrantVectorStorage]:
    embedder = GigaChatEmbedder(settings=settings)
    if settings.embedding_vector_size_from_api:
        embedder.embed(["warmup"])

    qdrant_url = settings.qdrant_url
    qdrant_path = settings.qdrant_path
    if qdrant_url is None and qdrant_path is None:
        qdrant_path = "qdrant_storage"
        logger.warning("qdrant_url and qdrant_path not set; using local path=%s", qdrant_path)

    storage = QdrantVectorStorage(
        collection_name=settings.qdrant_collection_name,
        vector_size=embedder.vector_size,
        url=qdrant_url,
        path=qdrant_path,
    )
    history_store = InMemoryHistoryStore(
        max_messages_per_session=settings.dialog_history_limit,
    )
    cache_store = InMemoryQueryCacheStore(
        max_entries_per_session=settings.cache_session_limit,
    )
    pipeline = QueryPipeline(
        embedder,
        storage,
        chat_model=GigaChatLangChainChatModel(settings=settings),
        history_store=history_store,
        cache_store=cache_store,
        settings=settings,
    )
    return pipeline, storage


def _json_safe(obj: Any) -> Any:
    """Привести к структуре, совместимой с JSON (рекурсивно)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("eval_dataset должен быть JSON-массивом объектов")
    required = {"id", "question", "reference_answer", "tags"}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Элемент {i}: ожидался объект")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Элемент {i}: отсутствуют поля: {sorted(missing)}")
        if not isinstance(row.get("tags"), list):
            raise ValueError(f"Элемент {i}: поле tags должно быть массивом строк")
    return data


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогон eval-датасета: ответы RAG, hits, latency → artifacts/eval.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Путь к eval_dataset.json (по умолчанию eval/eval_dataset.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Каталог для JSON-артефакта (по умолчанию artifacts/eval)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Фильтр source для Qdrant (как в CLI); по умолчанию — все источники",
    )
    parser.add_argument(
        "--allow-empty-index",
        action="store_true",
        help="Не завершать с ошибкой, если в коллекции нет векторов (для отладки)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    dataset_path = args.dataset
    if not dataset_path.is_file():
        logger.error("Файл датасета не найден: %s", dataset_path)
        return 1

    try:
        cases = _load_dataset(dataset_path)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Ошибка датасета: %s", exc)
        return 1

    query, storage = _build_eval_query_pipeline(settings)
    embedding_count = storage.count_embeddings()
    if embedding_count == 0 and not args.allow_empty_index:
        logger.error(
            "Векторное хранилище пустое (embeddings=0). "
            "Сначала выполните ingest (CLI), либо --allow-empty-index.",
        )
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"eval_{stamp}.json"

    results: list[dict[str, Any]] = []
    for row in cases:
        case_id = str(row["id"])
        question = str(row["question"]).strip()
        session_id = f"eval-{case_id}"
        started = perf_counter()
        record: dict[str, Any] = {
            "id": case_id,
            "session_id": session_id,
            "question": question,
            "reference_answer": row["reference_answer"],
            "tags": row["tags"],
            "source_filter": args.source,
        }
        try:
            if not question:
                raise ValueError("Пустой question")
            answer_result = query.answer(
                question,
                session_id=session_id,
                source=args.source,
            )
            elapsed_ms = (perf_counter() - started) * 1000.0
            record["answer"] = answer_result.answer
            record["hits"] = _json_safe(answer_result.hits)
            record["latency_ms"] = round(elapsed_ms, 3)
            record["hits_count"] = len(answer_result.hits)
            record["error"] = None
        except Exception as exc:  # noqa: BLE001 — собираем сырой отчёт по всем кейсам
            elapsed_ms = (perf_counter() - started) * 1000.0
            record["answer"] = None
            record["hits"] = []
            record["latency_ms"] = round(elapsed_ms, 3)
            record["hits_count"] = 0
            record["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("case failed: id=%s", case_id)

        results.append(record)

    artifact = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path.resolve()),
            "output_path": str(out_path.resolve()),
            "qdrant_collection": settings.qdrant_collection_name,
            "search_top_k": settings.search_top_k,
            "search_score_threshold": settings.search_score_threshold,
            "source_filter": args.source,
            "embeddings_indexed": embedding_count,
            "cases_total": len(cases),
            "cases_failed": sum(1 for r in results if r.get("error")),
        },
        "results": results,
    }

    out_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("eval artifact written: path=%s", out_path)
    print(f"Готово: {out_path}")
    return 0 if artifact["meta"]["cases_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
