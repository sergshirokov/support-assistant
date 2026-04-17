from __future__ import annotations

import logging
from pathlib import Path
from shlex import split as shlex_split
from time import perf_counter

from cache import BaseQueryCacheStore, JsonFileQueryCacheStore
from chunking.paragraph.chunker import ParagraphChunker
from config import configure_logging, get_settings
from dialogue import BaseHistoryStore, JsonFileHistoryStore
from ingestion import IngestionPipeline, TitlePreprocessor
from integrations.gigachat import GigaChatEmbedder, GigaChatLangChainChatModel
from query import QueryPipeline
from vector_storage import QdrantVectorStorage

logger = logging.getLogger(__name__)

_CLI_HISTORY_JSON_PATH = Path("history_store.json")
_CLI_CACHE_JSON_PATH = Path("query_cache.json")

def _build_history_store(*, max_messages_per_session: int) -> BaseHistoryStore:
    store = JsonFileHistoryStore(
        _CLI_HISTORY_JSON_PATH,
        max_messages_per_session=max_messages_per_session,
    )
    logger.info(
        "history store backend selected: json path=%s limit=%d",
        _CLI_HISTORY_JSON_PATH,
        max_messages_per_session,
    )
    return store


def _build_cache_store(*, max_entries_per_session: int) -> BaseQueryCacheStore:
    store = JsonFileQueryCacheStore(
        _CLI_CACHE_JSON_PATH,
        max_entries_per_session=max_entries_per_session,
    )
    logger.info(
        "cache store backend selected: json path=%s limit=%d",
        _CLI_CACHE_JSON_PATH,
        max_entries_per_session,
    )
    return store


def _build_components() -> tuple[
    IngestionPipeline,
    QueryPipeline,
    QdrantVectorStorage,
    BaseHistoryStore,
    BaseQueryCacheStore,
]:
    settings = get_settings()
    embedder = GigaChatEmbedder(settings=settings)
    if settings.embedding_vector_size_from_api:
        embedder.embed(["warmup"])

    qdrant_url = settings.qdrant_url
    qdrant_path = settings.qdrant_path
    if qdrant_url is None and qdrant_path is None:
        # Для CLI MVP по умолчанию используем локальный Qdrant.
        qdrant_path = "qdrant_storage"
        logger.warning("qdrant_url and qdrant_path not set; using local path=%s", qdrant_path)

    storage = QdrantVectorStorage(
        collection_name=settings.qdrant_collection_name,
        vector_size=embedder.vector_size,
        url=qdrant_url,
        path=qdrant_path,
    )
    ingest = IngestionPipeline(
        ParagraphChunker(),
        embedder,
        storage,
        preprocessor=TitlePreprocessor(),
    )
    history_store = _build_history_store(max_messages_per_session=settings.dialog_history_limit)
    cache_store = _build_cache_store(max_entries_per_session=settings.cache_session_limit)
    query = QueryPipeline(
        embedder,
        storage,
        chat_model=GigaChatLangChainChatModel(settings=settings),
        history_store=history_store,
        cache_store=cache_store,
        settings=settings,
    )
    logger.info(
        "components ready: collection=%s chat_model=%s qdrant_mode=%s",
        settings.qdrant_collection_name,
        settings.gigachat_chat_model,
        "url" if qdrant_url is not None else "path",
    )
    return ingest, query, storage, history_store, cache_store


def _print_help() -> None:
    print("Команды:")
    print("  help                    - список команд")
    print("  ingest                  - загрузить все .txt файлы из папки ingestion_data_dir")
    print("  cleardb                 - очистить все точки в текущей коллекции")
    print("  source                  - выбрать source-фильтр из списка")
    print("  stats                   - показать текущие настройки сессии")
    print("  clearhistory            - очистить историю текущей сессии")
    print("  clearcache              - очистить кэш текущей сессии")
    print("  exit                    - выход")
    print("Любой другой ввод считается вопросом ассистенту.")


def _select_source(storage: QdrantVectorStorage) -> tuple[bool, str | None]:
    sources = storage.list_sources()
    print()
    print("Выберите источник:")
    print("1. all (Все)")
    for idx, source in enumerate(sources, start=2):
        print(f"{idx}. {source}")

    raw = input("Номер: ").strip()
    try:
        choice = int(raw)
    except ValueError:
        print("Некорректный ввод, нужен номер.")
        return False, None

    if choice == 1:
        print("source filter: all")
        return True, None

    pos = choice - 2
    if pos < 0 or pos >= len(sources):
        print("Некорректный номер.")
        return False, None

    selected = sources[pos]
    print(f"source filter: {selected}")
    return True, selected


def _ingest_from_dir(ingest: IngestionPipeline, data_dir: Path) -> tuple[int, list[str]]:
    logger.info("ingest directory started: path=%s", data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Папка для ingest не найдена: {data_dir}")

    files = sorted(p for p in data_dir.glob("*.txt") if p.is_file())
    if not files:
        logger.warning("no .txt files found for ingestion: path=%s", data_dir)
        return 0, []

    logger.info("files found for ingestion: count=%d", len(files))
    total_chunks = 0
    indexed_sources: list[str] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        result = ingest.ingest(
            text,
            source=None,
            extra_payload={"file_name": file_path.name},
        )
        total_chunks += len(result.chunks)
        if result.chunks:
            indexed_sources.append(file_path.name)
        logger.info("file ingested: file=%s chunks=%d", file_path.name, len(result.chunks))
    logger.info(
        "ingest directory finished: path=%s total_chunks=%d indexed_sources=%d",
        data_dir,
        total_chunks,
        len(indexed_sources),
    )
    return total_chunks, indexed_sources


def _preview_chunk(text: str, max_len: int = 120) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def _print_answer_block(question_text: str, answer_text: str) -> None:
    print()
    print(f"Вопрос: {question_text}")
    print()
    print(f"Ответ: {answer_text}\n")


def _print_sources_block(hits: list[dict]) -> None:
    if not hits:
        return
    print("Источники:")
    for i, hit in enumerate(hits):
        payload = hit.get("payload") or {}
        chunk_id = hit.get("id")
        score = hit.get("score")
        source = payload.get("source", "unknown")
        file_name = payload.get("file_name", "unknown")
        preview = _preview_chunk(str(payload.get("text", "")))
        print(f"- chunk_id={chunk_id}")
        if isinstance(score, (int, float)):
            print(f"  similarity={score:.4f}")
        else:
            print("  similarity=unknown")
        print(f"  source={source}")
        print(f"  file={file_name}")
        print(f"  preview={preview}")
        if i < len(hits) - 1:
            print()


def run_console() -> None:
    settings = get_settings()
    configure_logging(settings)
    ingest, query, storage, history_store, cache_store = _build_components()
    session_id = "cli"
    current_source: str | None = None
    logger.info(
        "cli started: model=%s top_k=%d collection=%s log_level=%s log_format=%s",
        settings.gigachat_chat_model,
        settings.search_top_k,
        settings.qdrant_collection_name,
        settings.log_level,
        settings.log_format,
    )

    print("Support Assistant CLI")
    print(
        f"model={settings.gigachat_chat_model}, top_k={settings.search_top_k}, "
        f"collection={settings.qdrant_collection_name}"
    )
    if storage.count_embeddings() == 0:
        logger.warning("vector storage is empty")
        print("⚠️ Векторное хранилище пустое.")
        print("Выполните команду ingest для загрузки документов.")
    _print_help()

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not raw:
            continue

        parts = shlex_split(raw)
        cmd = parts[0].lower()
        logger.info("command received: cmd=%s session_id=%s", cmd, session_id)

        if cmd in {"exit", "quit", "q"}:
            print("До встречи.")
            break
        if cmd == "help":
            print()
            _print_help()
            continue
        if cmd == "stats":
            source_count = len(storage.list_sources())
            embedding_count = storage.count_embeddings()
            history_size = len(history_store.get(session_id))
            cache_entries_session = cache_store.count_entries(session_id)
            cache_entries_total = cache_store.count_entries()
            print()
            print(f"session_id: {session_id}")
            print(f"source filter: {current_source or 'all'}")
            print(f"top_k: {settings.search_top_k}")
            print(f"score_threshold: {settings.search_score_threshold}")
            print(f"history_limit: {settings.dialog_history_limit}")
            print(f"history_messages: {history_size}")
            print(f"cache_entries (session): {cache_entries_session}")
            print(f"cache_entries (total): {cache_entries_total}")
            print(f"cache_limit (session): {settings.cache_session_limit}")
            print(f"cache_file: {_CLI_CACHE_JSON_PATH}")
            print(f"collection: {settings.qdrant_collection_name}")
            print(f"indexed documents (sources): {source_count}")
            print(f"indexed embeddings: {embedding_count}")
            print(f"chat model: {settings.gigachat_chat_model}")
            continue
        if cmd == "source":
            changed, selected = _select_source(storage)
            if changed:
                current_source = selected
                logger.info("source filter updated: source=%s", current_source or "all")
            continue
        if cmd == "clearhistory":
            query.clear_history(session_id)
            logger.info("history cleared: session_id=%s", session_id)
            print()
            print("История очищена.")
            continue
        if cmd == "clearcache":
            cache_store.clear_session(session_id)
            logger.info("cache cleared: session_id=%s", session_id)
            print()
            print("Кэш очищен.")
            continue
        if cmd == "cleardb":
            storage.clear()
            logger.info("collection cleared: collection=%s", storage.collection_name)
            print()
            print("Коллекция очищена.")
            continue
        if cmd == "ingest":
            data_dir = Path(settings.ingestion_data_dir)
            try:
                chunk_count, indexed_sources = _ingest_from_dir(ingest, data_dir)
            except ValueError as exc:
                print()
                print(str(exc))
                continue
            print()
            print(f"Готово: indexed chunks={chunk_count}")
            if indexed_sources:
                print("Индексированные файлы:")
                for source in indexed_sources:
                    print(f"- {source}")
            else:
                print("Файлы для индексации не найдены или содержат пустой контент.")
            continue

        logger.info(
            "answer requested: query_len=%d session_id=%s source=%s",
            len(raw),
            session_id,
            current_source or "all",
        )
        started_at = perf_counter()
        answer = query.answer(raw, session_id=session_id, source=current_source)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        _print_answer_block(raw, answer.answer)
        print(f"Время ответа: {elapsed_ms:.2f} ms")
        print()
        _print_sources_block(answer.hits)


def main() -> None:
    run_console()


if __name__ == "__main__":
    main()
