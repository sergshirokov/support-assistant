from __future__ import annotations

from pathlib import Path
from shlex import split as shlex_split

from chunking.paragraph.chunker import ParagraphChunker
from config import get_settings
from dialogue import BaseHistoryStore, InMemoryHistoryStore
from ingestion import IngestionPipeline, TitlePreprocessor
from integrations.gigachat import GigaChatEmbedder, GigaChatLangChainChatModel
from query import QueryPipeline
from vector_storage import QdrantVectorStorage


def _build_components() -> tuple[
    IngestionPipeline, QueryPipeline, QdrantVectorStorage, BaseHistoryStore
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
    history_store = InMemoryHistoryStore()
    query = QueryPipeline(
        embedder,
        storage,
        chat_model=GigaChatLangChainChatModel(settings=settings),
        history_store=history_store,
        settings=settings,
    )
    return ingest, query, storage, history_store


def _print_help() -> None:
    print("Команды:")
    print("  help                    - список команд")
    print("  ingest                  - загрузить все .txt файлы из папки ingestion_data_dir")
    print("  cleardb                 - очистить все точки в текущей коллекции")
    print("  source                  - выбрать source-фильтр из списка")
    print("  stats                   - показать текущие настройки сессии")
    print("  clearhistory            - очистить историю текущей сессии")
    print("  exit                    - выход")
    print("Любой другой ввод считается вопросом ассистенту.")


def _select_source(storage: QdrantVectorStorage) -> tuple[bool, str | None]:
    sources = storage.list_sources()
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
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Папка для ingest не найдена: {data_dir}")

    files = sorted(p for p in data_dir.glob("*.txt") if p.is_file())
    if not files:
        return 0, []

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
    return total_chunks, indexed_sources


def _preview_chunk(text: str, max_len: int = 120) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def _print_answer_block(answer_text: str) -> None:
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
    ingest, query, storage, history_store = _build_components()
    session_id = "cli"
    current_source: str | None = None

    print("Support Assistant CLI")
    print(
        f"model={settings.gigachat_chat_model}, top_k={settings.search_top_k}, "
        f"collection={settings.qdrant_collection_name}"
    )
    if storage.count_embeddings() == 0:
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

        if cmd in {"exit", "quit", "q"}:
            print("До встречи.")
            break
        if cmd == "help":
            _print_help()
            continue
        if cmd == "stats":
            source_count = len(storage.list_sources())
            embedding_count = storage.count_embeddings()
            history_size = len(history_store.get(session_id))
            print(f"session_id: {session_id}")
            print(f"source filter: {current_source or 'all'}")
            print(f"top_k: {settings.search_top_k}")
            print(f"score_threshold: {settings.search_score_threshold}")
            print(f"history_limit: {settings.dialog_history_limit}")
            print(f"history_messages: {history_size}")
            print(f"collection: {settings.qdrant_collection_name}")
            print(f"indexed documents (sources): {source_count}")
            print(f"indexed embeddings: {embedding_count}")
            print(f"chat model: {settings.gigachat_chat_model}")
            continue
        if cmd == "source":
            changed, selected = _select_source(storage)
            if changed:
                current_source = selected
            continue
        if cmd == "clearhistory":
            query.clear_history(session_id)
            print("История очищена.")
            continue
        if cmd == "cleardb":
            storage.clear()
            print("Коллекция очищена.")
            continue
        if cmd == "ingest":
            data_dir = Path(settings.ingestion_data_dir)
            try:
                chunk_count, indexed_sources = _ingest_from_dir(ingest, data_dir)
            except ValueError as exc:
                print(str(exc))
                continue
            print(f"Готово: indexed chunks={chunk_count}")
            if indexed_sources:
                print("Индексированные файлы:")
                for source in indexed_sources:
                    print(f"- {source}")
            else:
                print("Файлы для индексации не найдены или содержат пустой контент.")
            continue

        answer = query.answer(raw, session_id=session_id, source=current_source)
        _print_answer_block(answer.answer)
        _print_sources_block(answer.hits)


def main() -> None:
    run_console()


if __name__ == "__main__":
    main()
