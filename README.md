# Support Assistant (MVP)

Демо-проект **RAG-ассистента поддержки**: локальный **Qdrant**, эмбеддинги через **GigaChat API**, генерация ответов через **LangChain + GigaChat**, чанкинг текста, векторный поиск с фильтром по **источнику**, история диалога и L1-кэш запросов в рамках CLI-сессии.

Цель репозитория — **портфолио / учебный MVP**: понятная архитектура, тесты, без клиентских данных и "закрытых" промптов под конкретного заказчика.

---

## Стек

| Компонент | Выбор |
|-----------|--------|
| Язык | Python **3.11** |
| Векторное хранилище | **Qdrant** (`qdrant-client`), локально через `path` или сервер через `url` |
| Эмбеддинги | Официальный SDK **`gigachat`** (`POST /embeddings`) |
| Генерация | **`langchain-gigachat`** + `langchain-core` |
| Конфиг | **`pydantic-settings`**, переменные окружения + `.env` в корне |
| Тесты | **pytest**, юниты с моками; интеграция с реальным GigaChat — отдельным маркером |

Зависимости: **`requirements.txt`**. Рекомендуется виртуальное окружение **`.venv`** (в git не коммитится).

---

## Архитектурные решения

### 1. Слои и зависимости

- **Векторное хранилище** не знает про чанкинг и не вызывает эмбеддер. Ответственность: векторы + payload + поиск.
- **Чанкинг** — отдельный слой: абстракция + реализации.
- **Эмбеддинги** — отдельный слой: абстракция + реализация GigaChat.
- **Чат-модель** — отдельный слой: абстракция + реализация GigaChat через LangChain.
- **История диалога** — отдельный слой: абстракция + in-memory/json-file реализации.
- **Кэш query-time** — отдельный слой: абстракция + in-memory/json-file реализации.
- **Preprocessor** — отдельный слой ingestion для подготовки контента и metadata.
- **Оркестрация** вынесена в пайплайны:
  - **`ingestion`** — текст -> preprocessor -> чанки -> эмбеддинги -> upsert (`IngestionPipeline`).
  - **`query`** — retrieve (эмбеддинг + поиск) + answer (RAG-генерация с историей) (`QueryPipeline`).

Так проще тестировать и подменять реализации (моки, другой бэкенд).

### 2. Абстракции и расширяемость

| Абстракция | Файл | Реализация(и) |
|------------|------|----------------|
| `BaseEmbedder` | `integrations/embedder_base.py` | `integrations/gigachat/embedder.py` — `GigaChatEmbedder` |
| `BaseChatModel` | `integrations/chat_model_base.py` | `integrations/gigachat/chat_model.py` — `GigaChatLangChainChatModel` |
| `BaseVectorStorage` | `vector_storage/vector_storage_base.py` | `vector_storage/qdrant/storage.py` — `QdrantVectorStorage` |
| `BaseChunker` | `chunking/chunker_base.py` | `chunking/paragraph/`, `chunking/adaptive/` |
| `BaseHistoryStore` | `dialogue/history_base.py` | `dialogue/inmemory/store.py` — `InMemoryHistoryStore`, `dialogue/json_file/store.py` — `JsonFileHistoryStore` |
| `BaseQueryCacheStore` | `cache/cache_base.py` | `cache/inmemory/store.py` — `InMemoryQueryCacheStore`, `cache/json_file/store.py` — `JsonFileQueryCacheStore` |
| `BasePreprocessor` | `ingestion/preprocessor_base.py` | `ingestion/preprocessors/title_preprocessor.py` — `TitlePreprocessor` |
| `BasePromptBuilder` | `query/prompt_builder.py` | `query/prompt_builder.py` — `RetrievalPromptBuilder` |

Зависимости в коде приложения — **на абстракции**, а не на конкретные SDK напрямую.

### 3. Структура папок "провайдер в подпапке"

- **`integrations/gigachat/`** — аналогично можно добавить `integrations/openai/` и т.д.
- **`vector_storage/qdrant/`** — базовый контракт в корне пакета, реализация во вложенной папке.
- **`ingestion/preprocessors/`** — место для новых preprocessors (HTML/PDF и т.д.).

### 4. Именование (Python)

- Модули и пакеты — **`snake_case`**.
- Договорённость команды: по возможности **один публичный класс на файл**.

### 5. Чанкинг

- **Абзацы**: разделитель — **два и более перевода строки** подряд; учитываются **LF и CRLF** (`(?:\r?\n){2,}`).
- **AdaptiveChunker**: целевой размер чанка, overlap, разбиение длинных кусков по предложениям, склейка коротких.

### 6. Preprocessor и metadata ingestion

- Для ingestion используется `TitlePreprocessor`.
- `source` = заголовок документа (первый непустой абзац).
- Заголовок **не индексируется как отдельный чанк**: в индекс идут только контентные абзацы.
- В payload чанка сейчас сохраняются:
  - `source` (заголовок),
  - `text` (текст чанка),
  - `chunk_index`,
  - `file_name`.

### 7. Размерность эмбеддингов

- По умолчанию **`embedding_vector_size`** задаётся в конфиге и **сверяется** с длиной вектора из API.
- Опция **`embedding_vector_size_from_api`**: после первого успешного `embed()` размерность берётся из ответа API.

### 8. Qdrant

- **`url`** — HTTP(S) к серверу Qdrant.
- **`path`** — локальное хранилище без отдельного процесса.
- В `QdrantVectorStorage` добавлены служебные операции:
  - `clear()` — очистить коллекцию,
  - `list_sources()` — список уникальных source,
  - `count_embeddings()` — число точек (векторов).

### 9. Конфигурация

- **`config/settings.py`**: единый **`Settings`**, **`get_settings()`** с `@lru_cache`.
- Пустые строки для `qdrant_url`, `qdrant_path`, `gigachat_credentials` приводятся к **`None`**.
- Поиск: `search_top_k`, `search_score_threshold`.
- Генерация: `gigachat_chat_model`, `llm_temperature`, `llm_max_tokens`, `rag_system_prompt`.
- История: `dialog_history_limit` (используется и для prompt-window, и для ротации storage в CLI).
- Кэш: `cache_session_limit` (лимит L1-кэша на одну сессию).
- CLI ingestion: `ingestion_data_dir` (по умолчанию `data`).
- Логи: `log_level`, `log_format`.

### 10. Query-time (retrieve + answer)

- `QueryPipeline.retrieve(...)`:
  - эмбеддинг запроса,
  - поиск в Qdrant по `top_k`, опционально `source`, `score_threshold`.
- `QueryPipeline.answer(...)`:
  - L1 cache check по ключу `(session_id, source, sha256(query))`,
  - retrieve,
  - сборка сообщений через `RetrievalPromptBuilder`,
  - генерация через `BaseChatModel`,
  - запись истории в `BaseHistoryStore`,
  - запись ответа в `BaseQueryCacheStore`.

### 11. История и кэш в CLI MVP

- CLI использует файловые реализации:
  - `history_store.json` (история),
  - `query_cache.json` (кэш).
- История хранится по `session_id`; кэш — по составному ключу `session_id|source|sha256`.
- На cache-hit LLM не вызывается, но история все равно пополняется (`user` + `assistant`), чтобы сохранять контекст диалога.
- В кэше хранится только `answer` (без `hits`).
- Реализована ротация:
  - для истории — по `dialog_history_limit`,
  - для кэша — по `cache_session_limit`.

### 12. Логирование

- Единая настройка через `config/logging_setup.py`.
- Поддерживаются форматы:
  - `text` (удобно для локального CLI),
  - `json` (удобно для контейнерного/серверного окружения).
- Ключевые runtime-события логируются в пайплайнах, Qdrant-storage, GigaChat-интеграциях и CLI.

---

## Архитектурная блок-схема

```text
                         +----------------------+
                         |      User (CLI)      |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |     app/main.py      |
                         +----+------------+----+
                              |            |
                    ingest    |            | question
                              v            v
                 +------------------+   +------------------+
                 | IngestionPipeline|   |   QueryPipeline  |
                 +---------+--------+   +----+---+---+-----+
                           |                 |   |   |   |
                           v                 |   |   |   |
                 +------------------+        |   |   |   |
                 | TitlePreprocessor|        |   |   |   |
                 +---------+--------+        |   |   |   |
                           |                 |   |   |   |
                           v                 |   |   |   |
                 +------------------+        |   |   |   |
                 | ParagraphChunker |        |   |   |   |
                 +---------+--------+        |   |   |   |
                           |                 |   |   |   |
                           v                 |   |   |   |
                 +------------------+<-------+   |   |   |
                 | GigaChatEmbedder |            |   |   |
                 +---------+--------+            |   |   |
                           |                     |   |   |
                           v                     |   |   |
                 +------------------+<-----------+   |   |
                 |QdrantVectorStorage|-------------->   |
                 +------------------+   hits+score      |
                                                 |       |
                                                 v       v
                                     +------------------+  +----------------------+
                                     |RetrievalPrompt...|  |InMemoryHistoryStore  |
                                     +------------------+  +----------------------+
                                                 |
                                                 v
                                     +----------------------+
                                     |GigaChatLangChainChat |
                                     +----------+-----------+
                                                |
                                                v
                                         answer to CLI
```

---

## Структура репозитория (основное)

```
app/                     # CLI MVP
config/                  # Settings, get_settings
cache/                   # BaseQueryCacheStore + inmemory/json_file
integrations/            # BaseEmbedder, BaseChatModel + gigachat/
vector_storage/          # BaseVectorStorage + qdrant/
chunking/                # BaseChunker, paragraph/, adaptive/
dialogue/                # BaseHistoryStore + inmemory/json_file
ingestion/               # IngestionPipeline + preprocessors/
query/                   # QueryPipeline + prompt_builder
tests/                   # Юнит-тесты + интеграция GigaChat
pytest.ini               # addopts = -m "not integration"
.env.example             # Шаблон переменных (без секретов)
requirements.txt
.vscode/                 # debug configs (pytest + CLI)
run_cli.py               # тонкий запуск CLI из корня
```

**`pytest.ini`** коммитится: `pythonpath`, `testpaths`, маркеры, **`addopts = -m "not integration"`**.  
**`.env`** в git не попадает (секреты). См. **`.env.example`**.

---

## Консольный MVP

Запуск:

```bash
python -m app.main
# или
python run_cli.py
```

Доступные команды:

- `help`
- `ingest` — индексировать все `.txt` из `INGESTION_DATA_DIR`
- `cleardb` — очистить коллекцию
- `source` — выбрать source-фильтр из списка (`all` + источники)
- `stats`
- `clearhistory`
- `clearcache`
- `exit`

Если при старте коллекция пуста, CLI показывает предупреждение и подсказывает выполнить `ingest`.

Вывод ответа включает:

- `Вопрос: ...`
- `Ответ: ...`
- `Время ответа: ... ms`
- `Источники` по каждому hit:
  - `chunk_id`
  - `similarity`
  - `source`
  - `file`
  - `preview`

---

## Запуск тестов

```bash
# активация .venv, затем:
pip install -r requirements.txt
python -m pytest
python -m pytest -m integration --override-ini=addopts=
```

Интеграционные тесты помечены **`@pytest.mark.integration`**; без ключа GigaChat они **skip**, а не падают.

В **Cursor / VS Code** можно использовать конфигурации из **`.vscode/launch.json`**:

- `Pytest: all tests`
- `Pytest: current file`
- `Pytest: integration only`
- `CLI MVP (module)`
- `CLI MVP (run_cli.py)`

---

## Переменные окружения (кратко)

Полный список — в **`.env.example`**. Важные группы:

| Область | Примеры переменных |
|---------|-------------------|
| Qdrant | `QDRANT_URL` **или** `QDRANT_PATH` |
| GigaChat embeddings | `GIGACHAT_CREDENTIALS`, `GIGACHAT_EMBEDDING_MODEL` |
| GigaChat generation | `GIGACHAT_CHAT_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` |
| Эмбеддинги | `EMBEDDING_VECTOR_SIZE`, `EMBEDDING_VECTOR_SIZE_FROM_API` |
| Чанкинг | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_MIN_SIZE` |
| Поиск | `SEARCH_TOP_K`, `SEARCH_SCORE_THRESHOLD` |
| История и кэш | `DIALOG_HISTORY_LIMIT`, `CACHE_SESSION_LIMIT` |
| Логирование | `LOG_LEVEL`, `LOG_FORMAT` |
| CLI ingest | `INGESTION_DATA_DIR` |

---

## Дорожная карта

- Рефактор orchestration-слоя query (выделение cache/history-политик в отдельный сервис/декоратор).
- Персистентный backend-уровень (например, Postgres) для users/licenses/audit и следующих этапов продукта.
- Поддержка preprocessors для HTML/PDF.
- Затем виджет на сайте.

---

## Лицензия

При публикации на GitHub имеет смысл добавить файл **`LICENSE`** (часто MIT для портфолио) и убедиться, что в истории коммитов **нет** закоммиченных ключей и `.env`.
