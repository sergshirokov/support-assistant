from __future__ import annotations

import os

import pytest

from config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://example:6333")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "my_collection")
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "512")

    s = Settings()
    assert s.qdrant_url == "http://example:6333"
    assert s.qdrant_collection_name == "my_collection"
    assert s.embedding_vector_size == 512


def test_empty_string_env_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "")
    s = Settings()
    assert s.qdrant_url is None


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://a:6333")
    first = get_settings()
    monkeypatch.setenv("QDRANT_URL", "http://b:6333")
    second = get_settings()
    assert first is second
    assert second.qdrant_url == "http://a:6333"

    get_settings.cache_clear()
    third = get_settings()
    assert third.qdrant_url == "http://b:6333"


def test_get_settings_after_clear_picks_new_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://first:6333")
    assert get_settings().qdrant_url == "http://first:6333"
    get_settings.cache_clear()
    monkeypatch.setenv("QDRANT_URL", "http://second:6333")
    assert get_settings().qdrant_url == "http://second:6333"


def test_settings_ignores_unknown_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNKNOWN_FUTURE_KEY_XYZ", "1")
    Settings()


def test_chunk_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    monkeypatch.setenv("CHUNK_MIN_SIZE", "40")
    s = Settings()
    assert s.chunk_size == 400
    assert s.chunk_overlap == 80
    assert s.chunk_min_size == 40
    assert s.adaptive_chunker_kwargs() == {
        "chunk_size": 400,
        "overlap": 80,
        "min_chunk_size": 40,
    }


def test_chunk_overlap_must_be_less_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings()


def test_chunk_min_size_must_not_exceed_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "50")
    monkeypatch.setenv("CHUNK_MIN_SIZE", "60")
    with pytest.raises(ValueError, match="chunk_min_size"):
        Settings()


def test_search_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_TOP_K", "10")
    monkeypatch.setenv("SEARCH_SCORE_THRESHOLD", "0.42")
    s = Settings()
    assert s.search_top_k == 10
    assert s.search_score_threshold == 0.42


def test_search_top_k_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_TOP_K", "0")
    with pytest.raises(ValueError, match="search_top_k"):
        Settings()


def test_dialog_history_limit_must_be_non_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIALOG_HISTORY_LIMIT", "-1")
    with pytest.raises(ValueError, match="dialog_history_limit"):
        Settings()


def test_cache_session_limit_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_SESSION_LIMIT", "0")
    with pytest.raises(ValueError, match="cache_session_limit"):
        Settings()


