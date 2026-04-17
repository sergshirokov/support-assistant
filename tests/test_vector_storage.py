from __future__ import annotations

import math

import pytest

from vector_storage import BaseVectorStorage, QdrantVectorStorage


def _storage(tmp_path, name: str = "test_collection") -> QdrantVectorStorage:
    return QdrantVectorStorage(
        collection_name=name,
        vector_size=4,
        path=str(tmp_path / "qdrant_data"),
    )


def test_base_vector_storage_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseVectorStorage()  # type: ignore[misc]


def test_constructor_requires_url_or_path() -> None:
    with pytest.raises(ValueError, match="ровно один"):
        QdrantVectorStorage(collection_name="c", vector_size=4, url=None, path=None)
    with pytest.raises(ValueError, match="ровно один"):
        QdrantVectorStorage(
            collection_name="c",
            vector_size=4,
            url="http://localhost:6333",
            path="/tmp/x",
        )


def test_upsert_empty_returns_empty_ids(tmp_path) -> None:
    storage = _storage(tmp_path)
    assert storage.upsert([], []) == []


def test_upsert_rejects_vector_payload_mismatch(tmp_path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(ValueError, match="Число векторов"):
        storage.upsert([[0.0, 0.0, 1.0, 0.0]], [])


def test_upsert_rejects_wrong_vector_size(tmp_path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(ValueError, match="Размер вектора"):
        storage.upsert([[0.0, 0.0, 1.0]], [{"source": "a"}])


def test_upsert_rejects_id_count_mismatch(tmp_path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(ValueError, match="Число id"):
        storage.upsert(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [{"source": "a"}, {"source": "b"}],
            ids=["only-one"],
        )


def test_search_rejects_query_vector_wrong_size(tmp_path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(ValueError, match="Размер вектора запроса"):
        storage.search([1.0, 0.0])


def test_upsert_and_search_returns_scores_and_payloads(tmp_path) -> None:
    storage = _storage(tmp_path)
    v_query = [1.0, 0.0, 0.0, 0.0]
    v_other = [0.0, 1.0, 0.0, 0.0]
    ids = storage.upsert(
        [v_query, v_other],
        [
            {"source": "doc1", "text": "a"},
            {"source": "doc1", "text": "b"},
        ],
        ids=[1, 2],
    )
    assert ids == [1, 2]

    hits = storage.search(v_query, top_k=2)
    assert len(hits) == 2
    assert hits[0]["id"] == 1
    assert hits[0]["payload"]["text"] == "a"
    assert hits[1]["id"] == 2
    assert math.isclose(hits[0]["score"], 1.0, rel_tol=1e-5)


def test_search_filter_by_source(tmp_path) -> None:
    storage = _storage(tmp_path)
    storage.upsert(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        [
            {"source": "alpha", "text": "x"},
            {"source": "beta", "text": "y"},
        ],
    )

    q = [1.0, 0.0, 0.0, 0.0]
    all_hits = storage.search(q, top_k=10)
    assert len(all_hits) == 2

    alpha_hits = storage.search(q, top_k=10, source="alpha")
    assert len(alpha_hits) == 1
    assert alpha_hits[0]["payload"]["source"] == "alpha"

    beta_hits = storage.search(q, top_k=10, source="beta")
    assert len(beta_hits) == 1
    assert beta_hits[0]["payload"]["source"] == "beta"


def test_search_score_threshold(tmp_path) -> None:
    storage = _storage(tmp_path)
    storage.upsert(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"source": "s"}, {"source": "s"}],
    )
    hits = storage.search([1.0, 0.0, 0.0, 0.0], top_k=10, score_threshold=0.99)
    assert len(hits) == 1


def test_qdrant_storage_is_base_vector_storage(tmp_path) -> None:
    storage = _storage(tmp_path)
    assert isinstance(storage, BaseVectorStorage)
    assert storage.vector_size == 4


def test_list_sources_returns_unique_sorted_values(tmp_path) -> None:
    storage = _storage(tmp_path)
    storage.upsert(
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"source": "beta"}, {"source": "alpha"}, {"source": "beta"}],
    )
    assert storage.list_sources() == ["alpha", "beta"]


def test_clear_removes_all_points(tmp_path) -> None:
    storage = _storage(tmp_path)
    storage.upsert(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"source": "s"}, {"source": "s"}],
    )
    assert storage.count_embeddings() == 2
    assert len(storage.search([1.0, 0.0, 0.0, 0.0], top_k=10)) == 2

    storage.clear()
    assert storage.count_embeddings() == 0
    assert storage.search([1.0, 0.0, 0.0, 0.0], top_k=10) == []
