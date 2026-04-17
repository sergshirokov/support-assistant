from __future__ import annotations

from typing import Any, Sequence
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FilterSelector,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from vector_storage.vector_storage_base import BaseVectorStorage


class QdrantVectorStorage(BaseVectorStorage):
    """Реализация :class:`BaseVectorStorage` поверх Qdrant."""

    def __init__(
        self,
        *,
        collection_name: str,
        vector_size: int,
        url: str | None = None,
        path: str | None = None,
        distance: Distance = Distance.COSINE,
    ) -> None:
        if (url is None) == (path is None):
            raise ValueError("Укажите ровно один из параметров: url или path")

        if url is not None:
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=path)

        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
        self._ensure_collection()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=self._distance),
        )

    def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
        ids: Sequence[str | int] | None = None,
    ) -> list[str | int]:
        if len(vectors) != len(payloads):
            raise ValueError("Число векторов и payload должно совпадать")
        if not vectors:
            return []

        point_ids: list[str | int]
        if ids is None:
            point_ids = [str(uuid4()) for _ in vectors]
        else:
            if len(ids) != len(vectors):
                raise ValueError("Число id должно совпадать с числом векторов")
            point_ids = list(ids)

        for v in vectors:
            if len(v) != self._vector_size:
                raise ValueError(
                    f"Размер вектора {len(v)} не совпадает с ожидаемым {self._vector_size}"
                )

        points = [
            PointStruct(id=pid, vector=list(vec), payload=dict(pl))
            for pid, vec, pl in zip(point_ids, vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        return point_ids

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        source: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self._vector_size:
            raise ValueError(
                f"Размер вектора запроса {len(query_vector)} не совпадает с {self._vector_size}"
            )

        query_filter: Filter | None = None
        if source is not None:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )

        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        return [
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload or {},
            }
            for p in hits
        ]

    def clear(self) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(filter=Filter()),
            wait=True,
        )

    def list_sources(self) -> list[str]:
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=None,
            with_vectors=False,
            with_payload=True,
            limit=10_000,
        )
        values = {
            str(p.payload.get("source"))
            for p in points
            if p.payload is not None and p.payload.get("source") is not None
        }
        return sorted(values)

    def count_embeddings(self) -> int:
        result = self._client.count(
            collection_name=self._collection_name,
            count_filter=None,
            exact=True,
        )
        return int(result.count)
