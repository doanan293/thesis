"""Qdrant writes for the corpus index (spec C §8.3).

One physical collection per embedding model (`chunks_<model_slug>`), partitioned by the
`collection_id` payload (tenant index), with `release_ids` listing the non-retired releases that
contain each chunk version. Payloads carry no text; BM25 vectors are inferred by the Qdrant
server from `models.Document`.
"""

import itertools
import uuid
from collections.abc import Mapping, Sequence

from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.corpus.bundle import model_slug
from pharma_agent.domain.corpus.models import IndexItem, IndexMismatch
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)

CURRENT_ALIAS = "chunks_current"
IDS_PER_REQUEST = 500
POINTS_PER_UPSERT = 256
OPERATIONS_PER_BATCH = 100
TENANT_FIELD = "collection_id"
KEYWORD_PAYLOAD_FIELDS = (
    "release_ids",
    "document_id",
    "section_id",
    "section_revision_id",
    "kind",
)


def physical_collection_name(model: str) -> str:
    return f"chunks_{model_slug(model)}"


def point_payload(item: IndexItem) -> dict[str, object]:
    return {
        "collection_id": str(item.collection_id),
        "release_ids": [str(release_id) for release_id in item.release_ids],
        "document_id": str(item.document_id),
        "section_id": str(item.section_id),
        "section_revision_id": str(item.section_revision_id),
        "kind": item.kind.value,
    }


class QdrantVectorIndex:
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        model: str,
        dimension: int,
        alias: str = CURRENT_ALIAS,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension
        self.collection_name = physical_collection_name(model)
        # Retrieval reads this alias (P3: settings.retrieval.qdrant_collection); the E2E
        # server uses its own alias so it never touches the dev index.
        self.alias = alias

    async def ensure_collection(self) -> str:
        if await self._client.collection_exists(self.collection_name):
            await self._verify_metadata()
        else:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self._dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
                metadata={"embedding_model": self._model, "dims": self._dimension},
            )
        await self._ensure_payload_indexes()
        await self._ensure_alias()
        return self.collection_name

    async def existing_ids(self, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        found: set[uuid.UUID] = set()
        for batch in itertools.batched(ids, IDS_PER_REQUEST):
            records = await self._client.retrieve(
                collection_name=self.collection_name,
                ids=[str(point_id) for point_id in batch],
                with_payload=False,
                with_vectors=False,
            )
            found.update(uuid.UUID(str(record.id)) for record in records)
        return found

    async def upsert(
        self, items: Sequence[IndexItem], vectors: Mapping[str, Sequence[float]]
    ) -> None:
        for batch in itertools.batched(items, POINTS_PER_UPSERT):
            points: list[models.PointStruct] = []
            for item in batch:
                vector = vectors.get(item.embedding_text_sha256)
                if vector is None:
                    raise ValueError(
                        f"no vector for embedding_text_sha256 {item.embedding_text_sha256}"
                    )
                if len(vector) != self._dimension:
                    raise IndexMismatch(
                        f"vector dimension {len(vector)} != {self._dimension} "
                        f"for {self.collection_name}"
                    )
                points.append(
                    models.PointStruct(
                        id=str(item.chunk_version_id),
                        vector={
                            DENSE_VECTOR_NAME: list(vector),
                            BM25_SPARSE_VECTOR_NAME: models.Document(
                                text=item.embedding_text, model=BM25_MODEL_NAME
                            ),
                        },
                        payload=point_payload(item),
                    )
                )
            await self._client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )

    async def set_release_ids(self, items: Sequence[IndexItem]) -> None:
        groups: dict[tuple[str, ...], list[models.ExtendedPointId]] = {}
        for item in items:
            key = tuple(str(release_id) for release_id in item.release_ids)
            groups.setdefault(key, []).append(str(item.chunk_version_id))
        operations: list[models.SetPayloadOperation] = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={"release_ids": list(release_ids)}, points=list(batch)
                )
            )
            for release_ids, point_ids in groups.items()
            for batch in itertools.batched(point_ids, IDS_PER_REQUEST)
        ]
        for batch in itertools.batched(operations, OPERATIONS_PER_BATCH):
            await self._client.batch_update_points(
                collection_name=self.collection_name,
                update_operations=list(batch),
                wait=True,
            )

    async def delete(self, ids: Sequence[uuid.UUID]) -> None:
        for batch in itertools.batched(ids, IDS_PER_REQUEST):
            points: list[models.ExtendedPointId] = [str(point_id) for point_id in batch]
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=points),
                wait=True,
            )

    async def count_release(self, release_id: uuid.UUID) -> int:
        result = await self._client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="release_ids",
                        match=models.MatchValue(value=str(release_id)),
                    )
                ]
            ),
            exact=True,
        )
        return result.count

    async def _verify_metadata(self) -> None:
        info = await self._client.get_collection(self.collection_name)
        metadata = info.config.metadata or {}
        if (
            metadata.get("embedding_model") != self._model
            or metadata.get("dims") != self._dimension
        ):
            raise IndexMismatch(
                f"collection {self.collection_name} has metadata {metadata}, expected "
                f"embedding_model={self._model} dims={self._dimension}"
            )

    async def _ensure_payload_indexes(self) -> None:
        info = await self._client.get_collection(self.collection_name)
        existing = set(info.payload_schema)
        if TENANT_FIELD not in existing:
            await self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=TENANT_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
                wait=True,
            )
        for field in KEYWORD_PAYLOAD_FIELDS:
            if field not in existing:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    async def _ensure_alias(self) -> None:
        aliases = (await self._client.get_aliases()).aliases
        if any(alias.alias_name == self.alias for alias in aliases):
            return
        await self._client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=self.collection_name, alias_name=self.alias
                    )
                )
            ]
        )
