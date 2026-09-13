import uuid

import pytest
from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.corpus.bundle import BlockKind
from pharma_agent.domain.corpus.identity import sha256_hex
from pharma_agent.domain.corpus.models import IndexItem, IndexMismatch
from pharma_agent.domain.corpus.ports import VectorIndex
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
)
from pharma_agent.infrastructure.retrieval.qdrant_index import (
    CURRENT_ALIAS,
    QdrantVectorIndex,
)
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, fake_vector

pytestmark = pytest.mark.integration

COLLECTION = "chunks_fake_embedding_4d"
COLLECTION_ID = uuid.uuid4()


def item(text: str, release_ids: list[uuid.UUID]) -> IndexItem:
    return IndexItem(
        chunk_version_id=uuid.uuid4(),
        collection_id=COLLECTION_ID,
        document_id=uuid.uuid4(),
        section_id=uuid.uuid4(),
        section_revision_id=uuid.uuid4(),
        kind=BlockKind.PROSE,
        embedding_text=text,
        embedding_text_sha256=sha256_hex(text),
        release_ids=release_ids,
    )


def fake_index(client: AsyncQdrantClient, dimension: int = 4) -> QdrantVectorIndex:
    return QdrantVectorIndex(client, model=FAKE_EMBEDDING_MODEL, dimension=dimension)


async def test_ensure_collection_creates_the_layout_once(
    qdrant_client: AsyncQdrantClient,
) -> None:
    index: VectorIndex = fake_index(qdrant_client)
    assert await index.ensure_collection() == COLLECTION
    assert await index.ensure_collection() == COLLECTION

    info = await qdrant_client.get_collection(COLLECTION)
    assert info.config.metadata == {
        "embedding_model": FAKE_EMBEDDING_MODEL,
        "dims": FAKE_EMBEDDING_DIMENSION,
    }
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[DENSE_VECTOR_NAME].size == FAKE_EMBEDDING_DIMENSION
    assert vectors[DENSE_VECTOR_NAME].distance is models.Distance.COSINE
    sparse = info.config.params.sparse_vectors
    assert sparse is not None
    assert sparse[BM25_SPARSE_VECTOR_NAME].modifier is models.Modifier.IDF
    assert set(info.payload_schema) == {
        "collection_id",
        "release_ids",
        "document_id",
        "section_id",
        "section_revision_id",
        "kind",
    }
    tenant = info.payload_schema["collection_id"].params
    assert isinstance(tenant, models.KeywordIndexParams) and tenant.is_tenant is True
    aliases = (await qdrant_client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        (CURRENT_ALIAS, COLLECTION)
    ]


async def test_ensure_collection_rejects_mismatched_metadata_and_keeps_alias(
    qdrant_client: AsyncQdrantClient,
) -> None:
    await fake_index(qdrant_client).ensure_collection()
    with pytest.raises(IndexMismatch, match="fake-embedding-4d"):
        await fake_index(qdrant_client, dimension=8).ensure_collection()

    other = QdrantVectorIndex(qdrant_client, model="Other-Embedding:v2", dimension=4)
    assert await other.ensure_collection() == "chunks_other_embedding_v2"
    aliases = (await qdrant_client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        (CURRENT_ALIAS, COLLECTION)
    ]


async def test_non_default_alias_is_created_and_used(
    qdrant_client: AsyncQdrantClient,
) -> None:
    index = QdrantVectorIndex(
        qdrant_client,
        model=FAKE_EMBEDDING_MODEL,
        dimension=FAKE_EMBEDDING_DIMENSION,
        alias="e2e_chunks_current",
    )
    assert index.alias == "e2e_chunks_current"
    assert await index.ensure_collection() == COLLECTION
    assert await index.ensure_collection() == COLLECTION
    aliases = (await qdrant_client.get_aliases()).aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        ("e2e_chunks_current", COLLECTION)
    ]

    release_id = uuid.uuid4()
    point = item("paracetamol hạ sốt", [release_id])
    await index.upsert(
        [point], {point.embedding_text_sha256: fake_vector(point.embedding_text)}
    )
    found = await qdrant_client.query_points(
        "e2e_chunks_current",
        query=fake_vector(point.embedding_text),
        using=DENSE_VECTOR_NAME,
        limit=1,
    )
    assert [str(p.id) for p in found.points] == [str(point.chunk_version_id)]

    await fake_index(qdrant_client).ensure_collection()
    names = {a.alias_name for a in (await qdrant_client.get_aliases()).aliases}
    assert names == {"e2e_chunks_current", CURRENT_ALIAS}


async def test_upsert_payload_release_ids_count_and_delete(
    qdrant_client: AsyncQdrantClient,
) -> None:
    index = fake_index(qdrant_client)
    await index.ensure_collection()
    r1, r2 = uuid.uuid4(), uuid.uuid4()
    fever = item("paracetamol hạ sốt cho người lớn", [r1])
    warfarin = item("warfarin tương tác chống đông máu", [r1])
    vectors = {
        x.embedding_text_sha256: fake_vector(x.embedding_text)
        for x in (fever, warfarin)
    }

    await index.upsert([fever, warfarin], vectors)

    unknown = uuid.uuid4()
    assert await index.existing_ids(
        [fever.chunk_version_id, warfarin.chunk_version_id, unknown]
    ) == {fever.chunk_version_id, warfarin.chunk_version_id}
    (record,) = await qdrant_client.retrieve(
        COLLECTION, ids=[str(fever.chunk_version_id)], with_payload=True
    )
    assert record.payload == {
        "collection_id": str(COLLECTION_ID),
        "release_ids": [str(r1)],
        "document_id": str(fever.document_id),
        "section_id": str(fever.section_id),
        "section_revision_id": str(fever.section_revision_id),
        "kind": "prose",
    }
    bm25 = await qdrant_client.query_points(
        COLLECTION,
        query=models.Document(text="chống đông", model=BM25_MODEL_NAME),
        using=BM25_SPARSE_VECTOR_NAME,
        limit=1,
    )
    assert bm25.points[0].id == str(warfarin.chunk_version_id)
    assert await index.count_release(r1) == 2

    await index.set_release_ids(
        [
            fever.model_copy(update={"release_ids": sorted([r1, r2])}),
            warfarin.model_copy(update={"release_ids": [r2]}),
        ]
    )
    assert (await index.count_release(r1), await index.count_release(r2)) == (1, 2)

    await index.delete([warfarin.chunk_version_id])
    assert await index.existing_ids([warfarin.chunk_version_id]) == set()
    assert await index.count_release(r2) == 1

    with pytest.raises(ValueError, match="no vector"):
        await index.upsert([item("không có vector", [r1])], vectors)
