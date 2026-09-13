import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Literal

import pytest
from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    COLLECTION_ID_KEY,
    DENSE_VECTOR_NAME,
    RELEASE_IDS_KEY,
    QdrantHybridRetriever,
)
from tests.domain.factories import COLLECTION_ID, RELEASE_ID, chunk_uuid
from tests.infrastructure.test_qdrant_adapter import FakeReader, record

pytestmark = pytest.mark.integration

DIM = 4
COLLECTION = "chunks_test"
NEXT_RELEASE = uuid.uuid5(uuid.NAMESPACE_URL, "next-release")
TEXTS = {
    "c0": "người lớn 500 mg mỗi 4 giờ",
    "c1": "trẻ em 10 mg/kg",
    "c2": "tối đa 4 g mỗi ngày",
    "c3": "trẻ em 15 mg/kg bản mới",
}
VECTORS = {
    "c0": [1.0, 0.0, 0.0, 0.0],
    "c1": [0.0, 1.0, 0.0, 0.0],
    "c2": [0.5, 0.5, 0.0, 0.0],
    "c3": [0.0, 0.9, 0.1, 0.0],
}
RELEASES = {
    "c0": [RELEASE_ID, NEXT_RELEASE],
    "c1": [RELEASE_ID],
    "c2": [RELEASE_ID, NEXT_RELEASE],
    "c3": [NEXT_RELEASE],
}


class FixedEmbedder:
    """Every query points at c0 in dense space, so only BM25 can lift the matching chunk."""

    @property
    def model(self) -> str:
        return "fake-embedding-4d"

    @property
    def dimension(self) -> int:
        return DIM

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
async def client(qdrant_client: AsyncQdrantClient) -> AsyncIterator[AsyncQdrantClient]:
    """P2's `qdrant_client` (collections wiped) with one small release-tagged collection."""
    await qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
        metadata={"embedding_model": "fake-embedding-4d", "dims": DIM},
    )
    await qdrant_client.create_payload_index(
        collection_name=COLLECTION,
        field_name=COLLECTION_ID_KEY,
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )
    await qdrant_client.create_payload_index(
        collection_name=COLLECTION,
        field_name=RELEASE_IDS_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await qdrant_client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=str(chunk_uuid(label)),
                vector={
                    DENSE_VECTOR_NAME: VECTORS[label],
                    BM25_SPARSE_VECTOR_NAME: models.Document(
                        text=text, model=BM25_MODEL_NAME
                    ),
                },
                payload={
                    COLLECTION_ID_KEY: str(COLLECTION_ID),
                    RELEASE_IDS_KEY: [str(r) for r in RELEASES[label]],
                },
            )
            for label, text in TEXTS.items()
        ],
        wait=True,
    )
    yield qdrant_client


def retriever_for(
    client: AsyncQdrantClient,
    release_id: uuid.UUID,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
) -> QdrantHybridRetriever:
    reader = FakeReader(
        {COLLECTION_ID: release_id},
        [record(label, release_id=r) for label in TEXTS for r in RELEASES[label]],
    )
    return QdrantHybridRetriever(
        client,
        FixedEmbedder(),
        reader,
        collection=COLLECTION,
        scope=["formulary"],
        mode=mode,
        prefetch_k=10,
        rrf_k=2,
    )


async def test_hybrid_search_returns_only_points_of_the_current_release(
    client: AsyncQdrantClient,
) -> None:
    query = [Query(text="trẻ em mg/kg", origin=QueryOrigin.INITIAL)]
    current = retriever_for(client, RELEASE_ID)
    await current.verify_collection(embedding_model="fake-embedding-4d", dimension=DIM)

    hits = (await current.search_many(query, top_k=4))[0]
    assert {h.chunk_version_id for h in hits} == {
        chunk_uuid("c0"),
        chunk_uuid("c1"),
        chunk_uuid("c2"),
    }
    # BM25 pulls the children-dosage chunk up despite the dense vector pointing at c0.
    assert chunk_uuid("c1") in [h.chunk_version_id for h in hits[:2]]
    assert {h.release_id for h in hits} == {RELEASE_ID}

    next_hits = (await retriever_for(client, NEXT_RELEASE).search_many(query, top_k=4))[
        0
    ]
    assert {h.chunk_version_id for h in next_hits} == {
        chunk_uuid("c0"),
        chunk_uuid("c2"),
        chunk_uuid("c3"),
    }
    assert {h.release_id for h in next_hits} == {NEXT_RELEASE}


async def test_bm25_mode_matches_terms_inside_the_current_release_only(
    client: AsyncQdrantClient,
) -> None:
    query = [Query(text="trẻ em mg/kg", origin=QueryOrigin.INITIAL)]

    current = (
        await retriever_for(client, RELEASE_ID, "bm25").search_many(query, top_k=4)
    )[0]
    assert current[0].chunk_version_id == chunk_uuid("c1")
    assert chunk_uuid("c2") not in {
        h.chunk_version_id for h in current
    }  # no query term
    assert {h.release_id for h in current} == {RELEASE_ID}

    upcoming = (
        await retriever_for(client, NEXT_RELEASE, "bm25").search_many(query, top_k=4)
    )[0]
    assert upcoming[0].chunk_version_id == chunk_uuid("c3")
    assert chunk_uuid("c1") not in {h.chunk_version_id for h in upcoming}
