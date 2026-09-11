import pytest
from qdrant_client import AsyncQdrantClient, models

from pharma_agent.domain.retrieval.models import HydrateStrategy, Query, QueryOrigin
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    QdrantHybridRetriever,
    QdrantHydrator,
)

pytestmark = pytest.mark.integration

DIM = 4


def payload(i: int, text: str, strategy: str = "full_section") -> dict:
    return {
        "chunk_id": f"c{i}",
        "section_id": "sec-1",
        "chunk_index": i,
        "hydrate_strategy": strategy,
        "source": "duoc_thu",
        "title": "Paracetamol",
        "section": "Liều dùng",
        "start_page": 10,
        "end_page": 10,
        "context_header": "Paracetamol > Liều dùng",
        "chunk_text": text,
        "embedding_text": f"Paracetamol > Liều dùng\n\n{text}",
    }


class FixedEmbedder:
    async def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
async def client():
    from testcontainers.community.qdrant import QdrantContainer

    with QdrantContainer("qdrant/qdrant:latest") as container:
        client = AsyncQdrantClient(
            url=f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"
        )
        await client.create_collection(
            collection_name="t",
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
        )
        await client.create_payload_index(
            collection_name="t",
            field_name="section_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name="t",
            field_name="chunk_index",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        texts = ["người lớn 500 mg mỗi 4 giờ", "trẻ em 10 mg/kg", "tối đa 4 g mỗi ngày"]
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
        await client.upsert(
            collection_name="t",
            points=[
                models.PointStruct(
                    id=i,
                    vector={
                        DENSE_VECTOR_NAME: vectors[i],
                        BM25_SPARSE_VECTOR_NAME: models.Document(
                            text=texts[i], model=BM25_MODEL_NAME
                        ),
                    },
                    payload=payload(i, texts[i]),
                )
                for i in range(3)
            ],
        )
        yield client
        await client.close()


async def test_hybrid_search_and_hydrate_against_real_qdrant(
    client: AsyncQdrantClient,
) -> None:
    retriever = QdrantHybridRetriever(
        client, FixedEmbedder(), "t", mode="hybrid", prefetch_k=10, rrf_k=2
    )
    await retriever.verify_collection(DIM)
    hits = await retriever.search_many(
        [Query(text="trẻ em mg/kg", origin=QueryOrigin.INITIAL)], top_k=3
    )
    ids = [h.chunk_id for h in hits[0]]
    assert set(ids) == {"c0", "c1", "c2"}
    assert (
        "c1" in ids[:2]
    )  # BM25 pulls the children-dosage chunk up despite the dense vector pointing at c0

    chunks = await QdrantHydrator(client, "t", window=1).hydrate(
        hits[0][0], HydrateStrategy.FULL_SECTION
    )
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
