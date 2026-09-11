from types import SimpleNamespace

import pytest
from qdrant_client import models

from pharma_agent.domain.retrieval.models import HydrateStrategy, Query, QueryOrigin
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_SPARSE_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    OpenAiEmbedder,
    QdrantHybridRetriever,
    QdrantHydrator,
    hit_from_point,
)

PAYLOAD = {
    "chunk_id": "c1",
    "section_id": "sec-1",
    "chunk_index": 2,
    "hydrate_strategy": "full_section",
    "source": "duoc_thu",
    "title": "Paracetamol",
    "section": "Liều dùng",
    "start_page": 10,
    "end_page": 11,
    "context_header": "Paracetamol > Liều dùng",
    "chunk_text": "Người lớn 500 mg",
    "embedding_text": "Paracetamol > Liều dùng\n\nNgười lớn 500 mg",
    "colloquial_mapping": {
        "key": "paracetamol",
        "aliases": ["thuốc hạ sốt"],
        "product_names": ["Panadol"],
    },
    "term_annotations": [{"term": "APAP", "vi": ["acetaminophen"]}],
    "table_id": "",
}


def test_hit_from_point_maps_payload_contract() -> None:
    hit = hit_from_point(PAYLOAD, score=0.42, query_text="q")
    assert hit.chunk_id == "c1" and hit.hydrate_strategy is HydrateStrategy.FULL_SECTION
    assert hit.fusion_score == 0.42 and hit.matched_queries == ["q"]
    assert (
        hit.colloquial_mapping is not None
        and hit.colloquial_mapping.product_names == ["Panadol"]
    )
    assert hit.term_annotations[0].vi == ["acetaminophen"]


class FakeEmbeddings:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.inputs: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]):
        self.inputs.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * self.dimension)
            for i in range(len(input))
        ]
        return SimpleNamespace(data=list(reversed(data)))


async def test_embedder_returns_vectors_in_input_order_and_checks_dimension() -> None:
    client = SimpleNamespace(embeddings=FakeEmbeddings(4))
    embedder = OpenAiEmbedder(client, model="m", dimension=4)
    vectors = await embedder.embed(["a", "b"])
    assert vectors == [[0.0] * 4, [1.0] * 4]
    bad = OpenAiEmbedder(
        SimpleNamespace(embeddings=FakeEmbeddings(3)), model="m", dimension=4
    )
    with pytest.raises(RetrievalError, match="dimension"):
        await bad.embed(["a"])


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeQdrant:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []
        self.scroll_calls: list[dict] = []

    async def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(payload=PAYLOAD, score=0.9)])

    async def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        points = [
            SimpleNamespace(
                payload={
                    **PAYLOAD,
                    "chunk_id": f"c{i}",
                    "chunk_index": i,
                    "chunk_text": f"t{i}",
                }
            )
            for i in (3, 1, 2)
        ]
        return points, None

    async def get_collection(self, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={DENSE_VECTOR_NAME: SimpleNamespace(size=2)}
                )
            )
        )


async def test_hybrid_search_builds_rrf_prefetch_query() -> None:
    client = FakeQdrant()
    retriever = QdrantHybridRetriever(
        client,
        FakeEmbedder(),
        "coll",
        mode="hybrid",
        prefetch_k=7,
        rrf_k=2,
        max_concurrent=2,
    )
    hits = await retriever.search_many(
        [Query(text="paracetamol", origin=QueryOrigin.INITIAL)], top_k=5
    )
    assert hits[0][0].chunk_id == "c1" and hits[0][0].matched_queries == ["paracetamol"]
    call = client.query_calls[0]
    assert (
        call["collection_name"] == "coll"
        and call["limit"] == 5
        and call["with_payload"] is True
    )
    dense, sparse = call["prefetch"]
    assert (
        dense.using == DENSE_VECTOR_NAME
        and dense.limit == 7
        and dense.query == [0.1, 0.2]
    )
    assert sparse.using == BM25_SPARSE_VECTOR_NAME and isinstance(
        sparse.query, models.Document
    )
    assert sparse.query.text == "paracetamol" and sparse.query.model == "Qdrant/bm25"
    assert isinstance(call["query"], models.RrfQuery) and call["query"].rrf.k == 2


async def test_dense_mode_and_collection_verification() -> None:
    client = FakeQdrant()
    retriever = QdrantHybridRetriever(
        client,
        FakeEmbedder(),
        "coll",
        mode="dense",
        prefetch_k=7,
        rrf_k=2,
        max_concurrent=2,
    )
    await retriever.search_many([Query(text="x", origin=QueryOrigin.INITIAL)], top_k=3)
    call = client.query_calls[0]
    assert (
        "prefetch" not in call
        and call["using"] == DENSE_VECTOR_NAME
        and call["query"] == [0.1, 0.2]
    )
    await retriever.verify_collection(expected_dimension=2)
    with pytest.raises(RetrievalError, match="dimension"):
        await retriever.verify_collection(expected_dimension=2560)


async def test_hydrator_full_section_and_window() -> None:
    client = FakeQdrant()
    hydrator = QdrantHydrator(client, "coll", window=1)
    hit = hit_from_point(PAYLOAD, score=0.9, query_text="q")
    chunks = await hydrator.hydrate(hit, HydrateStrategy.FULL_SECTION)
    assert [c.chunk_index for c in chunks] == [1, 2, 3]
    must = client.scroll_calls[0]["scroll_filter"].must
    assert must[0].key == "section_id" and must[0].match.value == "sec-1"
    await hydrator.hydrate(hit, HydrateStrategy.CHUNK_WINDOW)
    window = client.scroll_calls[1]["scroll_filter"].must[1]
    assert window.key == "chunk_index" and (window.range.gte, window.range.lte) == (
        1,
        3,
    )
    assert await hydrator.hydrate(hit, HydrateStrategy.SEARCH_ONLY) == []
