import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Literal

import pytest
from qdrant_client import models

from pharma_agent.domain.retrieval.models import (
    Chunk,
    ChunkRecord,
    HydrateStrategy,
    Query,
    QueryOrigin,
)
from pharma_agent.domain.retrieval.ports import ChunkKey, RetrievalError
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    BM25_MODEL_NAME,
    BM25_SPARSE_VECTOR_NAME,
    COLLECTION_ID_KEY,
    DENSE_VECTOR_NAME,
    RELEASE_IDS_KEY,
    OpenAiEmbedder,
    QdrantHybridRetriever,
    release_scope_filter,
)
from tests.domain.factories import COLLECTION_ID, RELEASE_ID, chunk_uuid, make_hit
from tests.fakes import FakeEmbedder, fake_vector

OTHER_COLLECTION = uuid.uuid5(uuid.NAMESPACE_URL, "other-collection")
OTHER_RELEASE = uuid.uuid5(uuid.NAMESPACE_URL, "other-release")


def record(
    label: str,
    *,
    release_id: uuid.UUID = RELEASE_ID,
    collection_id: uuid.UUID = COLLECTION_ID,
) -> ChunkRecord:
    hit = make_hit(label, strategy=HydrateStrategy.FULL_SECTION, text=f"text {label}")
    fields = hit.model_dump(exclude={"fusion_score", "rerank_score", "matched_queries"})
    return ChunkRecord.model_validate(
        {**fields, "release_id": release_id, "collection_id": collection_id}
    )


class FakeReader:
    def __init__(
        self, releases: dict[uuid.UUID, uuid.UUID], records: Sequence[ChunkRecord]
    ) -> None:
        self.releases = releases
        self.records = list(records)
        self.release_calls: list[list[str]] = []
        self.loaded: list[list[ChunkKey]] = []

    async def current_releases(
        self, collection_keys: Sequence[str]
    ) -> dict[uuid.UUID, uuid.UUID]:
        self.release_calls.append(list(collection_keys))
        return dict(self.releases)

    async def load_chunks(self, keys: Sequence[ChunkKey]) -> list[ChunkRecord]:
        self.loaded.append(list(keys))
        wanted = set(keys)
        return [r for r in self.records if (r.release_id, r.chunk_version_id) in wanted]

    async def section_chunks(
        self,
        release_id: uuid.UUID,
        section_revision_id: uuid.UUID,
        *,
        around: int | None,
        radius: int,
    ) -> list[Chunk]:
        return []


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


def point(
    label: str, score: float, collection_id: uuid.UUID = COLLECTION_ID
) -> SimpleNamespace:
    return SimpleNamespace(
        id=str(chunk_uuid(label)),
        score=score,
        payload={COLLECTION_ID_KEY: str(collection_id)},
    )


class FakeQdrant:
    def __init__(
        self, points: list[SimpleNamespace], metadata: dict | None = None
    ) -> None:
        self.points = points
        self.metadata = (
            metadata if metadata is not None else {"embedding_model": "m", "dims": 2}
        )
        self.query_calls: list[dict] = []

    async def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.points)

    async def get_collection(self, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={DENSE_VECTOR_NAME: SimpleNamespace(size=2)}
                ),
                metadata=self.metadata,
            )
        )


def retriever(
    client: FakeQdrant,
    reader: FakeReader,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    embedder: FakeEmbedder | None = None,
) -> QdrantHybridRetriever:
    return QdrantHybridRetriever(
        client,
        embedder if embedder is not None else FakeEmbedder(),
        reader,
        collection="chunks_current",
        scope=["formulary"],
        mode=mode,
        prefetch_k=7,
        rrf_k=2,
        max_concurrent=2,
    )


def test_release_scope_filter_matches_each_collection_with_its_release() -> None:
    scope = release_scope_filter(
        {COLLECTION_ID: RELEASE_ID, OTHER_COLLECTION: OTHER_RELEASE}
    )
    assert scope.must is None
    branches = scope.should
    assert isinstance(branches, list) and len(branches) == 2
    first = branches[0]
    assert isinstance(first, models.Filter) and isinstance(first.must, list)
    collection, release = first.must
    assert isinstance(collection, models.FieldCondition)
    assert isinstance(release, models.FieldCondition)
    assert (collection.key, release.key) == (COLLECTION_ID_KEY, RELEASE_IDS_KEY)
    assert isinstance(collection.match, models.MatchValue)
    assert collection.match.value == str(COLLECTION_ID)
    assert isinstance(release.match, models.MatchValue)
    assert release.match.value == str(RELEASE_ID)


async def test_hybrid_search_filters_by_release_and_loads_chunks_in_rank_order() -> (
    None
):
    client = FakeQdrant(
        [
            point("c2", 0.9),
            point("c1", 0.5),
            point("gone", 0.4),
            point("stray", 0.3, OTHER_COLLECTION),
        ]
    )
    reader = FakeReader({COLLECTION_ID: RELEASE_ID}, [record("c1"), record("c2")])

    hits = await retriever(client, reader).search_many(
        [Query(text="paracetamol", origin=QueryOrigin.INITIAL)], top_k=5
    )

    assert reader.release_calls == [["formulary"]]
    assert reader.loaded == [
        [
            (RELEASE_ID, chunk_uuid("c2")),
            (RELEASE_ID, chunk_uuid("c1")),
            (RELEASE_ID, chunk_uuid("gone")),
        ]
    ]
    assert [(h.chunk_version_id, h.fusion_score) for h in hits[0]] == [
        (chunk_uuid("c2"), 0.9),
        (chunk_uuid("c1"), 0.5),
    ]
    top = hits[0][0]
    assert top.release_id == RELEASE_ID and top.matched_queries == ["paracetamol"]
    assert top.chunk_text == "text c2"
    call = client.query_calls[0]
    assert call["collection_name"] == "chunks_current" and call["limit"] == 5
    assert call["with_payload"] == [COLLECTION_ID_KEY]
    expected_scope = release_scope_filter({COLLECTION_ID: RELEASE_ID})
    assert call["query_filter"] == expected_scope
    dense, sparse = call["prefetch"]
    assert (dense.using, dense.limit, dense.query) == (
        DENSE_VECTOR_NAME,
        7,
        fake_vector("paracetamol"),
    )
    assert dense.filter == expected_scope and sparse.filter == expected_scope
    assert sparse.using == BM25_SPARSE_VECTOR_NAME
    assert isinstance(sparse.query, models.Document)
    assert sparse.query.text == "paracetamol" and sparse.query.model == "Qdrant/bm25"
    assert isinstance(call["query"], models.RrfQuery) and call["query"].rrf.k == 2


async def test_dense_mode_uses_the_same_release_filter() -> None:
    client = FakeQdrant([point("c1", 0.8)])
    reader = FakeReader({COLLECTION_ID: RELEASE_ID}, [record("c1")])
    await retriever(client, reader, mode="dense").search_many(
        [Query(text="x", origin=QueryOrigin.INITIAL)], top_k=3
    )
    call = client.query_calls[0]
    assert "prefetch" not in call
    assert (call["using"], call["query"]) == (DENSE_VECTOR_NAME, fake_vector("x"))
    assert call["query_filter"] == release_scope_filter({COLLECTION_ID: RELEASE_ID})


async def test_bm25_mode_runs_one_sparse_query_without_embedding() -> None:
    client = FakeQdrant([point("c1", 0.8)])
    embedder = FakeEmbedder()
    reader = FakeReader({COLLECTION_ID: RELEASE_ID}, [record("c1")])

    hits = await retriever(client, reader, mode="bm25", embedder=embedder).search_many(
        [Query(text="liều trẻ em", origin=QueryOrigin.INITIAL)], top_k=3
    )

    assert embedder.batches == []
    assert [h.chunk_version_id for h in hits[0]] == [chunk_uuid("c1")]
    call = client.query_calls[0]
    assert "prefetch" not in call and call["using"] == BM25_SPARSE_VECTOR_NAME
    assert isinstance(call["query"], models.Document)
    assert (call["query"].text, call["query"].model) == ("liều trẻ em", BM25_MODEL_NAME)
    assert call["query_filter"] == release_scope_filter({COLLECTION_ID: RELEASE_ID})
    assert call["with_payload"] == [COLLECTION_ID_KEY]


async def test_search_without_a_current_release_fails_before_querying_qdrant() -> None:
    client = FakeQdrant([point("c1", 0.8)])
    with pytest.raises(RetrievalError, match="no current release"):
        await retriever(client, FakeReader({}, [])).search_many(
            [Query(text="x", origin=QueryOrigin.INITIAL)], top_k=3
        )
    assert client.query_calls == []


async def test_malformed_point_is_reported_as_retrieval_error() -> None:
    client = FakeQdrant([SimpleNamespace(id="not-a-uuid", score=0.1, payload={})])
    with pytest.raises(RetrievalError, match="qdrant query failed"):
        await retriever(
            client, FakeReader({COLLECTION_ID: RELEASE_ID}, [])
        ).search_many([Query(text="x", origin=QueryOrigin.INITIAL)], top_k=1)


async def test_verify_collection_checks_dimension_and_metadata() -> None:
    reader = FakeReader({}, [])
    await retriever(FakeQdrant([]), reader).verify_collection(
        embedding_model="m", dimension=2
    )
    with pytest.raises(RetrievalError, match="dimension"):
        await retriever(FakeQdrant([]), reader).verify_collection(
            embedding_model="m", dimension=2560
        )
    with pytest.raises(RetrievalError, match="metadata"):
        await retriever(
            FakeQdrant([], metadata={"embedding_model": "other", "dims": 2}), reader
        ).verify_collection(embedding_model="m", dimension=2)
