import asyncio
import json
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION
from pharma_agent.domain.retrieval.models import (
    Hit,
    HydrateStrategy,
    Query,
    RetrievedItem,
)
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.infrastructure.settings import Settings

from pharma_lab.evaluation.backend_retrieval import (
    RetrieveRequest,
    load_query_rows,
    run_retrieval,
    sample_quotas,
)
from pharma_lab.evaluation.cached_query_embedder import (
    CachedQueryEmbedder,
    QueryEmbeddingMissing,
)
from pharma_lab.evaluation.query_embedding_cache import QueryEmbeddingCache
from pharma_lab.evaluation.run_workspace import load_run_record

MODEL = "fake-embedding-4d"
RELEASE = uuid.UUID("11111111-1111-5111-8111-111111111111")
OTHER_RELEASE = uuid.UUID("22222222-2222-5222-8222-222222222222")
COLLECTION = uuid.UUID("33333333-3333-5333-8333-333333333333")
DOSAGE = "drug:paracetamol:lieu-luong-va-cach-dung"
ADULT = "liều paracetamol người lớn"
CHILD = "paracetamol trẻ em"
VECTORS = {ADULT: [0.25, 0.5, 0.75, 1.0], CHILD: [1.0, 0.75, 0.5, 0.25]}


def make_hit(ordinal: int, score: float, release: uuid.UUID = RELEASE) -> Hit:
    return Hit(
        chunk_version_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{DOSAGE}:{ordinal}"),
        release_id=release,
        collection_id=COLLECTION,
        document_key="drug:paracetamol",
        section_key=DOSAGE,
        section_revision_id=uuid.uuid5(uuid.NAMESPACE_URL, DOSAGE),
        ordinal=ordinal,
        hydrate_strategy=HydrateStrategy.FULL_SECTION,
        source="Dược thư Quốc gia Việt Nam",
        title="PARACETAMOL",
        section="Liều lượng và cách dùng",
        start_page=1133,
        end_page=1134,
        context_header="PARACETAMOL\n> Liều lượng và cách dùng",
        chunk_text=f"Đoạn {ordinal}",
        embedding_text=f"PARACETAMOL\n> Liều lượng và cách dùng\n\nĐoạn {ordinal}",
        kind="prose",
        table_key=None,
        colloquial_mapping=None,
        term_annotations=[],
        fusion_score=score,
        rerank_score=None,
        matched_queries=[],
    )


class FakeSearchService:
    def __init__(self, hits: dict[str, list[Hit]]) -> None:
        self.hits = hits
        self.calls: list[str] = []
        self.vectors: list[list[float]] = []
        self.embedder: CachedQueryEmbedder | None = None

    async def search(self, queries: Sequence[Query], rerank_query: str) -> SearchResult:
        self.calls.append(rerank_query)
        if self.embedder is not None:
            self.vectors.extend(await self.embedder.embed([q.text for q in queries]))
        return SearchResult(
            items=[RetrievedItem(hit=hit) for hit in self.hits[rerank_query]],
            queries=list(queries),
        )


@dataclass
class FakeStack:
    service: FakeSearchService
    settings: list[Settings] = field(default_factory=list)
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, stack: FakeStack) -> None:
        self.stack = stack

    def __call__(
        self, settings: Settings, *, embedder: CachedQueryEmbedder | None = None
    ) -> FakeStack:
        self.stack.settings.append(settings)
        self.stack.service.embedder = embedder
        return self.stack


def _evaluation(tmp_path: Path) -> Path:
    path = tmp_path / "evaluation.jsonl"
    rows = [
        {"query_id": "q1", "query": ADULT, "eval_group": "formulary"},
        {"query_id": "q2", "query": CHILD, "eval_group": "leaflet"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _query_cache(tmp_path: Path, rows: dict[str, str]) -> Path:
    path = tmp_path / "query_embeddings.jsonl"
    cache = QueryEmbeddingCache(path, vector_dim=4)
    for query_id, text in rows.items():
        # Each set appends a record with a new created_at, which changes the subset hash
        # in the run identity; write a vector once, like `pharma-lab embed queries` does.
        if cache.get(MODEL, query_id, text) is None:
            cache.set(MODEL, query_id, text, VECTORS[text])
    return path


def _request(
    tmp_path: Path,
    *,
    retriever: str = "hybrid",
    prefetch_k: int | None = 50,
    cache_rows: dict[str, str] | None = None,
    sample: int | None = None,
    sample_seed: int = 0,
) -> RetrieveRequest:
    env_file = tmp_path / "backend.env"
    env_file.write_text(
        f"PHARMA_RETRIEVAL__EMBEDDING__MODEL={MODEL}\n"
        "PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=4\n"
        "PHARMA_LANGFUSE__PUBLIC_KEY=pk\nPHARMA_LANGFUSE__SECRET_KEY=sk\n",
        encoding="utf-8",
    )
    rows = {"q1": ADULT, "q2": CHILD} if cache_rows is None else cache_rows
    return RetrieveRequest(
        evaluation_path=_evaluation(tmp_path),
        run_root=tmp_path / "run",
        retriever=retriever,
        candidate_k=2,
        prefetch_k=prefetch_k,
        rrf_k=2,
        limit=None,
        force=False,
        backend_env_file=env_file,
        query_embeddings=_query_cache(tmp_path, rows),
        sample=sample,
        sample_seed=sample_seed,
    )


def _hits(child_release: uuid.UUID = RELEASE) -> dict[str, list[Hit]]:
    return {
        ADULT: [make_hit(2, 0.9), make_hit(1, 0.5)],
        CHILD: [make_hit(1, 0.7, release=child_release)],
    }


def test_hybrid_retrieval_uses_cached_query_vectors(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    result = run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))

    assert stack.closed
    settings = stack.settings[0]
    assert settings.retrieval.mode == "hybrid"
    assert (settings.retrieval.candidate_k, settings.retrieval.prefetch_k) == (2, 50)
    assert settings.retrieval.collections == ["formulary"]
    assert settings.retrieval.rerank.protocol == "none"
    assert settings.retrieval.rerank.top_n == 2
    assert settings.retrieval.rerank.max_candidates == 2
    assert settings.langfuse.public_key is None
    assert stack.service.vectors == [VECTORS[ADULT], VECTORS[ADULT], VECTORS[CHILD]]
    assert result.artifact.query_count == 2
    first = json.loads(result.artifact.data_path.read_text("utf-8").splitlines()[0])
    assert [c["chunk_id"] for c in first["candidates"]] == [
        f"{DOSAGE}:chunk-002",
        f"{DOSAGE}:chunk-001",
    ]
    assert first["candidates"][0]["payload"] == {
        "chunk_id": f"{DOSAGE}:chunk-002",
        "section_id": DOSAGE,
        "chunk_index": 2,
    }
    assert first["candidates"][0]["document_text"] == make_hit(2, 0.9).embedding_text
    identity = load_run_record(tmp_path / "run" / "run.json").identity
    assert identity.release_id == str(RELEASE)
    assert identity.chunker_version == CHUNKER_VERSION
    assert identity.embedding_model == MODEL
    assert identity.collection_name == settings.retrieval.qdrant_collection
    assert identity.query_embeddings_sha256 is not None
    assert (identity.limit, identity.sample, identity.sample_seed) == (None, None, None)
    record = load_run_record(tmp_path / "run" / "run.json")
    assert record.origin == "backend"
    assert record.identity.evaluation_path == str(
        (tmp_path / "evaluation.jsonl").resolve()
    )


def test_existing_complete_candidates_are_reused(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))
    run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))
    assert len(stack.service.calls) == 3

    run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))

    assert len(stack.service.calls) == 4


def test_hits_from_another_release_stop_the_run(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits(child_release=OTHER_RELEASE)))

    with pytest.raises(RuntimeError, match="pinned to release"):
        run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))
    assert stack.closed


def test_bm25_needs_no_query_vectors(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    run_retrieval(
        _request(tmp_path, retriever="bm25", prefetch_k=None, cache_rows={}),
        backend_factory=FakeFactory(stack),
    )

    assert stack.settings[0].retrieval.mode == "bm25"
    assert stack.service.embedder is None
    identity = load_run_record(tmp_path / "run" / "run.json").identity
    assert (identity.retriever, identity.query_embeddings_sha256) == ("bm25", None)


def test_missing_query_vectors_stop_before_the_backend_starts(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    with pytest.raises(QueryEmbeddingMissing, match="missing 1 of 2"):
        run_retrieval(
            _request(tmp_path, cache_rows={"q1": ADULT}),
            backend_factory=FakeFactory(stack),
        )
    assert stack.settings == []


def test_cached_embedder_rejects_unknown_query_text(tmp_path: Path) -> None:
    embedder = CachedQueryEmbedder(
        {"0" * 64: [0.1, 0.2, 0.3, 0.4]},
        model=MODEL,
        dimension=4,
        source=tmp_path / "cache.jsonl",
    )

    with pytest.raises(QueryEmbeddingMissing, match="pharma-lab embed queries"):
        asyncio.run(embedder.embed(["câu hỏi chưa embed"]))


def test_unknown_retriever_and_dense_prefetch_are_rejected(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    with pytest.raises(ValueError, match="bm25, dense or hybrid"):
        run_retrieval(
            _request(tmp_path, retriever="sparse"), backend_factory=FakeFactory(stack)
        )
    with pytest.raises(ValueError, match="--prefetch-k"):
        run_retrieval(
            _request(tmp_path, retriever="dense", prefetch_k=10),
            backend_factory=FakeFactory(stack),
        )


GOLD_GROUP_SIZES = {
    "patient_natural": 500,
    "leaflet": 2500,
    "chunk_risk": 1000,
    "formulary": 5000,
    "noisy_confuser": 500,
    "multi_intent": 500,
}


def _gold(tmp_path: Path, sizes: dict[str, int]) -> Path:
    path = tmp_path / "gold.jsonl"
    rows = [
        {
            "query_id": f"{group}-{number:05d}",
            "query": f"câu hỏi {group} {number}",
            "eval_group": group,
        }
        for group, size in sizes.items()
        for number in range(size)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_sample_takes_ten_percent_of_every_eval_group(tmp_path: Path) -> None:
    rows = load_query_rows(_gold(tmp_path, GOLD_GROUP_SIZES), sample=1000)

    assert Counter(row["eval_group"] for row in rows) == {
        "patient_natural": 50,
        "leaflet": 250,
        "chunk_risk": 100,
        "formulary": 500,
        "noisy_confuser": 50,
        "multi_intent": 50,
    }


def test_sample_rounds_by_largest_remainder_then_file_order() -> None:
    assert sample_quotas({"a": 5, "b": 3, "c": 2}, 3) == {"a": 1, "b": 1, "c": 1}
    assert sample_quotas({"a": 1, "b": 1}, 1) == {"a": 1, "b": 0}


def test_sample_is_repeatable_per_seed_and_keeps_file_order(tmp_path: Path) -> None:
    path = _gold(tmp_path, {"formulary": 100, "leaflet": 100})
    position = {
        row["query_id"]: index for index, row in enumerate(load_query_rows(path))
    }

    first = [row["query_id"] for row in load_query_rows(path, sample=20, sample_seed=0)]
    again = [row["query_id"] for row in load_query_rows(path, sample=20, sample_seed=0)]
    other = [row["query_id"] for row in load_query_rows(path, sample=20, sample_seed=1)]

    assert first == again
    assert first != other
    assert [position[query_id] for query_id in first] == sorted(
        position[query_id] for query_id in first
    )


def test_sample_rejects_limit_oversize_and_rows_without_a_group(tmp_path: Path) -> None:
    path = _gold(tmp_path, {"formulary": 3})
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"query_id": "q1", "query": "câu hỏi"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_query_rows(path, 2, sample=2)
    with pytest.raises(ValueError, match="exceeds the 3 evaluation rows"):
        load_query_rows(path, sample=4)
    with pytest.raises(ValueError, match="eval_group on every row"):
        load_query_rows(plain, sample=1)


def test_sampled_run_records_the_sample_in_its_identity(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    result = run_retrieval(
        _request(tmp_path, sample=1, sample_seed=7), backend_factory=FakeFactory(stack)
    )

    assert result.artifact.query_count == 1
    identity = load_run_record(tmp_path / "run" / "run.json").identity
    assert (identity.limit, identity.sample, identity.sample_seed) == (None, 1, 7)
