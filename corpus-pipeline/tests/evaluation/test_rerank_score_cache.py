from pathlib import Path

import pytest

from corpus_pipeline.evaluation.query_embedding_cache import query_hash
from corpus_pipeline.evaluation.rerank_score_cache import (
    RerankScoreCache,
    RerankScoreCacheError,
    finalize_rerank_cache,
)
from corpus_pipeline.evaluation.rerankers import CachedReranker
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    build_candidate_artifact,
    document_hash,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate

RERANKER = "bge-reranker-v2-m3:f16"


def candidate(chunk_id: str, rank: int) -> RetrievalCandidate:
    text = f"document {chunk_id}"
    return RetrievalCandidate(
        chunk_id=chunk_id,
        score=0.1,
        rank=rank,
        source="hybrid",
        payload={"chunk_id": chunk_id},
        document_text=text,
        document_hash=document_hash(text),
    )


def test_cached_reranker_uses_score_then_original_rank_then_chunk_id(tmp_path: Path):
    cache_path = tmp_path / "scores.jsonl"
    cache = RerankScoreCache(cache_path)
    query_row = {"query_id": "q1", "query": "query"}
    cache.set(RERANKER, query_row, candidate("c2", 1), 0.8)
    cache.set(RERANKER, query_row, candidate("c1", 1), 0.8)
    cache.set(RERANKER, query_row, candidate("c3", 3), 0.2)

    ranked = CachedReranker(cache, reranker=RERANKER).rerank(
        query_row,
        [candidate("c2", 1), candidate("c1", 1), candidate("c3", 3)],
    )
    assert [item.chunk_id for item in ranked] == ["c1", "c2", "c3"]


def test_score_cache_rejects_non_finite_score(tmp_path: Path):
    cache = RerankScoreCache(tmp_path / "scores.jsonl")
    with pytest.raises(RerankScoreCacheError, match="finite"):
        cache.set(
            RERANKER, {"query_id": "q1", "query": "q"}, candidate("c1", 1), float("nan")
        )


def test_finalize_rejects_missing_pair(tmp_path: Path):
    rows = [{"query_id": "q1", "query": "query"}]
    candidates = build_candidate_artifact(
        rows=rows,
        retriever=type(
            "Retriever",
            (),
            {
                "search": lambda self, query, limit: [
                    candidate("c1", 1),
                    candidate("c2", 2),
                ]
            },
        )(),
        output_path=tmp_path / "candidates.jsonl",
        identity={"eval_sha256": "a" * 64},
        candidate_k=2,
    )
    partial = tmp_path / "partial.jsonl"
    partial_cache = RerankScoreCache(partial)
    partial_cache.set(RERANKER, rows[0], candidate("c1", 1), 0.8)
    with pytest.raises(RerankScoreCacheError, match="missing 1"):
        finalize_rerank_cache(
            candidate_data_path=candidates.data_path,
            candidate_manifest_path=candidates.manifest_path,
            partial_cache_path=partial,
            output_dir=tmp_path / "final",
            reranker=RERANKER,
            gguf_sha256="b" * 64,
            protocol="native_rerank",
            request_contract_sha256="c" * 64,
        )


def test_score_key_contains_query_and_document_hash(tmp_path: Path):
    cache = RerankScoreCache(tmp_path / "scores.jsonl")
    row = {"query_id": "q1", "query": "query"}
    item = candidate("c1", 1)
    key = cache.key_for(RERANKER, row, item)
    assert key.query_hash == query_hash("query")
    assert key.document_hash == document_hash(item.document_text or "")
