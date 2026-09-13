from pathlib import Path

import pytest

from seed_pipeline.evaluation.artifact_contracts import sha256_file
from seed_pipeline.evaluation.rerank_contract import prompt_contract_hash
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
    build_candidate_artifact,
)
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace
from seed_pipeline.runtime.catalog import require_model


class OneCandidateRetriever:
    def search(self, query: str, *, limit: int):
        del query, limit
        return [
            RetrievalCandidate(
                chunk_id="chunk-1",
                score=0.75,
                rank=1,
                source="test",
                payload={"chunk_id": "chunk-1", "section_id": "section-1"},
                document_text="candidate document",
            )
        ]


@pytest.fixture
def complete_run(tmp_path: Path) -> Path:
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_path.write_text(
        '{"query_id":"query-1","query":"test query","relevant_section_ids":["section-1"]}\n',
        encoding="utf-8",
    )
    identity = RunIdentity(
        evaluation_path=str(evaluation_path),
        evaluation_sha256=sha256_file(evaluation_path),
        collection_name="collection",
        embedding_model="embeddinggemma:300m",
        query_embeddings_sha256="query-cache",
        retriever="hybrid",
        candidate_k=1,
        rrf_k=2,
        limit=None,
        prefetch_k=1,
    )
    root = tmp_path / "run"
    workspace = RunWorkspace.open_or_create(root, identity)
    artifact = build_candidate_artifact(
        rows=[{"query_id": "query-1", "query": "test query"}],
        retriever=OneCandidateRetriever(),
        output_path=root / "candidates" / "candidates.jsonl",
        identity={"evaluation_sha256": identity.evaluation_sha256},
        candidate_k=1,
    )
    workspace.record_candidates(artifact)
    return root


@pytest.fixture
def candidate_bundle(complete_run: Path):
    from seed_pipeline.artifacts.bundle import load_bundle

    return load_bundle(
        complete_run / "candidates",
        expected_type="retrieval_candidates",
        require_complete=True,
    )


@pytest.fixture
def complete_rerank_cache(tmp_path: Path, candidate_bundle):
    model = "qwen3-reranker:0.6b-fp16"
    spec = require_model(model)
    contract = prompt_contract_hash(protocol=spec.reranker_protocol or "")
    cache = RerankScoreCache(
        tmp_path / "rerank-cache.jsonl",
        model_sha256=spec.sha256,
        request_contract_sha256=contract,
    )
    row = next(iter(CandidateArtifactReader.from_data_path(candidate_bundle.data_path)))
    item = row["candidates"][0]
    candidate = RetrievalCandidate(
        chunk_id=item["chunk_id"],
        score=float(item["retrieval_score"]),
        rank=int(item["retrieval_rank"]),
        source=str(item["source"]),
        payload=dict(item["payload"]),
        document_text=str(item["document_text"]),
        document_hash=str(item["document_hash"]),
    )
    cache.set(
        model,
        {"query_id": row["query_id"], "query": row["query"]},
        candidate,
        0.9,
        protocol=spec.reranker_protocol,
    )
    return cache
