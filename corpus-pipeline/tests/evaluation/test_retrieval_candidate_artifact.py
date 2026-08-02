import json
from pathlib import Path

import pytest

from corpus_pipeline.evaluation.artifact_contracts import ArtifactContractError
from corpus_pipeline.evaluation.query_embedding_cache import query_hash
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
    build_candidate_artifact,
    candidate_document_text,
    document_hash,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate


def test_builder_freezes_exact_document_text(tmp_path: Path):
    row = {"query_id": "q1", "query": "liều paracetamol"}
    candidate = RetrievalCandidate(
        chunk_id="c1",
        score=0.7,
        rank=1,
        source="hybrid",
        payload={"context_header": "Thuốc", "chunk_text": "Liều 500 mg"},
    )

    class FakeRetriever:
        accepts_query_row = True

        def search(self, query_row, limit):
            assert query_row == row
            assert limit == 50
            return [candidate]

    artifact = build_candidate_artifact(
        rows=[row],
        retriever=FakeRetriever(),
        output_path=tmp_path / "candidates.jsonl",
        identity={"eval_sha256": "a" * 64},
        candidate_k=50,
    )

    record = next(CandidateArtifactReader(artifact.data_path, artifact.manifest_path))
    frozen = record["candidates"][0]
    assert frozen["document_text"] == candidate_document_text(candidate.payload)
    assert frozen["document_hash"] == document_hash(frozen["document_text"])
    assert artifact.pair_count == 1


def test_reader_rejects_duplicate_chunk_ids(tmp_path: Path):
    data = tmp_path / "candidates.jsonl"
    data.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "q",
                "query_hash": query_hash("q"),
                "candidates": [
                    {
                        "chunk_id": "c1",
                        "retrieval_score": 0.1,
                        "retrieval_rank": 1,
                        "source": "hybrid",
                        "document_text": "one",
                        "document_hash": document_hash("one"),
                        "payload": {},
                    },
                    {
                        "chunk_id": "c1",
                        "retrieval_score": 0.2,
                        "retrieval_rank": 2,
                        "source": "hybrid",
                        "document_text": "two",
                        "document_hash": document_hash("two"),
                        "payload": {},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"artifact_type":"retrieval_candidates",'
        '"created_at":"now","data_filename":"candidates.jsonl",'
        '"data_sha256":"ignored","record_count":1,"identity":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactContractError, match="duplicate chunk_id"):
        list(CandidateArtifactReader(data, manifest))


def test_builder_rejects_candidate_k_overflow(tmp_path: Path):
    class EmptyRetriever:
        def search(self, query, limit):
            return []

    row = {"query_id": "q1", "query": "query"}
    with pytest.raises(ValueError, match="candidate_k"):
        build_candidate_artifact(
            rows=[row],
            retriever=EmptyRetriever(),
            output_path=tmp_path / "candidates.jsonl",
            identity={},
            candidate_k=0,
        )
