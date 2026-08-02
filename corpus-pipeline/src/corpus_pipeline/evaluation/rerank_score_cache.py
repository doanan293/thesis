from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_pipeline.config.paths import RERANK_SCORE_CACHE_DIR
from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    Completion,
    canonical_sha256,
    iter_jsonl_objects,
    require_finite_number,
    write_json,
)
from corpus_pipeline.evaluation.query_embedding_cache import model_slug, query_hash
from corpus_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
    document_hash,
)
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate


class RerankScoreCacheError(ArtifactContractError):
    """Raised when a rerank score cache cannot satisfy its candidate contract."""


def default_rerank_score_cache_path(eval_path: Path, reranker_name: str) -> Path:
    return (
        RERANK_SCORE_CACHE_DIR
        / model_slug(reranker_name)
        / f"{Path(eval_path).stem}.jsonl"
    )


@dataclass(frozen=True, order=True)
class RerankKey:
    reranker: str
    query_id: str
    query_hash: str
    chunk_id: str
    document_hash: str


def prompt_contract_hash(
    *,
    protocol: str,
    instruction: str = "",
    template_version: str = "rerank-prompt-v1",
) -> str:
    return canonical_sha256(
        {
            "protocol": protocol,
            "instruction": instruction,
            "template_version": template_version,
        }
    )


def _candidate_document_hash(candidate: RetrievalCandidate) -> str:
    if candidate.document_hash:
        return candidate.document_hash
    if candidate.document_text:
        return document_hash(candidate.document_text)
    from corpus_pipeline.evaluation.retrievers import candidate_document_text

    return document_hash(candidate_document_text(candidate.payload))


@dataclass
class RerankScoreCache:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.records: dict[RerankKey, float] = {}
        self.record_metadata: dict[RerankKey, dict[str, Any]] = {}
        self._load()

    def key_for(
        self,
        reranker: str,
        query_row: dict[str, Any],
        candidate: RetrievalCandidate,
    ) -> RerankKey:
        query_id = str(query_row.get("query_id") or "")
        query = str(query_row.get("query") or "")
        chunk_id = candidate.resolved_chunk_id
        if not query_id or not query:
            raise RerankScoreCacheError("Rerank key requires query_id and query")
        if not chunk_id:
            raise RerankScoreCacheError("Rerank key requires chunk_id")
        return RerankKey(
            str(reranker),
            query_id,
            query_hash(query),
            chunk_id,
            _candidate_document_hash(candidate),
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_number, record in enumerate(iter_jsonl_objects(self.path), start=1):
            try:
                key = RerankKey(
                    str(record["reranker"]),
                    str(record["query_id"]),
                    str(record["query_hash"]),
                    str(record["chunk_id"]),
                    str(record["document_hash"]),
                )
                score = require_finite_number(record["score"], "score")
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankScoreCacheError(
                    f"Invalid rerank score record at {self.path}:{line_number}"
                ) from exc
            self.records[key] = score
            self.record_metadata[key] = {
                **record,
                "score": score,
            }

    def set(
        self,
        reranker: str,
        query_row: dict[str, Any],
        candidate: RetrievalCandidate,
        score: float,
        *,
        protocol: str | None = None,
    ) -> RerankKey:
        try:
            score = require_finite_number(score, "score")
        except ArtifactContractError as exc:
            raise RerankScoreCacheError(str(exc)) from exc
        key = self.key_for(reranker, query_row, candidate)
        record = {
            "reranker": key.reranker,
            "query_id": key.query_id,
            "query_hash": key.query_hash,
            "chunk_id": key.chunk_id,
            "document_hash": key.document_hash,
            "score": score,
        }
        if protocol is not None:
            record["protocol"] = str(protocol)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.records[key] = score
        self.record_metadata[key] = record
        return key

    def require_score(
        self,
        reranker: str,
        query_row: dict[str, Any],
        candidate: RetrievalCandidate,
    ) -> float:
        key = self.key_for(reranker, query_row, candidate)
        try:
            return self.records[key]
        except KeyError as exc:
            raise RerankScoreCacheError(
                f"Missing rerank score for {key.query_id}/{key.chunk_id}"
            ) from exc

    def expected_keys_from_candidates(
        self,
        candidate_data_path: Path,
        reranker: str,
    ) -> set[RerankKey]:
        expected: set[RerankKey] = set()
        for record in CandidateArtifactReader.from_data_path(candidate_data_path):
            query_row = {"query_id": record["query_id"], "query": record["query"]}
            for item in record["candidates"]:
                candidate = RetrievalCandidate(
                    chunk_id=item["chunk_id"],
                    score=float(item["retrieval_score"]),
                    rank=int(item["retrieval_rank"]),
                    source=str(item["source"]),
                    payload=dict(item.get("payload") or {}),
                    document_text=str(item["document_text"]),
                    document_hash=str(item["document_hash"]),
                )
                expected.add(self.key_for(reranker, query_row, candidate))
        return expected


def finalize_rerank_cache(
    *,
    candidate_data_path: Path,
    candidate_manifest_path: Path,
    partial_cache_path: Path,
    output_dir: Path,
    reranker: str,
    gguf_sha256: str,
    protocol: str,
    request_contract_sha256: str,
    require_complete: bool = True,
    job_sha256: str | None = None,
) -> tuple[Path, Path, Completion]:
    reader = CandidateArtifactReader(candidate_data_path, candidate_manifest_path)
    list(reader)
    expected_cache = RerankScoreCache(partial_cache_path)
    expected = expected_cache.expected_keys_from_candidates(
        candidate_data_path, reranker
    )
    missing = expected - set(expected_cache.records)
    unexpected = set(expected_cache.records) - expected
    if unexpected:
        raise RerankScoreCacheError(
            f"Rerank score cache contains {len(unexpected)} unexpected keys"
        )
    completion = Completion(len(expected), len(expected) - len(missing), len(missing))
    if missing and require_complete:
        raise RerankScoreCacheError(
            f"Rerank score cache is missing {len(missing)} records"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "rerank_scores.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for key in sorted(expected & set(expected_cache.records)):
            handle.write(
                json.dumps(
                    expected_cache.record_metadata[key],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    identity = {
        "candidate_data_sha256": reader.manifest.data_sha256,
        "reranker": reranker,
        "gguf_sha256": gguf_sha256,
        "protocol": protocol,
        "request_contract_sha256": request_contract_sha256,
        "pair_count": len(expected),
    }
    if job_sha256 is not None:
        identity["job_sha256"] = str(job_sha256)
    manifest = ArtifactManifest.create(
        artifact_type="rerank_score_cache",
        data_path=data_path,
        record_count=completion.complete,
        identity=identity,
    )
    manifest_payload = manifest.to_dict()
    manifest_payload.update(
        total=completion.total,
        complete=completion.complete,
        missing=completion.missing,
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest_payload)
    return data_path, manifest_path, completion
