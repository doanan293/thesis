from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pharma_lab.cache.jsonl_records import (
    CacheRecordError,
    ValidatedSubset,
    append_record,
    load_records,
    rewrite_records,
)
from pharma_lab.cache.jsonl_records import (
    subset_sha256 as record_subset_sha256,
)
from pharma_lab.evaluation.artifact_contracts import (
    ArtifactContractError,
    require_finite_number,
)
from pharma_lab.evaluation.query_hash import query_hash
from pharma_lab.evaluation.rerank_contract import document_hash
from pharma_lab.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from pharma_lab.evaluation.retrieval_types import RetrievalCandidate
from pharma_lab.runtime.catalog import require_model


class RerankScoreCacheError(ArtifactContractError):
    """Raised when a rerank score cache cannot satisfy its candidate contract."""


@dataclass(frozen=True, order=True)
class RerankKey:
    reranker: str
    model_sha256: str
    request_contract_sha256: str
    query_id: str
    query_hash: str
    chunk_id: str
    document_hash: str


def _candidate_document_hash(candidate: RetrievalCandidate) -> str:
    if candidate.document_hash:
        return candidate.document_hash
    if candidate.document_text:
        return document_hash(candidate.document_text)
    from pharma_lab.evaluation.candidate_text import candidate_document_text

    return document_hash(candidate_document_text(candidate.payload))


@dataclass
class RerankScoreCache:
    path: Path
    model_sha256: str = ""
    request_contract_sha256: str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.model_sha256 = self.model_sha256
        self.request_contract_sha256 = self.request_contract_sha256
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
            reranker,
            self.model_sha256,
            self.request_contract_sha256,
            query_id,
            query_hash(query),
            chunk_id,
            _candidate_document_hash(candidate),
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            records = load_records(self.path)
        except CacheRecordError as exc:
            raise RerankScoreCacheError(str(exc)) from exc
        for line_number, record in enumerate(records, start=1):
            try:
                raw_contract = record.get("request_contract_sha256")
                if not isinstance(raw_contract, str) or not raw_contract:
                    raise RerankScoreCacheError(
                        f"Missing request contract in {self.path}:{line_number}"
                    )
                contract = raw_contract
                model_sha = str(record.get("model_sha256") or "")
                if self.model_sha256 and model_sha not in ("", self.model_sha256):
                    raise RerankScoreCacheError(
                        f"Model digest mismatch in {self.path}:{line_number}"
                    )
                if self.request_contract_sha256 and contract not in (
                    "",
                    self.request_contract_sha256,
                ):
                    raise RerankScoreCacheError(
                        f"Prompt contract mismatch in {self.path}:{line_number}"
                    )
                key = RerankKey(
                    str(record["reranker"]),
                    self.model_sha256 or model_sha,
                    self.request_contract_sha256 or contract,
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
            normalized = {
                **record,
                "reranker": key.reranker,
                "model_sha256": key.model_sha256,
                "request_contract_sha256": key.request_contract_sha256,
                "query_id": key.query_id,
                "query_hash": key.query_hash,
                "chunk_id": key.chunk_id,
                "document_hash": key.document_hash,
                "score": score,
            }
            self.records[key] = score
            self.record_metadata[key] = normalized

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
        contract = self.request_contract_sha256
        if not contract and protocol:
            spec = require_model(reranker)
            if spec.rerank_contract is None:
                raise RerankScoreCacheError(
                    f"Reranker {reranker} has no scoring contract"
                )
            contract = spec.rerank_contract.sha256
            key = RerankKey(
                key.reranker,
                key.model_sha256,
                contract,
                key.query_id,
                key.query_hash,
                key.chunk_id,
                key.document_hash,
            )
        record = {
            "reranker": key.reranker,
            "model_sha256": key.model_sha256,
            "request_contract_sha256": contract,
            "query_id": key.query_id,
            "query_hash": key.query_hash,
            "chunk_id": key.chunk_id,
            "document_hash": key.document_hash,
            "score": score,
        }
        if protocol is not None:
            record["protocol"] = protocol
        record = append_record(self.path, record, schema="rerank-score-v2")
        self.records[key] = score
        self.record_metadata[key] = record
        return key

    def validate_subset(
        self, candidate_data_path: Path, reranker: str
    ) -> ValidatedSubset:
        expected = self.expected_keys_from_candidates(candidate_data_path, reranker)
        found = self._stored(expected)
        missing = len(expected) - len(found)
        return ValidatedSubset(
            total=len(expected),
            complete=len(found),
            missing=missing,
            sha256=record_subset_sha256(found) if not missing else None,
        )

    def available_records(
        self, candidate_data_path: Path, reranker: str
    ) -> list[dict[str, Any]]:
        """Stored records of the candidate pairs that already have a score."""
        return self._stored(
            self.expected_keys_from_candidates(candidate_data_path, reranker)
        )

    def _stored(self, expected: builtins.set[RerankKey]) -> list[dict[str, Any]]:
        return [
            self.record_metadata[key]
            for key in sorted(expected)
            if key in self.record_metadata
        ]

    def subset_records(
        self, candidate_data_path: Path, reranker: str
    ) -> list[dict[str, Any]]:
        expected = self.expected_keys_from_candidates(candidate_data_path, reranker)
        missing = expected - set(self.record_metadata)
        if missing:
            raise RerankScoreCacheError(
                f"Rerank score cache is missing {len(missing)} records"
            )
        return [self.record_metadata[key] for key in sorted(expected)]

    def replace_keys(self, keys: builtins.set[RerankKey]) -> None:
        kept = [
            record for key, record in self.record_metadata.items() if key not in keys
        ]
        rewrite_records(
            self.path, sorted(kept, key=lambda item: json.dumps(item, sort_keys=True))
        )
        self.records.clear()
        self.record_metadata.clear()
        self._load()

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
    ) -> builtins.set[RerankKey]:
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
