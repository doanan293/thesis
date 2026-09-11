from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    iter_jsonl_objects,
    require_finite_number,
    write_json,
)
from corpus_pipeline.evaluation.query_hash import query_hash
from corpus_pipeline.evaluation.rerank_contract import document_hash
from corpus_pipeline.evaluation.retrieval_types import RetrievalCandidate
from corpus_pipeline.evaluation.retrievers import candidate_document_text

_RETRIEVAL_BATCH_SIZE = 64


@dataclass(frozen=True)
class CandidateArtifact:
    data_path: Path
    manifest_path: Path
    manifest: ArtifactManifest
    query_count: int
    pair_count: int


def _serialize_candidate(candidate: RetrievalCandidate) -> dict[str, Any]:
    text = candidate.document_text or candidate_document_text(candidate.payload)
    chunk_id = candidate.resolved_chunk_id
    if not chunk_id or not text:
        raise ArtifactContractError("Candidate requires chunk_id and document_text")
    return {
        "chunk_id": chunk_id,
        "retrieval_score": require_finite_number(candidate.score, "retrieval_score"),
        "retrieval_rank": candidate.rank,
        "source": candidate.source,
        "document_text": text,
        "document_hash": document_hash(text),
        "payload": {
            "chunk_id": chunk_id,
            "section_id": candidate.payload.get("section_id"),
            "chunk_index": candidate.payload.get("chunk_index"),
        },
    }


def _validate_query_record(record: dict[str, Any]) -> tuple[dict[str, Any], int]:
    query_id = str(record.get("query_id") or "")
    query = str(record.get("query") or "")
    if not query_id or not query.strip():
        raise ArtifactContractError("Candidate query requires query_id and query")
    expected_hash = query_hash(query)
    if str(record.get("query_hash") or "") != expected_hash:
        raise ArtifactContractError(f"query_hash mismatch for {query_id}")
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ArtifactContractError(f"candidates must be a list for {query_id}")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ArtifactContractError(f"candidate must be an object for {query_id}")
        chunk_id = str(candidate.get("chunk_id") or "")
        if not chunk_id:
            raise ArtifactContractError(f"candidate chunk_id is missing for {query_id}")
        if chunk_id in seen:
            raise ArtifactContractError(f"duplicate chunk_id {chunk_id} for {query_id}")
        seen.add(chunk_id)
        text = str(candidate.get("document_text") or "")
        if not text:
            raise ArtifactContractError(
                f"document_text is missing for {query_id}/{chunk_id}"
            )
        if str(candidate.get("document_hash") or "") != document_hash(text):
            raise ArtifactContractError(
                f"document_hash mismatch for {query_id}/{chunk_id}"
            )
        require_finite_number(candidate.get("retrieval_score"), "retrieval_score")
        if int(candidate.get("retrieval_rank") or 0) < 1:
            raise ArtifactContractError(
                f"retrieval_rank is invalid for {query_id}/{chunk_id}"
            )
    return record, len(candidates)


class CandidateArtifactReader:
    def __init__(self, data_path: Path, manifest_path: Path | None = None):
        self.data_path = Path(data_path)
        self.manifest_path = Path(
            manifest_path or self.data_path.with_name("manifest.json")
        )
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactContractError(
                f"Invalid candidate manifest: {self.manifest_path}"
            ) from exc
        self.manifest = ArtifactManifest.from_dict(payload)
        self._iterator = None

    def __iter__(self):
        self._iterator = self._iter_records()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = self._iter_records()
        return next(self._iterator)

    def _iter_records(self):
        count = 0
        pair_count = 0
        for raw in iter_jsonl_objects(self.data_path):
            record, pairs = _validate_query_record(raw)
            count += 1
            pair_count += pairs
            yield record
        if count != self.manifest.record_count:
            raise ArtifactContractError(
                f"Candidate query count mismatch: {count} != {self.manifest.record_count}"
            )
        expected_pairs = int(self.manifest.identity.get("pair_count", pair_count))
        if pair_count != expected_pairs:
            raise ArtifactContractError(
                f"Candidate pair count mismatch: {pair_count} != {expected_pairs}"
            )

    @classmethod
    def from_data_path(cls, data_path: Path) -> CandidateArtifactReader:
        return cls(data_path, Path(data_path).with_name("manifest.json"))


def build_candidate_artifact(
    *,
    rows: list[dict[str, Any]],
    retriever: Any,
    output_path: Path,
    identity: dict[str, Any],
    candidate_k: int,
) -> CandidateArtifact:
    if candidate_k < 1:
        raise ValueError("candidate_k must be >= 1")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_count = 0
    pair_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        batch_search = getattr(retriever, "search_batch", None)
        row_batches = (
            (
                rows[start : start + _RETRIEVAL_BATCH_SIZE]
                for start in range(0, len(rows), _RETRIEVAL_BATCH_SIZE)
            )
            if batch_search is not None
            else (rows,)
        )
        for batch_rows in row_batches:
            if batch_search is not None:
                search_results = batch_search(batch_rows, candidate_k)
                if len(search_results) != len(batch_rows):
                    raise ValueError(
                        "batch retrieval returned an unexpected result count"
                    )
            else:
                search_results = [None] * len(batch_rows)
            for row, batched_candidates in zip(batch_rows, search_results, strict=True):
                search = getattr(retriever, "search_query_row", None)
                if batched_candidates is not None:
                    candidates = batched_candidates
                elif search is not None:
                    candidates = search(row, limit=candidate_k)
                else:
                    candidates = (
                        retriever.search(row, limit=candidate_k)
                        if getattr(retriever, "accepts_query_row", False)
                        else retriever.search(
                            str(row.get("query") or ""), limit=candidate_k
                        )
                    )
                serialized = [
                    _serialize_candidate(candidate)
                    for candidate in candidates[:candidate_k]
                ]
                record = {
                    "query_id": str(row.get("query_id") or ""),
                    "query": str(row.get("query") or ""),
                    "query_hash": query_hash(str(row.get("query") or "")),
                    "candidates": serialized,
                }
                _validate_query_record(record)
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                query_count += 1
                pair_count += len(serialized)
    manifest = ArtifactManifest.create(
        artifact_type="retrieval_candidates",
        data_path=output_path,
        record_count=query_count,
        identity={**identity, "pair_count": pair_count, "candidate_k": candidate_k},
    )
    manifest_path = output_path.with_name("manifest.json")
    write_json(manifest_path, manifest.to_dict())
    return CandidateArtifact(
        output_path, manifest_path, manifest, query_count, pair_count
    )
