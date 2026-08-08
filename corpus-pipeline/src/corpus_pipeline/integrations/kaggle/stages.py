from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from corpus_pipeline.evaluation.rerank_score_cache import prompt_contract_hash
from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.integrations.kaggle.models import (
    JobIdentity,
    StageJob,
    StageName,
    StageRequest,
    canonical_sha256,
)
from corpus_pipeline.runtime.catalog import ModelKind, require_model


def _count_jsonl(path: Path) -> int:
    count = 0
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Input row must be an object at {path}:{line_number}"
                    )
                count += 1
    except OSError as exc:
        raise ValueError(f"Input file is missing: {path}") from exc
    if count < 1:
        raise ValueError(f"Input file is empty: {path}")
    return count


def _candidate_pair_count(path: Path) -> int:
    total = 0
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid candidate JSON at {path}:{line_number}"
                    ) from exc
                candidates = payload.get("candidates")
                if not isinstance(candidates, list):
                    raise ValueError(
                        f"candidates must be a list at {path}:{line_number}"
                    )
                total += len(candidates)
    except OSError as exc:
        raise ValueError(f"Candidate file is missing: {path}") from exc
    if total < 1:
        raise ValueError(f"Candidate artifact contains no pairs: {path}")
    return total


class StageAdapter(Protocol):
    name: StageName
    contract_version: int

    def build_job(self, request: StageRequest) -> StageJob:
        raise NotImplementedError


@dataclass(frozen=True)
class CorpusEmbedStage:
    name: StageName = StageName.CORPUS_EMBED
    contract_version: int = 1

    def build_job(self, request: StageRequest) -> StageJob:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(
                f"corpus-embed requires an embedding model: {request.model}"
            )
        input_sha = sha256_file(request.input_path)
        total = _count_jsonl(request.input_path)
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=input_sha,
            runtime_parameters={
                "vector_dimension": spec.vector_dimension,
                "batch_size": spec.kaggle_request_batch_size,
            },
        )
        output_dir = (
            request.output_dir / self.name.value / spec.slug / identity.sha256[:12]
        )
        return StageJob(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            identity=identity,
            input_path=request.input_path,
            output_dir=output_dir,
            local_cache_path=output_dir / "vector_embeddings.jsonl",
            data_filename="vector_embeddings.jsonl",
            expected_total=total,
            worker_module="corpus_pipeline.integrations.kaggle.workers.corpus_embed",
            worker_config={
                "model": request.model,
                "input_path": str(request.input_path),
                "gguf_root": str(request.gguf_root),
                "vector_dimension": spec.vector_dimension,
                "batch_size": spec.kaggle_request_batch_size,
                "job_sha256": identity.sha256,
            },
        )


@dataclass(frozen=True)
class QueryEmbedStage:
    name: StageName = StageName.QUERY_EMBED
    contract_version: int = 1

    def build_job(self, request: StageRequest) -> StageJob:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(
                f"query-embed requires an embedding model: {request.model}"
            )
        input_sha = sha256_file(request.input_path)
        total = _count_jsonl(request.input_path)
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=input_sha,
            runtime_parameters={"batch_size": spec.kaggle_request_batch_size},
        )
        output_dir = (
            request.output_dir / self.name.value / spec.slug / identity.sha256[:12]
        )
        return StageJob(
            self.name,
            self.contract_version,
            request.model,
            identity,
            request.input_path,
            output_dir,
            output_dir / "query_embeddings.jsonl",
            "query_embeddings.jsonl",
            total,
            "corpus_pipeline.integrations.kaggle.workers.query_embed",
            {
                "model": request.model,
                "input_path": str(request.input_path),
                "gguf_root": str(request.gguf_root),
                "vector_dimension": spec.vector_dimension,
                "batch_size": spec.kaggle_request_batch_size,
                "job_sha256": identity.sha256,
            },
        )


@dataclass(frozen=True)
class RerankStage:
    name: StageName = StageName.RERANK
    contract_version: int = 1

    def build_job(self, request: StageRequest) -> StageJob:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError(f"rerank requires a reranker model: {request.model}")
        if not spec.reranker_protocol:
            raise ValueError(
                f"reranker protocol is missing from catalog: {request.model}"
            )
        manifest_path = request.input_path.with_name("manifest.json")
        if not manifest_path.is_file():
            raise ValueError(
                f"rerank requires adjacent candidate manifest: {manifest_path}"
            )
        input_sha = canonical_sha256(
            {
                "candidate_sha256": sha256_file(request.input_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        pair_count = _candidate_pair_count(request.input_path)
        protocol_hash = prompt_contract_hash(protocol=spec.reranker_protocol)
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=input_sha,
            runtime_parameters={
                "protocol": spec.reranker_protocol,
                "prompt_contract_sha256": protocol_hash,
            },
        )
        output_dir = (
            request.output_dir / self.name.value / spec.slug / identity.sha256[:12]
        )
        return StageJob(
            self.name,
            self.contract_version,
            request.model,
            identity,
            request.input_path,
            output_dir,
            output_dir / "rerank_scores.jsonl",
            "rerank_scores.jsonl",
            pair_count,
            "corpus_pipeline.integrations.kaggle.workers.rerank",
            {
                "model": request.model,
                "protocol": spec.reranker_protocol,
                "parallelism": spec.kaggle_parallel,
                "candidate_path": str(request.input_path),
                "candidate_manifest_path": str(manifest_path),
                "gguf_root": str(request.gguf_root),
                "job_sha256": identity.sha256,
            },
        )


_ADAPTERS: dict[StageName, StageAdapter] = {
    StageName.CORPUS_EMBED: CorpusEmbedStage(),
    StageName.QUERY_EMBED: QueryEmbedStage(),
    StageName.RERANK: RerankStage(),
}


def get_stage_adapter(name: StageName | str) -> StageAdapter:
    try:
        return _ADAPTERS[StageName(name)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Kaggle pipeline stage: {name}") from exc
