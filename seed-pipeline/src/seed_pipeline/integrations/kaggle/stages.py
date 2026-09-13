from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

from seed_pipeline.integrations.kaggle.models import (
    InputBundle,
    InputFile,
    JobIdentity,
    StageJob,
    StageName,
    StageRequest,
)
from seed_pipeline.runtime.benchmarking import BenchmarkLevel
from seed_pipeline.runtime.catalog import ModelKind, require_model
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate


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
    @property
    def name(self) -> StageName: ...

    @property
    def contract_version(self) -> int: ...

    def build_job(self, request: StageRequest) -> StageJob: ...


def _required_runtime_profile(request: StageRequest) -> RuntimeCandidate:
    profile = request.runtime_profile
    if profile is None:
        raise ValueError("runtime profile is required for production stages")
    if isinstance(profile, RuntimeCandidate):
        return profile
    if isinstance(profile, dict):
        return RuntimeCandidate.from_dict(profile)
    raise ValueError("runtime profile must be a RuntimeCandidate")


@dataclass(frozen=True)
class CorpusEmbedStage:
    name: StageName = StageName.CORPUS_EMBED
    contract_version: int = 3

    def build_job(self, request: StageRequest) -> StageJob:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(
                f"corpus-embed requires an embedding model: {request.model}"
            )
        input_bundle = InputBundle.create(
            (InputFile.create("input", request.input_path),)
        )
        total = _count_jsonl(request.input_path)
        profile = _required_runtime_profile(request)
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=input_bundle.sha256,
            runtime_parameters={
                "vector_dimension": spec.vector_dimension,
                "batch_size": profile.request_batch_size,
                "runtime_profile": profile.to_dict(),
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
            input_bundle=input_bundle,
            output_dir=output_dir,
            local_cache_path=output_dir / "text_embeddings.jsonl",
            data_filename="text_embeddings.jsonl",
            expected_total=total,
            worker_module="seed_pipeline.integrations.kaggle.workers.corpus_embed",
            worker_config={
                "model": request.model,
                "gguf_root": str(request.gguf_root),
                "vector_dimension": spec.vector_dimension,
                "batch_size": profile.request_batch_size,
                "runtime_overrides": profile.to_dict(),
                "job_sha256": identity.sha256,
            },
        )


@dataclass(frozen=True)
class QueryEmbedStage:
    name: StageName = StageName.QUERY_EMBED
    contract_version: int = 2

    def build_job(self, request: StageRequest) -> StageJob:
        spec = require_model(request.model)
        if spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(
                f"query-embed requires an embedding model: {request.model}"
            )
        input_bundle = InputBundle.create(
            (InputFile.create("input", request.input_path),)
        )
        total = _count_jsonl(request.input_path)
        profile = _required_runtime_profile(request)
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=input_bundle.sha256,
            runtime_parameters={
                "batch_size": profile.request_batch_size,
                "runtime_profile": profile.to_dict(),
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
            input_bundle,
            output_dir,
            output_dir / "query_embeddings.jsonl",
            "query_embeddings.jsonl",
            total,
            "seed_pipeline.integrations.kaggle.workers.query_embed",
            {
                "model": request.model,
                "gguf_root": str(request.gguf_root),
                "vector_dimension": spec.vector_dimension,
                "batch_size": profile.request_batch_size,
                "runtime_overrides": profile.to_dict(),
                "job_sha256": identity.sha256,
            },
        )


@dataclass(frozen=True)
class RerankStage:
    name: StageName = StageName.RERANK
    contract_version: int = 3

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
        input_bundle = InputBundle.create(
            (
                InputFile.create("candidates", request.input_path),
                InputFile.create("candidate_manifest", manifest_path),
            )
        )
        pair_count = _candidate_pair_count(request.input_path)
        profile = _required_runtime_profile(request)
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        protocol_hash = spec.rerank_contract.sha256
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=input_bundle.sha256,
            runtime_parameters={
                "protocol": spec.reranker_protocol,
                "request_contract_sha256": protocol_hash,
                "runtime_profile": profile.to_dict(),
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
            input_bundle,
            output_dir,
            output_dir / "rerank_scores.jsonl",
            "rerank_scores.jsonl",
            pair_count,
            "seed_pipeline.integrations.kaggle.workers.rerank",
            {
                "model": request.model,
                "protocol": spec.reranker_protocol,
                "parallelism": profile.concurrency,
                "runtime_overrides": profile.to_dict(),
                "gguf_root": str(request.gguf_root),
                "job_sha256": identity.sha256,
            },
        )


@dataclass(frozen=True)
class BenchmarkStage:
    name: StageName
    base: StageAdapter
    contract_version: int = 1

    def build_job(self, request: StageRequest) -> StageJob:
        spec = require_model(request.model)
        if spec.kind is ModelKind.RERANKER:
            search_space = spec.rerank_search_space
        else:
            if spec.embedding_search_space is None:
                raise ValueError("embedding model has no runtime search space")
            search_space = (
                spec.embedding_search_space.query
                if self.base.name is StageName.QUERY_EMBED
                else spec.embedding_search_space.corpus
            )
        if search_space is None:
            raise ValueError("model has no runtime search space")
        try:
            base_request = replace(
                request,
                stage=self.base.name,
                runtime_profile=search_space.candidates[0],
            )
        except TypeError:
            base_request = cast(Any, copy.copy(request))
            base_request.stage = self.base.name
            base_request.runtime_profile = search_space.candidates[0]
        base_job = self.base.build_job(base_request)
        levels = tuple(
            BenchmarkLevel(candidate.request_batch_size, candidate.concurrency)
            for candidate in search_space.candidates
        )
        identity = JobIdentity.create(
            stage=self.name,
            contract_version=self.contract_version,
            model=request.model,
            model_sha256=spec.sha256,
            input_sha256=base_job.input_bundle.sha256,
            runtime_parameters={
                "benchmark_items": getattr(request, "benchmark_items", None)
                or base_job.expected_total,
                "benchmark_levels": [
                    {"batch_size": item.batch_size, "concurrency": item.concurrency}
                    for item in levels
                ],
            },
        )
        output_dir = (
            request.output_dir / self.name.value / spec.slug / identity.sha256[:12]
        )
        config = dict(base_job.worker_config)
        config.update(
            {
                "stage": self.name.value,
                "identity": identity.payload,
                "job_sha256": identity.sha256,
                "benchmark_levels": [
                    {"batch_size": item.batch_size, "concurrency": item.concurrency}
                    for item in levels
                ],
                "benchmark_candidates": [
                    candidate.to_dict() for candidate in search_space.candidates
                ],
                "benchmark_items": getattr(request, "benchmark_items", None)
                or base_job.expected_total,
            }
        )
        return replace(
            base_job,
            stage=self.name,
            contract_version=self.contract_version,
            identity=identity,
            output_dir=output_dir,
            local_cache_path=output_dir / "benchmark_results.jsonl",
            data_filename="benchmark_results.jsonl",
            worker_module="seed_pipeline.integrations.kaggle.workers.benchmark",
            worker_config=config,
        )


_ADAPTERS: dict[StageName, StageAdapter] = {
    StageName.CORPUS_EMBED: CorpusEmbedStage(),
    StageName.QUERY_EMBED: QueryEmbedStage(),
    StageName.RERANK: RerankStage(),
}
_ADAPTERS.update(
    {
        StageName.RERANK_BENCHMARK: BenchmarkStage(
            StageName.RERANK_BENCHMARK, RerankStage()
        ),
        StageName.QUERY_EMBED_BENCHMARK: BenchmarkStage(
            StageName.QUERY_EMBED_BENCHMARK, QueryEmbedStage()
        ),
        StageName.CORPUS_EMBED_BENCHMARK: BenchmarkStage(
            StageName.CORPUS_EMBED_BENCHMARK, CorpusEmbedStage()
        ),
    }
)


def get_stage_adapter(name: StageName | str) -> StageAdapter:
    try:
        return _ADAPTERS[StageName(name)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported Kaggle pipeline stage: {name}") from exc
