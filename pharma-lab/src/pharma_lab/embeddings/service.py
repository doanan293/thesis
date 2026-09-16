"""Run text embedding for bundle inputs locally (llama.cpp) or on Kaggle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pharma_lab.cache.jsonl_records import merge_records
from pharma_lab.config.paths import COMPOSE_FILE, GGUF_ROOT, WORK_DIR
from pharma_lab.embeddings.text_cache import (
    TextEmbeddingCache,
    cache_record_key,
    embed_missing,
    read_embedding_inputs,
)
from pharma_lab.integrations.kaggle.job_lock import (
    kaggle_cache_lock,
    kaggle_job_lock,
)
from pharma_lab.runtime.catalog import ModelKind, require_model
from pharma_lab.runtime.client import LlamaCppClient
from pharma_lab.runtime.compose import LlamaCppComposeManager, resolve_server


@dataclass(frozen=True)
class TextEmbeddingRequest:
    inputs_path: Path
    model: str
    cache_path: Path
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float
    kaggle_account: str | None = None


@dataclass(frozen=True)
class TextEmbeddingResult:
    cache_path: Path | None
    actions: tuple[str, ...]
    incomplete: bool = False


class TextEmbeddingBackend(Protocol):
    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult: ...


def open_text_cache(path: Path, model: str) -> TextEmbeddingCache:
    spec = require_model(model)
    if spec.kind is not ModelKind.EMBEDDING or spec.vector_dimension is None:
        raise ValueError(f"--model must select an embedding model: {model}")
    return TextEmbeddingCache(
        path, model=model, model_sha256=spec.sha256, vector_dim=spec.vector_dimension
    )


class LocalTextEmbeddingBackend:
    def __init__(
        self,
        *,
        compose_file: Path = COMPOSE_FILE,
        gguf_root: Path = GGUF_ROOT,
        server_mode: str = "compose",
        server_urls: Sequence[str] = (),
    ) -> None:
        self.compose_file = compose_file
        self.gguf_root = gguf_root
        self.server_mode = server_mode
        self.server_urls = list(server_urls)

    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        spec = require_model(request.model)
        if request.dry_run:
            open_text_cache(request.cache_path, request.model)
            return TextEmbeddingResult(None, ("dry-run",))
        if request.force:
            request.cache_path.unlink(missing_ok=True)
        cache = open_text_cache(request.cache_path, request.model)
        inputs = read_embedding_inputs(request.inputs_path)
        if not cache.missing(inputs):
            return TextEmbeddingResult(request.cache_path, ("cached",))
        endpoints = resolve_server(
            self.server_mode,
            self.server_urls,
            spec,
            LlamaCppComposeManager(self.compose_file),
            self.gguf_root,
        )
        run = embed_missing(
            inputs,
            cache,
            [
                LlamaCppClient(endpoint, timeout=request.request_timeout_seconds)
                for endpoint in endpoints
            ],
            batch_size=spec.local_request_batch_size,
        )
        return TextEmbeddingResult(
            request.cache_path,
            (f"cached={run.cached}", f"embedded={run.embedded}"),
            incomplete=run.stopped_early,
        )


class KaggleTextEmbeddingBackend:
    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        with kaggle_job_lock(request.cache_path):
            return self._run_unlocked(request)

    @staticmethod
    def _run_unlocked(request: TextEmbeddingRequest) -> TextEmbeddingResult:
        from pharma_lab.integrations.kaggle.auto_profile import (
            ensure_runtime_profile,
        )
        from pharma_lab.integrations.kaggle.models import StageName
        from pharma_lab.integrations.kaggle.service import run_kaggle_stage

        spec = require_model(request.model)
        resolution = ensure_runtime_profile(
            workload="corpus-embed",
            benchmark_stage=StageName.CORPUS_EMBED_BENCHMARK.value,
            model=request.model,
            input_path=request.inputs_path,
            gguf_root=GGUF_ROOT,
            budget_seconds=request.budget_seconds,
            dry_run=request.dry_run,
            force=request.force,
            kaggle_account=request.kaggle_account,
        )
        if resolution.profile is None:
            return TextEmbeddingResult(
                None, (f"profile={resolution.action}",), incomplete=True
            )
        result = run_kaggle_stage(
            stage=StageName.CORPUS_EMBED,
            model=request.model,
            input_path=request.inputs_path,
            output_dir=WORK_DIR / "kaggle-text-embeddings" / spec.slug,
            gguf_root=GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
            runtime_profile=resolution.profile.selected,
            kaggle_account=request.kaggle_account,
        )
        actions = tuple(action.reason for action in result.actions)
        if result.artifact_path is None:
            return TextEmbeddingResult(
                None, actions, incomplete=not result.completion.is_complete
            )
        remote = open_text_cache(result.artifact_path, request.model)
        with kaggle_cache_lock(request.cache_path):
            merge_records(
                request.cache_path,
                remote.records_for(remote.digests()),
                key=cache_record_key,
                equivalent=lambda left, right: left["embedding"] == right["embedding"],
            )
        return TextEmbeddingResult(
            request.cache_path, actions, incomplete=not result.completion.is_complete
        )
