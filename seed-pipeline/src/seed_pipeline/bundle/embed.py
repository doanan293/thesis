"""Pre-compute bundle embeddings keyed by sha256(embedding_text) and store them in the bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BundleManifest,
    KnowledgeBundle,
    model_slug,
    read_bundle,
)

from seed_pipeline.bundle.chunks import iter_section_chunks
from seed_pipeline.bundle.io import write_validated_bundle
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from seed_pipeline.config.paths import BUNDLE_EMBED_WORK_DIR, text_embedding_cache_path
from seed_pipeline.embeddings.service import (
    TextEmbeddingBackend,
    TextEmbeddingRequest,
    open_text_cache,
)
from seed_pipeline.embeddings.text_cache import (
    EmbeddingInput,
    TextEmbeddingError,
    write_embedding_inputs,
)
from seed_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class BundleEmbedRequest:
    bundle_dir: Path
    model: str
    force: bool = False
    dry_run: bool = False
    budget_seconds: int = DEFAULT_BUDGET_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    kaggle_account: str | None = None
    cache_path: Path | None = None
    work_dir: Path = BUNDLE_EMBED_WORK_DIR


@dataclass(frozen=True)
class BundleEmbedResult:
    manifest: BundleManifest | None
    inputs: int
    vectors: int
    embeddings_file: str | None
    actions: tuple[str, ...]
    incomplete: bool


def collect_embedding_inputs(bundle: KnowledgeBundle) -> list[EmbeddingInput]:
    unique: dict[str, EmbeddingInput] = {}
    for item in iter_section_chunks(bundle):
        for draft in item.drafts:
            unique.setdefault(
                draft.embedding_text_sha256,
                EmbeddingInput(draft.embedding_text_sha256, draft.embedding_text),
            )
    return [unique[digest] for digest in sorted(unique)]


def embed_bundle(
    request: BundleEmbedRequest, backend: TextEmbeddingBackend
) -> BundleEmbedResult:
    spec = require_model(request.model)
    bundle = read_bundle(request.bundle_dir)
    inputs = collect_embedding_inputs(bundle)
    inputs_path = request.work_dir / spec.slug / "embedding_inputs.jsonl"
    write_embedding_inputs(inputs, inputs_path)
    cache_path = request.cache_path or text_embedding_cache_path(request.model)
    stage = backend.run(
        TextEmbeddingRequest(
            inputs_path=inputs_path,
            model=request.model,
            cache_path=cache_path,
            force=request.force,
            dry_run=request.dry_run,
            budget_seconds=request.budget_seconds,
            request_timeout_seconds=request.request_timeout_seconds,
            kaggle_account=request.kaggle_account,
        )
    )
    if request.dry_run or stage.incomplete or stage.cache_path is None:
        return BundleEmbedResult(
            manifest=None,
            inputs=len(inputs),
            vectors=0,
            embeddings_file=None,
            actions=stage.actions,
            incomplete=stage.incomplete,
        )
    cache = open_text_cache(stage.cache_path, request.model)
    vectors: dict[str, list[float]] = {}
    for item in inputs:
        vector = cache.get(item.embedding_text_sha256)
        if vector is None:
            raise TextEmbeddingError(
                f"Embedding cache {stage.cache_path} is missing {item.embedding_text_sha256}"
            )
        vectors[item.embedding_text_sha256] = vector
    updated = bundle.model_copy(
        update={"embeddings": {**bundle.embeddings, request.model: vectors}}
    )
    manifest = write_validated_bundle(updated, request.bundle_dir)
    return BundleEmbedResult(
        manifest=manifest,
        inputs=len(inputs),
        vectors=len(vectors),
        embeddings_file=f"embeddings/{model_slug(request.model)}.jsonl",
        actions=stage.actions,
        incomplete=False,
    )
