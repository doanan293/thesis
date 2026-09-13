from __future__ import annotations

import os
from pathlib import Path

from seed_pipeline.evaluation.artifact_contracts import canonical_sha256
from seed_pipeline.runtime.catalog import require_model

WORKSPACE_ROOT_ENV = "SEED_PIPELINE_ROOT"


class ConfigurationError(RuntimeError):
    """Raised when the corpus workspace cannot be resolved."""


def resolve_workspace_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    elif configured := os.environ.get(WORKSPACE_ROOT_ENV):
        candidate = Path(configured).expanduser().resolve()
    else:
        search_roots = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        search_roots.extend(Path(__file__).resolve().parent.parents)
        candidate = next(
            (root for root in search_roots if (root / "pyproject.toml").is_file()),
            None,
        )
        if candidate is None:
            raise ConfigurationError(
                "Could not find a corpus workspace containing pyproject.toml"
            )
    if not (candidate / "pyproject.toml").is_file():
        raise ConfigurationError(f"Corpus workspace is invalid: {candidate}")
    return candidate


PROJECT_ROOT = resolve_workspace_root()
DATA_DIR = PROJECT_ROOT / "data"

HEAVY_DATA_DIR = DATA_DIR / "heavy"
RESOURCES_DIR = DATA_DIR / "resources"
RESOURCES_ANKHANG_DIR = RESOURCES_DIR / "ankhang"
RESOURCES_CURATION_DIR = RESOURCES_DIR / "curation"
MANIFESTS_DIR = DATA_DIR / "manifests"

HEAVY_RAW_DIR = HEAVY_DATA_DIR / "raw"
RAW_DIR = HEAVY_RAW_DIR
RAW_ANKHANG_DIR = RAW_DIR / "ankhang"
RAW_ANKHANG_HTML_DIR = RAW_ANKHANG_DIR / "html"
RAW_CURATION_DIR = RESOURCES_CURATION_DIR
RAW_ANKHANG_SNAPSHOTS_DIR = RAW_ANKHANG_DIR / "snapshots"

INTERIM_DIR = HEAVY_DATA_DIR / ".work" / "manual"
PROCESSED_DIR = HEAVY_DATA_DIR / "processed"

TEXT_INTERIM_DIR = INTERIM_DIR / "text"
RAG_INTERIM_DIR = INTERIM_DIR / "rag"
DOCLING_INTERIM_DIR = INTERIM_DIR / "docling"
CANONICAL_INTERIM_DIR = INTERIM_DIR / "canonical"
ANKHANG_MARKDOWN_INTERIM_DIR = INTERIM_DIR / "ankhang_markdown"

RAG_FINAL_DIR = PROCESSED_DIR / "rag-final"
PROCESSED_EVALUATION_DIR = PROCESSED_DIR / "evaluation"

RETRIEVAL_EVAL_DIR = DATA_DIR / "retrieval_eval"
HEAVY_RETRIEVAL_EVAL_DIR = HEAVY_DATA_DIR / "retrieval_eval"
DATA_RUNS_DIR = DATA_DIR
RETRIEVAL_EVAL_RUNS_DIR = RETRIEVAL_EVAL_DIR

DATA_CACHE_DIR = HEAVY_DATA_DIR / "cache"
QUERY_EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / "query_embeddings"
RERANK_SCORE_CACHE_DIR = DATA_CACHE_DIR / "rerank_scores"
VECTOR_EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / "vector_embeddings"

WORK_DIR = HEAVY_DATA_DIR / ".work"
RUNTIME_PROFILE_DIR = HEAVY_DATA_DIR / "runtime_kaggle_profiles"
RAG_FINAL_SECTIONS_PATH = RAG_FINAL_DIR / "sections.jsonl"
RAG_FINAL_CHUNKS_PATH = RAG_FINAL_DIR / "chunks.jsonl"
RAG_FINAL_MANIFEST_PATH = RAG_FINAL_DIR / "manifest.json"
RAG_FINAL_VALIDATION_PATH = RAG_FINAL_DIR / "validation_report.json"


def retrieval_run_roots(run: str) -> tuple[Path, Path]:
    return RETRIEVAL_EVAL_DIR / run, HEAVY_RETRIEVAL_EVAL_DIR / run


def chunk_embedding_bundle_dir(model: str, corpus_sha256: str) -> Path:
    return VECTOR_EMBEDDING_CACHE_DIR / require_model(model).slug / corpus_sha256


def query_embedding_bundle_dir(model: str, evaluation_sha256: str) -> Path:
    return QUERY_EMBEDDING_CACHE_DIR / require_model(model).slug / evaluation_sha256


def query_embedding_cache_path(model: str) -> Path:
    return QUERY_EMBEDDING_CACHE_DIR / f"{require_model(model).slug}.jsonl"


def rerank_score_cache_path(
    model: str,
    model_sha256: str,
    request_contract_sha256: str,
) -> Path:
    identity = canonical_sha256(
        {
            "model": model,
            "model_sha256": model_sha256,
            "request_contract_sha256": request_contract_sha256,
        }
    )
    return RERANK_SCORE_CACHE_DIR / require_model(model).slug / f"{identity}.jsonl"
