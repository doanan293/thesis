from __future__ import annotations

import os
from pathlib import Path

from corpus_pipeline.runtime.catalog import require_model

WORKSPACE_ROOT_ENV = "CORPUS_PIPELINE_ROOT"


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

RAW_DIR = DATA_DIR / "raw"
RAW_ANKHANG_DIR = RAW_DIR / "ankhang"
RAW_ANKHANG_HTML_DIR = RAW_ANKHANG_DIR / "html"
RAW_CURATION_DIR = RAW_DIR / "curation"
RAW_ANKHANG_SNAPSHOTS_DIR = RAW_ANKHANG_DIR / "snapshots"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TEXT_INTERIM_DIR = INTERIM_DIR / "text"
RAG_INTERIM_DIR = INTERIM_DIR / "rag"
DOCLING_INTERIM_DIR = INTERIM_DIR / "docling"
CANONICAL_INTERIM_DIR = INTERIM_DIR / "canonical"
ANKHANG_MARKDOWN_INTERIM_DIR = INTERIM_DIR / "ankhang_markdown"

RAG_FINAL_DIR = PROCESSED_DIR / "rag-final"
PROCESSED_EVALUATION_DIR = PROCESSED_DIR / "evaluation"

DATA_RUNS_DIR = DATA_DIR / "runs"
RETRIEVAL_EVAL_RUNS_DIR = DATA_RUNS_DIR / "retrieval_eval"

DATA_CACHE_DIR = DATA_DIR / "cache"
QUERY_EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / "query_embeddings"
RERANK_SCORE_CACHE_DIR = DATA_CACHE_DIR / "rerank_scores"
VECTOR_EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / "vector_embeddings"

WORK_DIR = DATA_DIR / ".work"
RAG_FINAL_SECTIONS_PATH = RAG_FINAL_DIR / "sections.jsonl"
RAG_FINAL_CHUNKS_PATH = RAG_FINAL_DIR / "chunks.jsonl"
RAG_FINAL_MANIFEST_PATH = RAG_FINAL_DIR / "manifest.json"
RAG_FINAL_VALIDATION_PATH = RAG_FINAL_DIR / "validation_report.json"


def chunk_embedding_bundle_dir(model: str, corpus_sha256: str) -> Path:
    return VECTOR_EMBEDDING_CACHE_DIR / require_model(model).slug / corpus_sha256


def query_embedding_bundle_dir(model: str, evaluation_sha256: str) -> Path:
    return QUERY_EMBEDDING_CACHE_DIR / require_model(model).slug / evaluation_sha256
