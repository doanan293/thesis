from __future__ import annotations

import os
from pathlib import Path

from pharma_lab.runtime.catalog import require_model

WORKSPACE_ROOT_ENV = "PHARMA_LAB_ROOT"


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

SOURCES_DIR = DATA_DIR / "sources"
LEAFLETS_DIR = SOURCES_DIR / "leaflets"
LEAFLETS_HTML_DIR = LEAFLETS_DIR / "html"
LEAFLETS_URLS_DIR = LEAFLETS_DIR / "urls"
LEAFLETS_MANIFEST_PATH = LEAFLETS_DIR / "manifest.json"
FORMULARY_PDF_PATH = SOURCES_DIR / "duoc-thu-quoc-gia-viet-nam.pdf"
SOURCES_CURATION_DIR = SOURCES_DIR / "curation"

CORPUS_DIR = DATA_DIR / "corpus"
RAG_FINAL_DIR = CORPUS_DIR / "rag-final"
RAG_FINAL_SECTIONS_PATH = RAG_FINAL_DIR / "sections.jsonl"
BUNDLE_DIR = CORPUS_DIR / "formulary"

EVALUATION_DIR = DATA_DIR / "evaluation"
GOLD_DIR = EVALUATION_DIR / "gold"
RUNS_DIR = EVALUATION_DIR / "runs"

CACHE_DIR = DATA_DIR / "cache"
TEXT_EMBEDDING_CACHE_DIR = CACHE_DIR / "text_embeddings"
QUERY_EMBEDDING_CACHE_DIR = CACHE_DIR / "query_embeddings"
RERANK_SCORE_CACHE_DIR = CACHE_DIR / "rerank_scores"
KAGGLE_PROFILE_DIR = CACHE_DIR / "kaggle_profiles"
LOCAL_PROFILE_DIR = CACHE_DIR / "local_profiles"

WORK_DIR = DATA_DIR / "work"
BUILD_WORK_DIR = WORK_DIR / "build"
LOCK_DIR = WORK_DIR / "locks"
LOGS_DIR = WORK_DIR / "logs"
TEXT_INTERIM_DIR = BUILD_WORK_DIR / "text"
RAG_INTERIM_DIR = BUILD_WORK_DIR / "rag"
DOCLING_INTERIM_DIR = BUILD_WORK_DIR / "docling"
CANONICAL_INTERIM_DIR = BUILD_WORK_DIR / "canonical"
LEAFLET_MARKDOWN_INTERIM_DIR = BUILD_WORK_DIR / "leaflet_markdown"
BUNDLE_EMBED_WORK_DIR = WORK_DIR / "bundle-embed"
EVALUATION_CHUNKS_PATH = WORK_DIR / "evaluation-chunks" / "chunks.jsonl"

COMPOSE_FILE = PROJECT_ROOT.parent / "compose.yaml"
GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"
BACKEND_ENV_FILE = PROJECT_ROOT.parent / "backend" / ".env"


def run_dir(run: str) -> Path:
    return RUNS_DIR / run


def query_embedding_cache_path(model: str) -> Path:
    return QUERY_EMBEDDING_CACHE_DIR / f"{require_model(model).slug}.jsonl"


def text_embedding_cache_path(model: str) -> Path:
    return TEXT_EMBEDDING_CACHE_DIR / f"{require_model(model).slug}.jsonl"


def rerank_score_cache_path(model: str) -> Path:
    return RERANK_SCORE_CACHE_DIR / f"{require_model(model).slug}.jsonl"


def rerank_log_path(model: str) -> Path:
    return LOGS_DIR / "rerank" / f"{require_model(model).slug}.log"
