import importlib.util

import pytest
from typer.testing import CliRunner

from seed_pipeline.cli.app import app
from seed_pipeline.config import paths

REMOVED_MODULES = (
    "seed_pipeline.vector_store",
    "seed_pipeline.integrations.postgres",
    "seed_pipeline.corpus.metadata.build_rag_metadata",
    "seed_pipeline.corpus.metadata.qdrant_payload_contract",
    "seed_pipeline.corpus.metadata.term_enrichment",
    "seed_pipeline.cli.commands.vectors",
    "seed_pipeline.evaluation.retrievers",
    "seed_pipeline.evaluation.retrieval_service",
    "seed_pipeline.evaluation.dump_retrieval_candidates",
    "seed_pipeline.evaluation.rejudge_service",
    "seed_pipeline.evaluation.rejudging",
    "seed_pipeline.evaluation.query_embedding_artifact",
    "seed_pipeline.artifacts.snapshot",
    "seed_pipeline.corpus.crawling.collect_urls",
    "seed_pipeline.corpus.crawling.download_html",
    "seed_pipeline.corpus.crawling.integrate_ankhang",
    "seed_pipeline.bundle.parity",
)

REMOVED_PATH_NAMES = (
    "PROCESSED_DIR",
    "PROCESSED_EVALUATION_DIR",
    "DATA_CACHE_DIR",
    "RUNTIME_PROFILE_DIR",
    "BUNDLES_DIR",
    "DEFAULT_BUNDLE_DIR",
    "INTERIM_DIR",
    "RETRIEVAL_EVAL_DIR",
    "HEAVY_RETRIEVAL_EVAL_DIR",
    "DATA_RUNS_DIR",
    "RETRIEVAL_EVAL_RUNS_DIR",
    "RAG_FINAL_MANIFEST_PATH",
    "RAG_FINAL_VALIDATION_PATH",
    "retrieval_run_roots",
    "query_embedding_bundle_dir",
    "HEAVY_DATA_DIR",
    "HEAVY_RAW_DIR",
    "RAW_DIR",
    "RAW_ANKHANG_DIR",
    "RAW_ANKHANG_SNAPSHOTS_DIR",
    "RAW_CURATION_DIR",
    "RESOURCES_DIR",
    "RESOURCES_CURATION_DIR",
    "RESOURCES_ANKHANG_DIR",
    "RAW_ANKHANG_HTML_DIR",
    "MANIFESTS_DIR",
    "MIGRATION_DIR",
)


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_old_chunk_contract_modules_are_gone(module: str) -> None:
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize(
    "args",
    [
        ["vectors", "upload"],
        ["embed", "chunks"],
        ["evaluation", "rejudge-current"],
        ["bundle", "parity"],
    ],
)
def test_old_commands_are_gone(args: list[str]) -> None:
    assert CliRunner().invoke(app, [*args, "--help"]).exit_code == 2


@pytest.mark.parametrize("name", REMOVED_PATH_NAMES)
def test_pre_layout_path_names_are_gone(name: str) -> None:
    assert not hasattr(paths, name)


def test_paths_no_longer_expose_the_unified_chunk_contract() -> None:
    assert not hasattr(paths, "RAG_FINAL_CHUNKS_PATH")
    assert not hasattr(paths, "VECTOR_EMBEDDING_CACHE_DIR")


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("seed_pipeline.evaluation.query_embedding_service", "query_checkpoint_path"),
        ("seed_pipeline.evaluation.rerank_score_cache", "finalize_rerank_cache"),
        (
            "seed_pipeline.integrations.kaggle.workers.query_embed",
            "_legacy_query_records",
        ),
    ],
)
def test_legacy_cache_helpers_are_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)
