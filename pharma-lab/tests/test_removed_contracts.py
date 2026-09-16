import importlib.util

import pytest
from typer.testing import CliRunner

from pharma_lab.cli.app import app
from pharma_lab.config import paths

REMOVED_MODULES = (
    "pharma_lab.vector_store",
    "pharma_lab.integrations.postgres",
    "pharma_lab.corpus.metadata.build_rag_metadata",
    "pharma_lab.corpus.metadata.qdrant_payload_contract",
    "pharma_lab.corpus.metadata.term_enrichment",
    "pharma_lab.cli.commands.vectors",
    "pharma_lab.evaluation.retrievers",
    "pharma_lab.evaluation.retrieval_service",
    "pharma_lab.evaluation.dump_retrieval_candidates",
    "pharma_lab.evaluation.rejudge_service",
    "pharma_lab.evaluation.rejudging",
    "pharma_lab.evaluation.query_embedding_artifact",
    "pharma_lab.artifacts.snapshot",
    "pharma_lab.corpus.crawling.collect_urls",
    "pharma_lab.corpus.crawling.download_html",
    "pharma_lab.corpus.crawling.integrate_ankhang",
    "pharma_lab.bundle.parity",
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
        ("pharma_lab.evaluation.query_embedding_service", "query_checkpoint_path"),
        ("pharma_lab.evaluation.rerank_score_cache", "finalize_rerank_cache"),
        (
            "pharma_lab.integrations.kaggle.workers.query_embed",
            "_legacy_query_records",
        ),
    ],
)
def test_legacy_cache_helpers_are_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("pharma_lab.runtime.model_profiles", "CompletionScoring"),
        ("pharma_lab.runtime.model_profiles", "qwen3_rerank_contract"),
        ("pharma_lab.runtime.model_profiles", "bge_gemma_rerank_contract"),
        ("pharma_lab.runtime.model_profiles", "build_qwen3_yes_no_prompt"),
        ("pharma_lab.runtime.model_profiles", "build_bge_gemma_yes_no_prompt"),
        ("pharma_lab.evaluation.rerankers", "build_qwen_rerank_prompt"),
        ("pharma_lab.evaluation.rerank_contract", "prompt_contract_hash"),
        ("pharma_lab.runtime.client", "CompletionRerankResult"),
        ("pharma_lab.runtime.client", "CompletionPromptTiming"),
        ("pharma_lab.runtime.benchmarking", "compare_cache_arms"),
        ("pharma_lab.runtime.benchmarking", "CacheComparison"),
        ("pharma_lab.runtime.model_profiles", "RerankRuntimeProfile"),
        ("pharma_lab.runtime.benchmarking", "rerank_levels"),
    ],
)
def test_completion_logprobs_code_is_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "name",
    [
        "completion_payload",
        "rerank_completion",
        "rerank_completions_async",
        "rerank_completion_results_async",
    ],
)
def test_llama_cpp_client_has_no_completion_scoring(name: str) -> None:
    from pharma_lab.runtime.client import LlamaCppClient

    assert not hasattr(LlamaCppClient, name)
