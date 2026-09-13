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
)


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_old_chunk_contract_modules_are_gone(module: str) -> None:
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("args", [["vectors", "upload"], ["embed", "chunks"]])
def test_old_commands_are_gone(args: list[str]) -> None:
    assert CliRunner().invoke(app, args).exit_code == 2


def test_paths_no_longer_expose_the_unified_chunk_contract() -> None:
    assert not hasattr(paths, "RAG_FINAL_CHUNKS_PATH")
    assert not hasattr(paths, "VECTOR_EMBEDDING_CACHE_DIR")
