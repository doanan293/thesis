import importlib
import inspect

import pytest
from pharma_agent.application.chat.context import PipelineOptions
from pharma_agent.application.chat.runner import ChatTurnRunner, TurnOutcome
from pharma_agent.domain.corpus.bundle import BUNDLE_SCHEMA_VERSION, model_slug
from pharma_agent.domain.corpus.chunking import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    chunk_section,
)
from pharma_agent.infrastructure.composition import (
    RetrievalStack,
    build_retrieval_service,
)

PUBLIC_MODULES = (
    "pharma_agent.domain.corpus.bundle",
    "pharma_agent.domain.corpus.identity",
    "pharma_agent.domain.corpus.chunking",
    "pharma_agent.domain.corpus.enrichment",
    "pharma_agent.domain.corpus.hydrate",
    "pharma_agent.infrastructure.composition",
    # The E2E harness runs the chat graph in-process.
    "pharma_agent.application.chat.context",
    "pharma_agent.application.chat.graph",
    "pharma_agent.application.chat.runner",
    "pharma_agent.infrastructure.llm.openai_adapter",
    "pharma_agent.infrastructure.settings",
)


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_backend_public_module_imports(module: str) -> None:
    assert importlib.import_module(module).__name__ == module


def test_pinned_backend_names_have_expected_values() -> None:
    assert BUNDLE_SCHEMA_VERSION == "knowledge-bundle/v1"
    assert CHUNKER_VERSION == "chunker-v1"
    assert MAX_CHUNK_CHARS == 3000
    assert model_slug("qwen3-embedding:4b-fp16") == "qwen3_embedding_4b_fp16"
    assert callable(chunk_section)
    assert "embedder" in inspect.signature(build_retrieval_service).parameters
    assert RetrievalStack.__name__ == "RetrievalStack"


def test_chat_runner_accepts_pipeline_options() -> None:
    assert "pipeline" in inspect.signature(ChatTurnRunner).parameters
    assert PipelineOptions() == PipelineOptions(rephrase=True, judge_refine=True)
    assert "context_text" in TurnOutcome.model_fields
