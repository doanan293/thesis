import pytest

from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.infrastructure.composition import build_application
from pharma_agent.infrastructure.llm.openai_adapter import OpenAiLlmAdapter
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import (
    LlamaCppCompletionReranker,
    NoopReranker,
)
from pharma_agent.infrastructure.retrieval.qdrant_adapter import QdrantHybridRetriever
from pharma_agent.infrastructure.settings import Settings


async def test_build_application_wires_real_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    app = build_application(Settings(_env_file=None))
    assert isinstance(app.runner, ChatTurnRunner)
    assert isinstance(app.deps.llm, OpenAiLlmAdapter)
    assert isinstance(app.retriever, QdrantHybridRetriever)
    assert isinstance(app.reranker, LlamaCppCompletionReranker)
    await app.aclose()


async def test_build_application_respects_rerank_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "none")
    monkeypatch.setenv("PHARMA_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    app = build_application(Settings(_env_file=None))
    assert isinstance(app.reranker, NoopReranker)
    await app.aclose()


def test_build_application_requires_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        build_application(Settings(_env_file=None))
