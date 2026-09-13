from uuid import uuid4

import pytest
from langfuse import get_client
from openai.resources.embeddings import AsyncEmbeddings

from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.infrastructure.composition import (
    build_application,
    build_retrieval_service,
)
from pharma_agent.infrastructure.llm.openai_adapter import OpenAiLlmAdapter
from pharma_agent.infrastructure.observability.langfuse_retrieval import (
    LangfuseTracedReranker,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import (
    LlamaCppCompletionReranker,
    NoopReranker,
)
from pharma_agent.infrastructure.retrieval.postgres_corpus import PostgresCorpusReader
from pharma_agent.infrastructure.retrieval.qdrant_adapter import QdrantHybridRetriever
from pharma_agent.infrastructure.settings import Settings
from tests.fakes import FakeEmbedder


async def test_build_application_wires_real_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    app = build_application(Settings(_env_file=None))
    assert isinstance(app.runner, ChatTurnRunner)
    assert isinstance(app.deps.llm, OpenAiLlmAdapter)
    assert isinstance(app.retrieval.retriever, QdrantHybridRetriever)
    assert isinstance(app.retrieval.reranker, LlamaCppCompletionReranker)
    await app.aclose()


async def test_build_application_respects_rerank_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "none")
    monkeypatch.setenv("PHARMA_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    app = build_application(Settings(_env_file=None))
    assert isinstance(app.retrieval.reranker, NoopReranker)
    await app.aclose()


def test_build_application_requires_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        build_application(Settings(_env_file=None))


async def test_langfuse_traces_embeddings_and_rerank(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    public_key = f"pk-test-{uuid4().hex}"
    environment = {
        "PHARMA_LLM__DEFAULT__API_KEY": "sk-test",
        "PHARMA_SKILLS_DIR": str(tmp_path),
        "PHARMA_QDRANT__CHECK_COMPATIBILITY": "false",
        "PHARMA_LANGFUSE__PUBLIC_KEY": public_key,
        "PHARMA_LANGFUSE__SECRET_KEY": "sk-test",
        "PHARMA_LANGFUSE__HOST": "http://langfuse.test",
        # Set here so build_application's setdefault leaves them for monkeypatch to undo.
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": "sk-test",
        "LANGFUSE_HOST": "http://langfuse.test",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    app = build_application(Settings(_env_file=None))

    assert isinstance(app.retrieval.reranker, LangfuseTracedReranker)
    # The Langfuse OpenAI integration is installed when the app is built, before any call.
    assert hasattr(AsyncEmbeddings.create, "__wrapped__")
    await app.aclose()
    get_client(public_key=public_key).shutdown()


async def test_build_retrieval_service_owns_its_database_unless_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    settings = Settings(_env_file=None)

    stack = build_retrieval_service(settings)
    assert isinstance(stack.retriever, QdrantHybridRetriever)
    assert isinstance(stack.reader, PostgresCorpusReader)
    assert stack.owns_database is True
    await stack.aclose()

    shared = Database(settings.postgres.dsn, pool_size=1)
    borrowed = build_retrieval_service(settings, database=shared)
    assert borrowed.database is shared and borrowed.owns_database is False
    await borrowed.aclose()
    await shared.dispose()


async def test_build_retrieval_service_uses_an_injected_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    embedder = FakeEmbedder()

    stack = build_retrieval_service(Settings(_env_file=None), embedder=embedder)

    assert stack.embedder is embedder and stack.embed_client is None
    await stack.aclose()
