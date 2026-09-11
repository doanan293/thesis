"""The only module that knows concrete adapters. Builds TurnDeps and the ChatTurnRunner."""

import os
from dataclasses import dataclass

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from pharma_agent.application.chat.context import TurnDeps
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.retrieval.ports import Reranker
from pharma_agent.domain.retrieval.service import RetrievalConfig, RetrievalService
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.llm.openai_adapter import (
    OpenAiLlmAdapter,
    default_client_factory,
    langfuse_client_factory,
)
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import build_reranker
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    OpenAiEmbedder,
    QdrantHybridRetriever,
    QdrantHydrator,
)
from pharma_agent.infrastructure.settings import Settings
from pharma_agent.infrastructure.skills.filesystem_catalog import FileSystemSkillCatalog


@dataclass
class Application:
    settings: Settings
    deps: TurnDeps
    runner: ChatTurnRunner
    retriever: QdrantHybridRetriever
    embedder: OpenAiEmbedder
    reranker: Reranker
    qdrant: AsyncQdrantClient
    embed_client: AsyncOpenAI

    async def aclose(self) -> None:
        await self.qdrant.close()
        await self.embed_client.close()
        close_reranker = getattr(self.reranker, "aclose", None)
        if close_reranker is not None:
            await close_reranker()


def build_application(settings: Settings) -> Application:
    if settings.langfuse.enabled:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse.public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse.secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse.host)
        client_factory = langfuse_client_factory
    else:
        client_factory = default_client_factory
    llm = OpenAiLlmAdapter(settings.llm, client_factory=client_factory)

    retrieval_settings = settings.retrieval
    embed_client = AsyncOpenAI(
        api_key=retrieval_settings.embedding.api_key,
        base_url=retrieval_settings.embedding.base_url,
        timeout=60.0,
        max_retries=2,
    )
    embedder = OpenAiEmbedder(
        embed_client,
        model=retrieval_settings.embedding.model,
        dimension=retrieval_settings.embedding.dimension,
    )
    qdrant = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=int(settings.qdrant.timeout_seconds),
    )
    retriever = QdrantHybridRetriever(
        qdrant,
        embedder,
        retrieval_settings.collection_alias,
        mode=retrieval_settings.mode,
        prefetch_k=retrieval_settings.prefetch_k,
        rrf_k=retrieval_settings.rrf_k,
        max_concurrent=retrieval_settings.max_concurrent_searches,
    )
    hydrator = QdrantHydrator(
        qdrant,
        retrieval_settings.collection_alias,
        window=retrieval_settings.hydrate_window,
    )
    reranker = build_reranker(retrieval_settings.rerank)
    retrieval = RetrievalService(
        retriever,
        reranker,
        hydrator,
        RetrievalConfig(
            candidate_k=retrieval_settings.candidate_k,
            rerank_top_n=retrieval_settings.rerank.top_n,
        ),
    )

    deps = TurnDeps(
        llm=llm,
        guardrail=GuardrailService(llm),
        retrieval=retrieval,
        skills=FileSystemSkillCatalog(settings.skills_dir),
        clock=SystemClock(),
    )
    runner = ChatTurnRunner(build_chat_graph(), deps, settings.budget)
    return Application(
        settings=settings,
        deps=deps,
        runner=runner,
        retriever=retriever,
        embedder=embedder,
        reranker=reranker,
        qdrant=qdrant,
        embed_client=embed_client,
    )
