"""The only module that knows concrete adapters: the retrieval stack, TurnDeps and the runner."""

import os
from dataclasses import dataclass

from langfuse import get_client
from langgraph.checkpoint.base import BaseCheckpointSaver
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from pharma_agent.application.chat.context import TurnDeps
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.tracing import TurnTracer
from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.llm.port import LlmPort
from pharma_agent.domain.retrieval.ports import CorpusReader, Reranker
from pharma_agent.domain.retrieval.service import RetrievalConfig, RetrievalService
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.domain.skill.ports import SkillCatalog
from pharma_agent.infrastructure.llm.openai_adapter import (
    OpenAiLlmAdapter,
    default_client_factory,
    langfuse_client_factory,
)
from pharma_agent.infrastructure.observability.langfuse_retrieval import (
    LangfuseTracedReranker,
)
from pharma_agent.infrastructure.openai_client import build_async_openai
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import build_reranker
from pharma_agent.infrastructure.retrieval.postgres_corpus import (
    PostgresCorpusReader,
    PostgresHydrator,
)
from pharma_agent.infrastructure.retrieval.qdrant_adapter import (
    OpenAiEmbedder,
    QdrantHybridRetriever,
)
from pharma_agent.infrastructure.settings import RetrievalSettings, Settings
from pharma_agent.infrastructure.skills.filesystem_catalog import FileSystemSkillCatalog


@dataclass
class RetrievalStack:
    """Retrieval over Postgres (schema `corpus`) and Qdrant. Public entry point for the
    seed-pipeline evaluation (spec C §4); release everything with `aclose`."""

    settings: RetrievalSettings
    service: RetrievalService
    retriever: QdrantHybridRetriever
    embedder: Embedder
    reranker: Reranker
    reader: CorpusReader
    qdrant: AsyncQdrantClient
    embed_client: AsyncOpenAI | None
    database: Database
    owns_database: bool

    async def aclose(self) -> None:
        await self.qdrant.close()
        if self.embed_client is not None:
            await self.embed_client.close()
        close_reranker = getattr(self.reranker, "aclose", None)
        if close_reranker is not None:
            await close_reranker()
        if self.owns_database:
            await self.database.dispose()


@dataclass
class Application:
    settings: Settings
    deps: TurnDeps
    runner: ChatTurnRunner
    retrieval: RetrievalStack

    async def aclose(self) -> None:
        await self.retrieval.aclose()


def _export_langfuse_environment(settings: Settings) -> None:
    """Langfuse's OpenAI integration reads its credentials from the environment."""
    if settings.langfuse.enabled:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse.public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse.secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse.host)


def build_retrieval_service(
    settings: Settings,
    *,
    database: Database | None = None,
    embedder: Embedder | None = None,
) -> RetrievalStack:
    """Build retrieval from settings.

    Pass `database` to reuse a pool the caller disposes, and `embedder` to replace the
    OpenAI-compatible query embedder (the evaluation injects a cached one).
    """
    _export_langfuse_environment(settings)
    retrieval_settings = settings.retrieval
    db = (
        database
        if database is not None
        else Database(
            settings.postgres.dsn,
            pool_size=settings.postgres.pool_size,
            echo=settings.postgres.echo,
        )
    )
    embed_client: AsyncOpenAI | None = None
    if embedder is not None:
        query_embedder = embedder
    else:
        embed_client = build_async_openai(
            api_key=retrieval_settings.embedding.api_key,
            base_url=retrieval_settings.embedding.base_url,
            timeout=retrieval_settings.embedding.timeout_seconds,
            max_retries=retrieval_settings.embedding.max_retries,
            traced=settings.langfuse.enabled,
        )
        query_embedder = OpenAiEmbedder(
            embed_client,
            model=retrieval_settings.embedding.model,
            dimension=retrieval_settings.embedding.dimension,
        )
    qdrant = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=int(settings.qdrant.timeout_seconds),
        check_compatibility=settings.qdrant.check_compatibility,
    )
    reader = PostgresCorpusReader(db.sessions)
    retriever = QdrantHybridRetriever(
        qdrant,
        query_embedder,
        reader,
        collection=retrieval_settings.qdrant_collection,
        scope=retrieval_settings.collections,
        mode=retrieval_settings.mode,
        prefetch_k=retrieval_settings.prefetch_k,
        rrf_k=retrieval_settings.rrf_k,
        max_concurrent=retrieval_settings.max_concurrent_searches,
    )
    reranker = build_reranker(retrieval_settings.rerank)
    if settings.langfuse.enabled:
        reranker = LangfuseTracedReranker(
            reranker,
            get_client(public_key=settings.langfuse.public_key),
            protocol=retrieval_settings.rerank.protocol,
            model=retrieval_settings.rerank.model,
        )
    service = RetrievalService(
        retriever,
        reranker,
        PostgresHydrator(reader, window=retrieval_settings.hydrate_window),
        RetrievalConfig(
            candidate_k=retrieval_settings.candidate_k,
            rerank_top_n=retrieval_settings.rerank.top_n,
            rerank_candidates=retrieval_settings.rerank.max_candidates,
        ),
    )
    return RetrievalStack(
        settings=retrieval_settings,
        service=service,
        retriever=retriever,
        embedder=query_embedder,
        reranker=reranker,
        reader=reader,
        qdrant=qdrant,
        embed_client=embed_client,
        database=db,
        owns_database=database is None,
    )


def build_application(
    settings: Settings,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    skills: SkillCatalog | None = None,
    tracer: TurnTracer | None = None,
    database: Database | None = None,
    llm: LlmPort | None = None,
    embedder: Embedder | None = None,
) -> Application:
    """`llm` and `embedder` replace the OpenAI-compatible adapters built from settings."""
    _export_langfuse_environment(settings)
    client_factory = (
        langfuse_client_factory if settings.langfuse.enabled else default_client_factory
    )
    llm_port: LlmPort = (
        llm
        if llm is not None
        else OpenAiLlmAdapter(settings.llm, client_factory=client_factory)
    )
    retrieval = build_retrieval_service(settings, database=database, embedder=embedder)
    deps = TurnDeps(
        llm=llm_port,
        guardrail=GuardrailService(llm_port),
        retrieval=retrieval.service,
        skills=skills
        if skills is not None
        else FileSystemSkillCatalog(settings.skills_dir),
        clock=SystemClock(),
    )
    runner = ChatTurnRunner(
        build_chat_graph(checkpointer=checkpointer),
        deps,
        settings.budget,
        tracer=tracer,
    )
    return Application(settings=settings, deps=deps, runner=runner, retrieval=retrieval)
