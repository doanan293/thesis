"""Builds the corpus import and release services with resources they own.

Used by `pharma-agent corpus ...`, which runs outside the HTTP container: it opens its own
database engine, Qdrant client and embedding client from Settings and closes them on exit.
"""

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.domain.corpus.ports import Embedder, VectorIndex
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.openai_client import build_async_openai
from pharma_agent.infrastructure.persistence.postgres.corpus_repository import (
    PostgresCorpusRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.embedding_cache import (
    PostgresEmbeddingCache,
)
from pharma_agent.infrastructure.retrieval.qdrant_adapter import OpenAiEmbedder
from pharma_agent.infrastructure.retrieval.qdrant_index import (
    CURRENT_ALIAS,
    QdrantVectorIndex,
)
from pharma_agent.infrastructure.settings import Settings

# A CLI run needs one connection at a time plus one for gc savepoint retries.
CORPUS_POOL_SIZE = 2


@dataclass
class CorpusServices:
    importer: ImportKnowledgeBundle
    releases: ReleaseService
    index: VectorIndex
    embedder: Embedder


@asynccontextmanager
async def open_corpus_services(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    alias: str = CURRENT_ALIAS,
) -> AsyncGenerator[CorpusServices]:
    """Open resources and yield the services; everything is closed on exit.

    `embedder` replaces the OpenAI-compatible embedder built from settings (tests and the
    E2E server pass `FakeEmbedder`); the Qdrant collection follows its model and dimension.
    `alias` is the Qdrant alias retrieval reads (P3 switches the default to
    `settings.retrieval.qdrant_collection`; the E2E server passes `e2e_chunks_current`).
    """
    database = Database(
        settings.postgres.dsn, pool_size=CORPUS_POOL_SIZE, echo=settings.postgres.echo
    )
    qdrant = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=int(settings.qdrant.timeout_seconds),
        check_compatibility=settings.qdrant.check_compatibility,
    )
    async with AsyncExitStack() as stack:
        stack.push_async_callback(database.dispose)
        stack.push_async_callback(qdrant.close)
        if embedder is None:
            embedding = settings.retrieval.embedding
            # Bulk corpus embeddings are not chat turns, so they are not sent to Langfuse.
            embed_client = build_async_openai(
                api_key=embedding.api_key,
                base_url=embedding.base_url,
                timeout=embedding.timeout_seconds,
                max_retries=embedding.max_retries,
                traced=False,
            )
            stack.push_async_callback(embed_client.close)
            embedder = OpenAiEmbedder(
                embed_client, model=embedding.model, dimension=embedding.dimension
            )
        repository = PostgresCorpusRepository(database.sessions)
        cache = PostgresEmbeddingCache(database.sessions)
        index = QdrantVectorIndex(
            qdrant, model=embedder.model, dimension=embedder.dimension, alias=alias
        )
        clock = SystemClock()
        corpus = settings.corpus
        yield CorpusServices(
            importer=ImportKnowledgeBundle(
                repository,
                cache,
                index,
                embedder,
                clock,
                embed_batch_size=corpus.embed_batch_size,
                embed_max_concurrent=corpus.embed_max_concurrent,
            ),
            releases=ReleaseService(
                repository,
                cache,
                index,
                embedder,
                clock,
                embed_batch_size=corpus.embed_batch_size,
                embed_max_concurrent=corpus.embed_max_concurrent,
            ),
            index=index,
            embedder=embedder,
        )
