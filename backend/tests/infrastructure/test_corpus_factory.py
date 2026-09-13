import pytest

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.infrastructure.corpus_factory import open_corpus_services
from pharma_agent.infrastructure.retrieval.qdrant_adapter import OpenAiEmbedder
from pharma_agent.infrastructure.retrieval.qdrant_index import QdrantVectorIndex
from pharma_agent.infrastructure.settings import Settings
from tests.fakes import FakeEmbedder


async def test_open_corpus_services_wires_real_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    monkeypatch.setenv("PHARMA_RETRIEVAL__EMBEDDING__MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("PHARMA_RETRIEVAL__EMBEDDING__DIMENSION", "1024")
    settings = Settings(_env_file=None)

    async with open_corpus_services(settings) as services:
        assert isinstance(services.importer, ImportKnowledgeBundle)
        assert isinstance(services.releases, ReleaseService)
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.collection_name == "chunks_qwen3_embedding_0_6b"
        assert isinstance(services.embedder, OpenAiEmbedder)
        assert (services.embedder.model, services.embedder.dimension) == (
            "qwen3-embedding:0.6b",
            1024,
        )


async def test_open_corpus_services_uses_an_injected_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    embedder = FakeEmbedder()

    async with open_corpus_services(
        Settings(_env_file=None), embedder=embedder
    ) as services:
        assert services.embedder is embedder
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.collection_name == "chunks_fake_embedding_4d"
        assert services.index.alias == "chunks_current"

    async with open_corpus_services(
        Settings(_env_file=None), embedder=embedder, alias="e2e_chunks_current"
    ) as services:
        assert isinstance(services.index, QdrantVectorIndex)
        assert services.index.alias == "e2e_chunks_current"
