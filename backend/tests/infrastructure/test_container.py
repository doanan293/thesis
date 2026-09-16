import pytest

from pharma_agent.infrastructure.container import CORPUS_NOT_READY, open_container
from pharma_agent.infrastructure.settings import Settings

pytestmark = pytest.mark.integration


async def test_container_without_llm_serves_queries_only(
    migrated_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    async with open_container(Settings(_env_file=None)) as container:
        assert container.chat is None and container.summarizer is None
        assert container.feedback is not None
        assert await container.health_checks["postgres"]() is True
        assert container.health_reasons == {}
        page = await container.queries.list_conversations("a" * 32, limit=5)
        assert page.items == [] and page.next_cursor is None


async def test_container_with_llm_builds_chat_and_corpus_check(
    migrated_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PHARMA_POSTGRES__DSN", migrated_dsn)
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_QDRANT__CHECK_COMPATIBILITY", "false")
    # Nothing listens on the discard port, so the Qdrant call fails fast.
    monkeypatch.setenv("PHARMA_QDRANT__URL", "http://127.0.0.1:9")
    async with open_container(Settings(_env_file=None)) as container:
        assert container.chat is not None and container.summarizer is not None
        assert set(container.health_checks) == {"postgres", "corpus"}
        assert container.health_reasons == {"corpus": CORPUS_NOT_READY}
        assert await container.health_checks["corpus"]() is False
