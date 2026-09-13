"""Adapters injected into the composition root replace the ones built from settings."""

from pathlib import Path

import pytest

from pharma_agent.infrastructure.composition import build_application
from pharma_agent.infrastructure.container import open_container
from pharma_agent.infrastructure.settings import Settings
from tests.fakes import FakeEmbedder, FakeLlm

DEVELOPER_ENV = (
    "PHARMA_LLM__DEFAULT__API_KEY",
    "PHARMA_LANGFUSE__PUBLIC_KEY",
    "PHARMA_LANGFUSE__SECRET_KEY",
)


def _without_llm_or_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in DEVELOPER_ENV:
        monkeypatch.delenv(name, raising=False)


async def test_build_application_uses_the_injected_llm_and_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_llm_or_langfuse(monkeypatch)
    # Without an API key, building the OpenAI adapter would raise: injection must skip it.
    settings = Settings(_env_file=None, qdrant={"check_compatibility": False})
    llm, embedder = FakeLlm(), FakeEmbedder()
    application = build_application(settings, llm=llm, embedder=embedder)
    try:
        assert application.deps.llm is llm
        assert application.retrieval.embedder is embedder
    finally:
        await application.aclose()


@pytest.mark.integration
async def test_injected_llm_builds_the_agent_without_an_api_key(
    migrated_dsn: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _without_llm_or_langfuse(monkeypatch)
    settings = Settings(
        _env_file=None,
        postgres={"dsn": migrated_dsn},
        qdrant={"check_compatibility": False},
        skills_dir=tmp_path,
    )
    assert settings.llm.configured is False
    async with open_container(
        settings, llm=FakeLlm(), embedder=FakeEmbedder()
    ) as container:
        assert container.chat is not None
        assert container.summarizer is not None
