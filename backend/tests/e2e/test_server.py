import pytest
from fastapi import FastAPI

from tests.corpus_fixtures import COLLECTION_KEY
from tests.e2e import server
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL


def test_config_reads_the_e2e_environment() -> None:
    assert server.E2EConfig.from_env({}) == server.E2EConfig()
    config = server.E2EConfig.from_env(
        {
            "E2E_POSTGRES_DSN": "postgresql+psycopg://u:p@db:5432/app_e2e",
            "E2E_QDRANT_URL": "http://qdrant:6333",
        }
    )
    assert (config.postgres_dsn, config.qdrant_url) == (
        "postgresql+psycopg://u:p@db:5432/app_e2e",
        "http://qdrant:6333",
    )


def test_settings_ignore_the_developer_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_LANGFUSE__PUBLIC_KEY", "pk-lf-dev")
    monkeypatch.setenv("PHARMA_LANGFUSE__SECRET_KEY", "sk-lf-dev")
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-dev")
    monkeypatch.setenv(
        "PHARMA_POSTGRES__DSN", "postgresql+psycopg://u:p@elsewhere:5432/thesis"
    )
    settings = server.e2e_settings(server.E2EConfig())
    assert settings.langfuse.enabled is False
    assert settings.llm.configured is False
    assert settings.postgres.dsn == server.DEFAULT_POSTGRES_DSN
    assert settings.auth.cookie_secure is False
    assert settings.auth.require_csrf_secret() == server.E2E_CSRF_SECRET
    assert settings.retrieval.embedding.model == FAKE_EMBEDDING_MODEL
    assert settings.retrieval.embedding.dimension == FAKE_EMBEDDING_DIMENSION
    assert settings.retrieval.qdrant_collection == "e2e_chunks_current"
    assert settings.retrieval.collections == [COLLECTION_KEY]
    assert settings.retrieval.rerank.protocol == "none"
    assert settings.budget.deadline_seconds == 5.0


def test_reset_refuses_a_database_without_e2e_in_its_name() -> None:
    with pytest.raises(ValueError, match="must contain 'e2e'"):
        server.reset_database(
            "postgresql+psycopg://thesis:thesis@localhost:5433/thesis"
        )


def test_main_serves_the_e2e_app_with_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        server.uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs)
    )
    server.main(["--port", "8001"])
    assert isinstance(calls["app"], FastAPI)
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8001)
