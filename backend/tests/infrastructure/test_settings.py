import pytest

from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.infrastructure.settings import Settings


def test_defaults_match_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.retrieval.qdrant_collection == "chunks_current"
    assert settings.retrieval.collections == ["formulary"]
    assert settings.retrieval.mode == "hybrid"
    assert settings.retrieval.embedding.model == "qwen3-embedding:4b-fp16"
    assert settings.retrieval.embedding.dimension == 2560
    assert (
        settings.retrieval.prefetch_k,
        settings.retrieval.rrf_k,
        settings.retrieval.candidate_k,
    ) == (50, 2, 30)
    assert settings.retrieval.rerank.protocol == "completion_logprobs"
    assert settings.retrieval.rerank.model == "qwen3-reranker:4b-fp16"
    assert (
        settings.budget.max_llm_calls == 10 and settings.budget.deadline_seconds == 90
    )
    assert settings.llm.resolve(LlmRole.GUARDRAIL).model == "gpt-5-nano"
    assert settings.llm.resolve(LlmRole.ANSWER).model == "gpt-5-mini"
    assert settings.llm.resolve(LlmRole.ANSWER).api_key == "sk-test"
    assert settings.langfuse.enabled is False


def test_role_override_and_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-cloud")
    monkeypatch.setenv(
        "PHARMA_LLM__ROLES__ANSWER__BASE_URL", "http://localhost:8000/v1"
    )
    monkeypatch.setenv("PHARMA_LLM__ROLES__ANSWER__API_KEY", "local")
    monkeypatch.setenv("PHARMA_LLM__ROLES__ANSWER__MODEL", "qwen3-8b")
    monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "native_rerank")
    monkeypatch.setenv("PHARMA_QDRANT__URL", "http://qdrant:6333")
    monkeypatch.setenv("PHARMA_RETRIEVAL__COLLECTIONS", '["formulary", "leaflets"]')
    monkeypatch.setenv("PHARMA_RETRIEVAL__MODE", "bm25")
    settings = Settings(_env_file=None)
    answer = settings.llm.resolve(LlmRole.ANSWER)
    assert (answer.base_url, answer.api_key, answer.model) == (
        "http://localhost:8000/v1",
        "local",
        "qwen3-8b",
    )
    judge = settings.llm.resolve(LlmRole.JUDGE)
    assert (judge.base_url, judge.api_key, judge.model) == (
        None,
        "sk-cloud",
        "gpt-5-mini",
    )
    assert settings.retrieval.rerank.protocol == "native_rerank"
    assert settings.qdrant.url == "http://qdrant:6333"
    assert settings.retrieval.collections == ["formulary", "leaflets"]
    assert settings.retrieval.mode == "bm25"


def test_missing_api_key_means_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm.configured is False
    with pytest.raises(ValueError, match="api_key"):
        settings.llm.resolve(LlmRole.ANSWER)


def test_platform_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PHARMA_AUTH__JWT_SECRET", "PHARMA_POSTGRES__DSN"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert (
        settings.postgres.dsn
        == "postgresql+psycopg://thesis:thesis@localhost:5433/thesis"
    )
    assert (
        settings.postgres.conninfo == "postgresql://thesis:thesis@localhost:5433/thesis"
    )
    assert settings.memory.summary_every_turns == 2
    assert settings.memory.context_turns == 4
    assert settings.auth.google_enabled is False
    with pytest.raises(ValueError, match="PHARMA_AUTH__JWT_SECRET"):
        settings.auth.require_jwt_secret()


def test_auth_secret_and_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_AUTH__JWT_SECRET", "x" * 40)
    monkeypatch.setenv("PHARMA_AUTH__GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("PHARMA_AUTH__GOOGLE_CLIENT_SECRET", "secret")
    settings = Settings(_env_file=None)
    assert settings.auth.require_jwt_secret() == "x" * 40
    assert settings.auth.google_enabled is True


def test_short_jwt_secret_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_AUTH__JWT_SECRET", "short")
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None)


def test_reasoning_effort_defaults_apply_only_to_built_in_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-cloud")
    monkeypatch.setenv("PHARMA_LLM__ROLES__ANSWER__MODEL", "qwen3-8b")
    monkeypatch.setenv("PHARMA_LLM__ROLES__JUDGE__REASONING_EFFORT", "medium")
    settings = Settings(_env_file=None)
    assert settings.llm.resolve(LlmRole.GUARDRAIL).reasoning_effort == "minimal"
    assert settings.llm.resolve(LlmRole.SUMMARIZER).reasoning_effort == "minimal"
    assert settings.llm.resolve(LlmRole.REFINE).reasoning_effort == "low"
    assert settings.llm.resolve(LlmRole.JUDGE).reasoning_effort == "medium"
    assert settings.llm.resolve(LlmRole.ANSWER).reasoning_effort is None


def test_embedding_and_rerank_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = Settings(_env_file=None).retrieval
    assert (defaults.embedding.timeout_seconds, defaults.embedding.max_retries) == (
        60.0,
        2,
    )
    assert defaults.rerank.api_key is None
    assert (defaults.rerank.max_candidates, defaults.rerank.max_concurrent) == (40, 2)

    monkeypatch.setenv("PHARMA_RETRIEVAL__EMBEDDING__TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("PHARMA_RETRIEVAL__EMBEDDING__MAX_RETRIES", "0")
    monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__API_KEY", "rerank-secret")
    monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__MAX_CANDIDATES", "20")
    configured = Settings(_env_file=None).retrieval
    assert (configured.embedding.timeout_seconds, configured.embedding.max_retries) == (
        5.0,
        0,
    )
    assert configured.rerank.api_key == "rerank-secret"
    assert configured.rerank.max_candidates == 20


def test_corpus_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = Settings(_env_file=None).corpus
    assert (corpus.embed_batch_size, corpus.embed_max_concurrent, corpus.gc_keep) == (
        32,
        1,
        2,
    )
    monkeypatch.setenv("PHARMA_CORPUS__EMBED_MAX_CONCURRENT", "4")
    monkeypatch.setenv("PHARMA_CORPUS__GC_KEEP", "0")
    configured = Settings(_env_file=None).corpus
    assert (configured.embed_max_concurrent, configured.gc_keep) == (4, 0)
    monkeypatch.setenv("PHARMA_CORPUS__EMBED_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="embed_batch_size"):
        Settings(_env_file=None)
