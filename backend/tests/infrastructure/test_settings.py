import pytest

from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.infrastructure.settings import Settings


def test_defaults_match_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert (
        settings.retrieval.collection_alias == "thesis_chunks_qwen3_embedding_4b_fp16"
    )
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


def test_missing_api_key_means_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHARMA_LLM__DEFAULT__API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm.configured is False
    with pytest.raises(ValueError, match="api_key"):
        settings.llm.resolve(LlmRole.ANSWER)
