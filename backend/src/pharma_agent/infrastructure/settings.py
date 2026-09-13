from pathlib import Path
from typing import Literal

from openai.types import ReasoningEffort
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.llm.models import LlmRole

DEFAULT_ROLE_MODELS: dict[LlmRole, str] = {
    LlmRole.GUARDRAIL: "gpt-5-nano",
    LlmRole.REPHRASE: "gpt-5-nano",
    LlmRole.SKILL_SELECTOR: "gpt-5-nano",
    LlmRole.SUMMARIZER: "gpt-5-nano",
    LlmRole.JUDGE: "gpt-5-mini",
    LlmRole.REFINE: "gpt-5-mini",
    LlmRole.ANSWER: "gpt-5-mini",
}

# Applied only while a role runs its built-in model: these are reasoning models whose default
# effort ("medium") bills hidden reasoning tokens. Custom models get no reasoning_effort unless
# it is set explicitly, because many OpenAI-compatible servers reject the parameter.
DEFAULT_ROLE_REASONING: dict[LlmRole, ReasoningEffort] = {
    LlmRole.GUARDRAIL: "minimal",
    LlmRole.REPHRASE: "minimal",
    LlmRole.SKILL_SELECTOR: "minimal",
    LlmRole.SUMMARIZER: "minimal",
    LlmRole.JUDGE: "low",
    LlmRole.REFINE: "low",
    LlmRole.ANSWER: "low",
}


class LlmEndpoint(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    reasoning_effort: ReasoningEffort = None


class ResolvedEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str | None
    api_key: str
    model: str
    reasoning_effort: ReasoningEffort = None


class LlmSettings(BaseModel):
    default: LlmEndpoint = Field(default_factory=LlmEndpoint)
    roles: dict[LlmRole, LlmEndpoint] = Field(default_factory=dict)
    timeout_seconds: float = 60.0
    max_retries: int = 2

    @property
    def configured(self) -> bool:
        return bool(self.default.api_key) or all(
            (self.roles.get(role) or LlmEndpoint()).api_key for role in LlmRole
        )

    def resolve(self, role: LlmRole) -> ResolvedEndpoint:
        override = self.roles.get(role) or LlmEndpoint()
        api_key = override.api_key or self.default.api_key
        if not api_key:
            raise ValueError(
                f"no api_key for LLM role {role.value}: set PHARMA_LLM__DEFAULT__API_KEY"
            )
        model = override.model or self.default.model
        effort = override.reasoning_effort or self.default.reasoning_effort
        if model is None:
            model = DEFAULT_ROLE_MODELS[role]
            effort = effort or DEFAULT_ROLE_REASONING[role]
        return ResolvedEndpoint(
            base_url=override.base_url or self.default.base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=effort,
        )


class EmbeddingSettings(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "llama"
    model: str = "qwen3-embedding:4b-fp16"
    dimension: int = 2560
    timeout_seconds: float = 60.0
    max_retries: int = Field(default=2, ge=0)


class RerankSettings(BaseModel):
    protocol: Literal["completion_logprobs", "native_rerank", "none"] = (
        "completion_logprobs"
    )
    base_url: str = "http://localhost:11435"
    api_key: str | None = None
    model: str = "qwen3-reranker:4b-fp16"
    top_n: int = 8
    # Candidates scored per search round, taken round-robin across queries by fused rank.
    # A single query (candidate_k=30) is scored in full, like the evaluation.
    max_candidates: int = Field(default=40, ge=1)
    timeout_seconds: float = 120.0
    # Match the server's parallel slots (llama-server --parallel / LLAMA_RERANKER_PARALLEL).
    max_concurrent: int = Field(default=2, ge=1)


class RetrievalSettings(BaseModel):
    # Alias of the physical collection chunks_<embedding model slug> (spec C §8.3).
    qdrant_collection: str = "chunks_current"
    # Corpus collections the agent searches; each one needs a current release.
    collections: list[str] = Field(default_factory=lambda: ["formulary"])
    # bm25 = sparse only with no query embedding (evaluation baseline).
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    prefetch_k: int = 50
    rrf_k: int = 2
    candidate_k: int = 30
    hydrate_window: int = 1
    max_concurrent_searches: int = 3
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    check_compatibility: bool = True


class LangfuseSettings(BaseModel):
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


class PostgresSettings(BaseModel):
    dsn: str = "postgresql+psycopg://thesis:thesis@localhost:5433/thesis"
    pool_size: int = 10
    echo: bool = False

    @property
    def conninfo(self) -> str:
        """libpq connection string for psycopg (the checkpointer pool)."""
        return self.dsn.replace("postgresql+psycopg://", "postgresql://", 1)


class AuthSettings(BaseModel):
    jwt_secret: SecretStr | None = None
    jwt_lifetime_seconds: int = 7 * 24 * 3600
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    frontend_url: str = "http://localhost:3000"

    @field_validator("jwt_secret")
    @classmethod
    def _secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("jwt_secret must be at least 32 characters")
        return value

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    def require_jwt_secret(self) -> str:
        if self.jwt_secret is None:
            raise ValueError("set PHARMA_AUTH__JWT_SECRET (at least 32 characters)")
        return self.jwt_secret.get_secret_value()


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class MemorySettings(BaseModel):
    context_turns: int = 4
    context_chars: int = 4000
    summary_every_turns: int = 2
    summary_max_chars: int = 1500


class CheckpointSettings(BaseModel):
    retention_days: int = Field(default=7, ge=1)


class CorpusSettings(BaseModel):
    """Corpus import and release commands (spec C §8.2, §8.5)."""

    embed_batch_size: int = Field(default=32, ge=1)
    embed_max_concurrent: int = Field(default=1, ge=1)
    gc_keep: int = Field(default=2, ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHARMA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LlmSettings = Field(default_factory=LlmSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    checkpoints: CheckpointSettings = Field(default_factory=CheckpointSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    skills_dir: Path = Path("skills")
