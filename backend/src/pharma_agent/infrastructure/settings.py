from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
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


class LlmEndpoint(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class ResolvedEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str | None
    api_key: str
    model: str


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
        return ResolvedEndpoint(
            base_url=override.base_url or self.default.base_url,
            api_key=api_key,
            model=override.model or self.default.model or DEFAULT_ROLE_MODELS[role],
        )


class EmbeddingSettings(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "llama"
    model: str = "qwen3-embedding:4b-fp16"
    dimension: int = 2560


class RerankSettings(BaseModel):
    protocol: Literal["completion_logprobs", "native_rerank", "none"] = (
        "completion_logprobs"
    )
    base_url: str = "http://localhost:11435"
    model: str = "qwen3-reranker:4b-fp16"
    top_n: int = 8
    timeout_seconds: float = 120.0
    max_concurrent: int = 2


class RetrievalSettings(BaseModel):
    collection_alias: str = "thesis_chunks_qwen3_embedding_4b_fp16"
    mode: Literal["hybrid", "dense"] = "hybrid"
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


class LangfuseSettings(BaseModel):
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


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
    skills_dir: Path = Path("skills")
