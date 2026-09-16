"""Per-item records of an E2E run, stored as append-only JSONL (spec §6.2)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_RECORD = ConfigDict(extra="forbid")


class TokenUsage(BaseModel):
    model_config = _RECORD

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class CitedSection(BaseModel):
    model_config = _RECORD

    index: int
    chunk_version_id: str
    chunk_id: str | None
    section_id: str | None


class ItemRecord(BaseModel):
    """A record about one golden item within one configuration."""

    model_config = _RECORD

    item_id: str
    config: str


class AnswerRecord(ItemRecord):
    status: str
    answer_mode: str | None = None
    answer_text: str = ""
    context_text: str = ""
    citations: list[CitedSection] = Field(default_factory=list)
    # Gold labels of every chunk the final evidence set held, in rank order.
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    standalone_query: str = ""
    search_queries: list[list[str]] = Field(default_factory=list)
    judge_outcomes: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    usage_by_role: dict[str, TokenUsage] = Field(default_factory=dict)
    latency_seconds: float = 0.0
    # Wall time outside LLM calls: retrieval, reranking and graph overhead.
    non_llm_seconds: float = 0.0
    error: str | None = None
    retryable: bool = False

    @classmethod
    def failed(cls, item_id: str, config: str, error: BaseException) -> AnswerRecord:
        return cls(
            item_id=item_id,
            config=config,
            status="error",
            error=f"{type(error).__name__}: {error}"[:500],
            retryable=True,
        )

    @property
    def total_tokens(self) -> int:
        return sum(
            usage.prompt_tokens + usage.completion_tokens
            for usage in self.usage_by_role.values()
        )


class Judgement(ItemRecord):
    category: str
    behaviour_correct: bool
    injection_followed: bool | None = None
    key_fact_verdicts: list[str] | None = None
    key_fact_recall: float | None = None
    contradiction: bool | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    citation_support: float | None = None
    faithfulness: float | None = None
    factual_correctness: float | None = None
    answer_relevancy: float | None = None
    error: str | None = None


class JsonlStore[T: ItemRecord]:
    """Append-only records keyed by `item_id`; the last line for an item wins."""

    def __init__(self, path: Path, model: type[T]) -> None:
        self.path = Path(path)
        self._model = model

    def latest(self) -> dict[str, T]:
        records: dict[str, T] = {}
        if not self.path.is_file():
            return records
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = self._model.model_validate_json(line)
                    records[record.item_id] = record
        return records

    def append(self, record: T) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
