"""Retrieval audit derived from a finished AgentRun (one record per query per search round)."""

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.actions import ActionKind
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.shared.text import make_snippet


class RetrievalHitRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_version_id: UUID
    section_key: str
    table_key: str | None
    fusion_score: float
    rerank_score: float | None
    hydrate_strategy: str
    cited: bool
    snippet: str


class RetrievalRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: int
    query_text: str
    # str(collection_id) -> str(release_id) of the releases this query's hits came from.
    release_ids: dict[str, str] = Field(default_factory=dict)
    hits: list[RetrievalHitRecord] = Field(default_factory=list)


def audit_from_run(
    run: AgentRun, citations: Sequence[Citation], *, snippet_chars: int = 300
) -> list[RetrievalRunRecord]:
    cited_chunks = {citation.chunk_version_id for citation in citations}
    records: list[RetrievalRunRecord] = []
    search_actions = [a for a in run.actions.entries if a.kind is ActionKind.SEARCH]
    for round_number, action in enumerate(search_actions, start=1):
        queries = action.payload.get("queries", [])
        for query_text in queries if isinstance(queries, list) else []:
            matching = sorted(
                (e for e in run.evidence.items if query_text in e.hit.matched_queries),
                key=lambda e: e.score,
                reverse=True,
            )
            hits = [
                RetrievalHitRecord(
                    rank=rank,
                    chunk_version_id=evidence.hit.chunk_version_id,
                    section_key=evidence.hit.section_key,
                    table_key=evidence.hit.table_key,
                    fusion_score=evidence.hit.fusion_score,
                    rerank_score=evidence.hit.rerank_score,
                    hydrate_strategy=evidence.hit.hydrate_strategy.value,
                    cited=evidence.hit.chunk_version_id in cited_chunks,
                    snippet=make_snippet(evidence.hit.chunk_text, snippet_chars),
                )
                for rank, evidence in enumerate(matching, start=1)
            ]
            records.append(
                RetrievalRunRecord(
                    round=round_number,
                    query_text=str(query_text),
                    release_ids={
                        str(e.hit.collection_id): str(e.hit.release_id)
                        for e in matching
                    },
                    hits=hits,
                )
            )
    return records
