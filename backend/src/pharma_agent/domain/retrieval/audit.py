"""Retrieval audit derived from a finished AgentRun (one record per query per search round)."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.actions import ActionKind
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.conversation.models import Citation


class RetrievalHitRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_id: str
    section_id: str
    table_id: str
    fusion_score: float
    rerank_score: float | None
    hydrate_strategy: str
    cited: bool
    snippet: str


class RetrievalRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    round: int
    query_text: str
    hits: list[RetrievalHitRecord] = Field(default_factory=list)


def audit_from_run(
    run: AgentRun, citations: Sequence[Citation], *, snippet_chars: int = 300
) -> list[RetrievalRunRecord]:
    cited_chunks = {citation.chunk_id for citation in citations}
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
                    chunk_id=evidence.hit.chunk_id,
                    section_id=evidence.hit.section_id,
                    table_id=evidence.hit.table_id,
                    fusion_score=evidence.hit.fusion_score,
                    rerank_score=evidence.hit.rerank_score,
                    hydrate_strategy=evidence.hit.hydrate_strategy.value,
                    cited=evidence.hit.chunk_id in cited_chunks,
                    snippet=" ".join(evidence.hit.chunk_text.split())[:snippet_chars],
                )
                for rank, evidence in enumerate(matching, start=1)
            ]
            records.append(
                RetrievalRunRecord(
                    round=round_number, query_text=str(query_text), hits=hits
                )
            )
    return records
