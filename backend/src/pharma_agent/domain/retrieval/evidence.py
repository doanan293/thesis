from uuid import UUID

from pydantic import BaseModel, Field

from pharma_agent.domain.retrieval.models import (
    Chunk,
    Hit,
    HydrateStrategy,
    RetrievedItem,
)
from pharma_agent.domain.shared.text import make_snippet


class Evidence(BaseModel):
    ref: str
    hit: Hit
    chunks: list[Chunk] = Field(default_factory=list)
    applied_strategy: HydrateStrategy
    superseded: bool = False

    @property
    def score(self) -> float:
        return self.hit.score

    def ordered_chunks(self) -> list[Chunk]:
        tables = [c for c in self.chunks if c.is_table]
        body = [c for c in self.chunks if not c.is_table]
        return sorted(tables, key=lambda c: c.ordinal) + sorted(
            body, key=lambda c: c.ordinal
        )

    def _uses_hit_text(self) -> bool:
        return self.applied_strategy is HydrateStrategy.SEARCH_ONLY or not self.chunks

    def text(self) -> str:
        if self._uses_hit_text():
            return self.hit.chunk_text
        return "\n\n".join(c.text for c in self.ordered_chunks())

    def text_chunk_version_ids(self) -> list[UUID]:
        """Chunk versions whose text `text()` joins, in the same order."""
        if self._uses_hit_text():
            return [self.hit.chunk_version_id]
        return [c.chunk_version_id for c in self.ordered_chunks()]

    def char_count(self) -> int:
        return len(self.text())


class EvidenceSet(BaseModel):
    items: list[Evidence] = Field(default_factory=list)

    def active(self) -> list[Evidence]:
        return sorted(
            (e for e in self.items if not e.superseded),
            key=lambda e: e.score,
            reverse=True,
        )

    def rerank_scores(self) -> dict[UUID, float]:
        """Rerank scores already computed in this run, superseded evidence included."""
        return {
            e.hit.chunk_version_id: e.hit.rerank_score
            for e in self.items
            if e.hit.rerank_score is not None
        }

    def supersede_all(self) -> None:
        for evidence in self.items:
            evidence.superseded = True

    def merge(self, retrieved: list[RetrievedItem]) -> list[Evidence]:
        by_chunk = {e.hit.chunk_version_id: e for e in self.items}
        for item in retrieved:
            existing = by_chunk.get(item.hit.chunk_version_id)
            if existing is None:
                evidence = Evidence(
                    ref=f"E{len(self.items) + 1}",
                    hit=item.hit,
                    chunks=list(item.chunks),
                    applied_strategy=item.hit.hydrate_strategy
                    if item.chunks
                    else HydrateStrategy.SEARCH_ONLY,
                )
                self.items.append(evidence)
                by_chunk[item.hit.chunk_version_id] = evidence
                continue
            merged_queries = list(existing.hit.matched_queries)
            merged_queries.extend(
                q for q in item.hit.matched_queries if q not in merged_queries
            )
            best_rerank = _max_optional(
                existing.hit.rerank_score, item.hit.rerank_score
            )
            existing.hit = item.hit.model_copy(
                update={
                    "rerank_score": best_rerank,
                    "fusion_score": max(
                        existing.hit.fusion_score, item.hit.fusion_score
                    ),
                    "matched_queries": merged_queries,
                }
            )
            if item.chunks:
                existing.chunks = list(item.chunks)
                existing.applied_strategy = item.hit.hydrate_strategy
            existing.superseded = False
        return self.active()

    def pack(self, max_chars: int) -> list[Evidence]:
        """Fit active evidence into max_chars by downgrading hydrate strategy, never truncating text."""
        packed: list[Evidence] = []
        remaining = max_chars
        for evidence in self.active():
            fitted = _fit(evidence, remaining)
            if fitted is None:
                continue
            packed.append(fitted)
            remaining -= fitted.char_count()
        return packed

    def summary_view(self, snippet_chars: int = 300) -> str:
        lines: list[str] = []
        for evidence in self.active():
            hit = evidence.hit
            parts = [
                evidence.ref,
                hit.context_header,
                hit.page_label,
                make_snippet(hit.chunk_text, snippet_chars),
            ]
            line = " | ".join(part for part in parts if part)
            hints = hit.term_hints()
            if hints:
                line += f" | gợi ý thuật ngữ: {', '.join(hints[:6])}"
            lines.append(line)
        return "\n".join(lines) if lines else "(không có evidence)"

    def context_view(
        self, packed: list[Evidence]
    ) -> tuple[str, list[tuple[int, Evidence]]]:
        numbered: list[tuple[int, Evidence]] = []
        blocks: list[str] = []
        for index, evidence in enumerate(packed, start=1):
            numbered.append((index, evidence))
            hit = evidence.hit
            label = f" ({hit.page_label})" if hit.page_label else ""
            blocks.append(f"[{index}] {hit.context_header}{label}\n{evidence.text()}")
        return "\n\n".join(blocks), numbered


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _window(evidence: Evidence, radius: int = 1) -> list[Chunk]:
    centre = evidence.hit.ordinal
    return [c for c in evidence.chunks if abs(c.ordinal - centre) <= radius]


def _fit(evidence: Evidence, remaining: int) -> Evidence | None:
    candidates: list[Evidence] = []
    if evidence.applied_strategy is HydrateStrategy.FULL_SECTION:
        candidates.append(evidence)
        candidates.append(
            evidence.model_copy(
                update={
                    "chunks": _window(evidence),
                    "applied_strategy": HydrateStrategy.CHUNK_WINDOW,
                }
            )
        )
    elif evidence.applied_strategy is HydrateStrategy.CHUNK_WINDOW:
        candidates.append(evidence)
    candidates.append(
        evidence.model_copy(
            update={"chunks": [], "applied_strategy": HydrateStrategy.SEARCH_ONLY}
        )
    )
    for candidate in candidates:
        if candidate.char_count() <= remaining:
            return candidate
    return None
