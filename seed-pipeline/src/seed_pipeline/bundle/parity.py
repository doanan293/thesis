"""Compare `chunk_section` output with the pre-migration `rag-final/chunks.jsonl` (spec §7.3).

Only two normalizations are applied: an old page 0 means "no page", and an old leaflet
mapping without `key` (an uncurated leaflet, keyed by its slug in the bundle, P1) is
compared without that key.
Chunk identifiers are not compared; `chunk_index` must equal the draft ordinal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from pharma_agent.domain.corpus.bundle import KnowledgeBundle
from pharma_agent.domain.corpus.chunking import MAX_CHUNK_CHARS, ChunkDraft
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for

from seed_pipeline.bundle.chunks import iter_section_chunks

COMPARED_FIELDS = (
    "chunk_text",
    "start_page",
    "end_page",
    "hydrate_strategy",
    "embedding_text",
    "term_annotations",
    "colloquial_mapping",
)


@dataclass(frozen=True)
class ParityMismatch:
    section_key: str
    ordinal: int | None
    field: str
    old: object
    new: object


@dataclass
class ParityReport:
    sections_checked: int = 0
    chunks_checked: int = 0
    mismatch_count: int = 0
    mismatches: list[ParityMismatch] = field(default_factory=list)
    max_recorded: int = 50

    @property
    def ok(self) -> bool:
        return self.mismatch_count == 0

    def record(self, mismatch: ParityMismatch) -> None:
        self.mismatch_count += 1
        if len(self.mismatches) < self.max_recorded:
            self.mismatches.append(mismatch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sections_checked": self.sections_checked,
            "chunks_checked": self.chunks_checked,
            "mismatch_count": self.mismatch_count,
            "mismatches": [asdict(mismatch) for mismatch in self.mismatches],
        }


def legacy_chunk_view(chunk: Mapping[str, Any]) -> dict[str, object]:
    mapping = dict(chunk.get("colloquial_mapping") or {})
    return {
        "chunk_text": str(chunk["chunk_text"]),
        "start_page": _legacy_page(chunk.get("start_page")),
        "end_page": _legacy_page(chunk.get("end_page")),
        "hydrate_strategy": str(chunk.get("hydrate_strategy") or ""),
        "embedding_text": str(chunk["embedding_text"]),
        "term_annotations": list(chunk.get("term_annotations") or []),
        "colloquial_mapping": mapping,
    }


def draft_view(draft: ChunkDraft, hydrate_strategy: str) -> dict[str, object]:
    colloquial = (
        {}
        if draft.colloquial is None
        else {
            name: value
            for name, value in draft.colloquial.model_dump().items()
            if value not in ("", [])
        }
    )
    return {
        "chunk_text": draft.chunk_text,
        "start_page": draft.start_page,
        "end_page": draft.end_page,
        "hydrate_strategy": hydrate_strategy,
        "embedding_text": draft.embedding_text,
        "term_annotations": [term.model_dump() for term in draft.term_annotations],
        "colloquial_mapping": colloquial,
    }


def check_chunk_parity(
    bundle: KnowledgeBundle,
    old_chunks: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    max_recorded: int = 50,
) -> ParityReport:
    old_by_section: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in old_chunks:
        old_by_section[str(chunk["section_id"])].append(chunk)
    report = ParityReport(max_recorded=max_recorded)
    for item in iter_section_chunks(bundle, max_chars=max_chars):
        section_key = item.section.key
        old = sorted(
            old_by_section.pop(section_key, []),
            key=lambda chunk: int(chunk["chunk_index"]),
        )
        report.sections_checked += 1
        strategy = hydrate_strategy_for(item.section).value
        if len(old) != len(item.drafts):
            report.record(
                ParityMismatch(
                    section_key, None, "chunk_count", len(old), len(item.drafts)
                )
            )
        for chunk, draft in zip(old, item.drafts, strict=False):
            report.chunks_checked += 1
            old_ordinal = int(chunk["chunk_index"])
            if old_ordinal != draft.ordinal:
                report.record(
                    ParityMismatch(
                        section_key,
                        draft.ordinal,
                        "ordinal",
                        old_ordinal,
                        draft.ordinal,
                    )
                )
            before = legacy_chunk_view(chunk)
            after = draft_view(draft, strategy)
            after["colloquial_mapping"] = comparable_mapping(
                before["colloquial_mapping"], after["colloquial_mapping"]
            )
            for name in COMPARED_FIELDS:
                if before[name] != after[name]:
                    report.record(
                        ParityMismatch(
                            section_key, draft.ordinal, name, before[name], after[name]
                        )
                    )
    for section_key, chunks in sorted(old_by_section.items()):
        report.record(
            ParityMismatch(section_key, None, "missing_section", len(chunks), 0)
        )
    return report


def comparable_mapping(old: object, new: object) -> object:
    """Drop the bundle's slug key where the old build had no key for that leaflet."""
    if isinstance(old, dict) and isinstance(new, dict) and "key" not in old:
        return {name: value for name, value in new.items() if name != "key"}
    return new


def _legacy_page(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value
