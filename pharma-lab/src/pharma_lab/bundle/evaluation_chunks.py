"""Chunk rows for the evaluation dataset builders, derived from the backend chunker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pharma_agent.domain.corpus.bundle import BlockKind, DocumentKind, KnowledgeBundle
from pharma_agent.domain.corpus.chunking import ChunkDraft
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for

from pharma_lab.bundle.chunks import gold_chunk_label, iter_section_chunks

CHUNK_ROLES: dict[BlockKind, str] = {
    BlockKind.PROSE: "prose",
    BlockKind.TABLE: "table",
    BlockKind.INDEX_ENTRIES: "index_entry",
    BlockKind.LIST: "appendix_list",
}
CONTENT_TYPES: dict[DocumentKind, str] = {
    DocumentKind.DRUG_MONOGRAPH: "drug_monograph",
    DocumentKind.GENERAL_MONOGRAPH: "general_monograph",
    DocumentKind.LEAFLET: "brand_page",
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


def evaluation_chunk_rows(bundle: KnowledgeBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in iter_section_chunks(bundle):
        strategy = hydrate_strategy_for(item.section).value
        for draft in item.drafts:
            view = draft_view(draft, strategy)
            rows.append(
                {
                    "chunk_id": gold_chunk_label(draft.section_key, draft.ordinal),
                    "section_id": draft.section_key,
                    "chunk_index": draft.ordinal,
                    "title": item.document.title,
                    "section": item.section.heading,
                    "source": item.document.source.title,
                    "content_type": CONTENT_TYPES[item.document.kind],
                    "context_header": draft.context_header,
                    "context_path": list(item.section.context_path),
                    "chunk_text": draft.chunk_text,
                    "embedding_text": draft.embedding_text,
                    "hydrate_strategy": strategy,
                    "chunk_role": CHUNK_ROLES[draft.kind],
                    "chunk_content_type": (
                        "table" if draft.kind is BlockKind.TABLE else "paragraph"
                    ),
                    "table_id": draft.table_key or "",
                    "start_page": draft.start_page,
                    "end_page": draft.end_page,
                    "colloquial_mapping": view["colloquial_mapping"],
                    "term_annotations": view["term_annotations"],
                }
            )
    return rows


def write_evaluation_chunks(bundle: KnowledgeBundle, path: Path) -> int:
    rows = evaluation_chunk_rows(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)
