from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HydrateStrategy(StrEnum):
    FULL_SECTION = "full_section"
    CHUNK_WINDOW = "chunk_window"
    SEARCH_ONLY = "search_only"


class QueryOrigin(StrEnum):
    INITIAL = "initial"
    REFINED = "refined"


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    origin: QueryOrigin

    @property
    def normalized(self) -> str:
        return " ".join(self.text.lower().split())


class TermAnnotation(BaseModel):
    term: str
    vi: list[str] = Field(default_factory=list)
    en: list[str] = Field(default_factory=list)


class ColloquialMapping(BaseModel):
    key: str = ""
    aliases: list[str] = Field(default_factory=list)
    visual_sign: str = ""
    product_names: list[str] = Field(default_factory=list)


def page_label(start_page: int | None, end_page: int | None) -> str:
    """Vietnamese page label; empty when the source has no page numbers (leaflets)."""
    if start_page is None and end_page is None:
        return ""
    if start_page is None or end_page is None or start_page == end_page:
        return f"trang {start_page if start_page is not None else end_page}"
    return f"trang {start_page}-{end_page}"


class Hit(BaseModel):
    """One chunk version found by search, as published in one release, plus scores."""

    chunk_version_id: UUID
    release_id: UUID
    collection_id: UUID
    document_key: str
    section_key: str
    section_revision_id: UUID
    ordinal: int
    hydrate_strategy: HydrateStrategy
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    context_header: str
    chunk_text: str
    # What was embedded and what the reranker scores, unchanged from today.
    embedding_text: str
    kind: str
    table_key: str | None
    colloquial_mapping: ColloquialMapping | None = None
    term_annotations: list[TermAnnotation] = Field(default_factory=list)
    fusion_score: float = 0.0
    rerank_score: float | None = None
    matched_queries: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fusion_score

    @property
    def page_label(self) -> str:
        return page_label(self.start_page, self.end_page)

    def term_hints(self) -> list[str]:
        hints: list[str] = []
        if self.colloquial_mapping is not None:
            hints.extend(self.colloquial_mapping.aliases)
            hints.extend(self.colloquial_mapping.product_names)
        for annotation in self.term_annotations:
            hints.append(annotation.term)
            hints.extend(annotation.vi)
            hints.extend(annotation.en)
        seen: set[str] = set()
        unique: list[str] = []
        for hint in hints:
            key = hint.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(hint.strip())
        return unique


class Chunk(BaseModel):
    """One chunk of a section revision in the release a hit came from."""

    chunk_version_id: UUID
    section_revision_id: UUID
    ordinal: int
    text: str
    kind: str
    table_key: str | None = None
    start_page: int | None = None
    end_page: int | None = None

    @property
    def is_table(self) -> bool:
        return self.kind == "table"


class RetrievedItem(BaseModel):
    hit: Hit
    chunks: list[Chunk] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """One chunk version as published in one release, with the fields a search hit shows."""

    model_config = ConfigDict(frozen=True)

    chunk_version_id: UUID
    release_id: UUID
    collection_id: UUID
    document_key: str
    section_key: str
    section_revision_id: UUID
    ordinal: int
    hydrate_strategy: HydrateStrategy
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    context_header: str
    chunk_text: str
    embedding_text: str
    kind: str
    table_key: str | None
    colloquial_mapping: ColloquialMapping | None = None
    term_annotations: list[TermAnnotation] = Field(default_factory=list)

    def to_hit(self, *, fusion_score: float, query_text: str) -> Hit:
        return Hit.model_validate(
            {
                **self.model_dump(),
                "fusion_score": fusion_score,
                "matched_queries": [query_text],
            }
        )
