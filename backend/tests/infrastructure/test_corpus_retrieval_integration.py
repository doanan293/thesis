"""Corpus platform end to end on real Postgres and Qdrant (spec C §12).

P2's factory imports the fixture bundle and the retrieval factory searches it; only the
embedder is replaced, by the deterministic FakeEmbedder, on both sides.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import pytest
from qdrant_client import AsyncQdrantClient

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    DocumentKind,
    KnowledgeBundle,
    SectionRecord,
)
from pharma_agent.domain.corpus.chunking import chunk_section
from pharma_agent.domain.retrieval.models import (
    Hit,
    HydrateStrategy,
    Query,
    QueryOrigin,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.composition import (
    RetrievalStack,
    build_retrieval_service,
)
from pharma_agent.infrastructure.corpus_factory import (
    CorpusServices,
    open_corpus_services,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.retrieval.postgres_corpus import PostgresHydrator
from pharma_agent.infrastructure.settings import Settings
from tests.corpus_fixtures import COLLECTION_KEY, DOSAGE_SECTION, small_bundle
from tests.corpus_rows import long_paragraph, reset_corpus
from tests.fakes import FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_MODEL, FakeEmbedder

pytestmark = pytest.mark.integration

LONG_SECTION_SUFFIX = ":lieu-dung-keo-dai"
# Seven blocks of about 2,500 characters: one chunk each, and the section is longer than
# FULL_SECTION_MAX_CHARS, so the hydrate policy picks chunk_window.
FIRST_MARKERS = [f"kxdoan{i}" for i in range(1, 8)]
EDITED_MARKERS = ["kxmoi1", *FIRST_MARKERS[1:]]


def settings_for(
    dsn: str,
    qdrant_url: str,
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    *,
    qdrant_collection: str | None = None,
) -> Settings:
    """Both factories read the Qdrant alias from `retrieval.qdrant_collection`; without an
    override it keeps the settings default."""
    return Settings(
        _env_file=None,
        postgres={"dsn": dsn},
        qdrant={"url": qdrant_url},
        retrieval={
            "mode": mode,
            "collections": [COLLECTION_KEY],
            "prefetch_k": 100,
            "embedding": {
                "model": FAKE_EMBEDDING_MODEL,
                "dimension": FAKE_EMBEDDING_DIMENSION,
            },
            "rerank": {"protocol": "none"},
            **({"qdrant_collection": qdrant_collection} if qdrant_collection else {}),
        },
    )


@dataclass
class Platform:
    corpus: CorpusServices
    retrieval: RetrievalStack
    hydrator: PostgresHydrator
    dsn: str
    qdrant_url: str


@pytest.fixture
async def platform(
    migrated_dsn: str, qdrant_url: str, qdrant_client: AsyncQdrantClient
) -> AsyncIterator[Platform]:
    """Empty corpus schema and Qdrant (`qdrant_client` wipes every collection)."""
    database = Database(migrated_dsn, pool_size=1)
    await reset_corpus(database)
    await database.dispose()
    settings = settings_for(migrated_dsn, qdrant_url)
    async with open_corpus_services(settings, embedder=FakeEmbedder()) as corpus:
        retrieval = build_retrieval_service(settings, embedder=FakeEmbedder())
        try:
            yield Platform(
                corpus=corpus,
                retrieval=retrieval,
                hydrator=PostgresHydrator(
                    retrieval.reader, window=settings.retrieval.hydrate_window
                ),
                dsn=migrated_dsn,
                qdrant_url=qdrant_url,
            )
        finally:
            await retrieval.aclose()


def long_section(bundle: KnowledgeBundle, markers: Sequence[str]) -> SectionRecord:
    document = next(
        d for d in bundle.documents if d.kind is DocumentKind.DRUG_MONOGRAPH
    )
    ordinal = 1 + max(
        s.ordinal for s in bundle.sections if s.document_key == document.key
    )
    return SectionRecord(
        key=f"{document.key}{LONG_SECTION_SUFFIX}",
        document_key=document.key,
        heading="Liều dùng kéo dài",
        context_path=["Liều dùng kéo dài"],
        ordinal=ordinal,
        blocks=[
            BlockRecord(kind=BlockKind.PROSE, markdown=long_paragraph(marker))
            for marker in markers
        ],
    )


def bundle_with_long_section(markers: Sequence[str]) -> KnowledgeBundle:
    """The fixture bundle plus one long section, edited in memory like P2's
    `with_section_text`. Its chunks have no bundled vectors, so the import embeds them."""
    base = small_bundle()
    return base.model_copy(
        update={"sections": [*base.sections, long_section(base, markers)]}
    )


async def search(stack: RetrievalStack, text: str) -> list[Hit]:
    [hits] = await stack.retriever.search_many(
        [Query(text=text, origin=QueryOrigin.INITIAL)], top_k=10
    )
    return hits


def with_marker(hits: Sequence[Hit], marker: str) -> list[Hit]:
    return [hit for hit in hits if f"Mã đoạn {marker}." in hit.chunk_text]


async def test_imported_bundle_is_searchable_and_hydrates_both_strategies(
    platform: Platform,
) -> None:
    retrieval = platform.retrieval
    with pytest.raises(RetrievalError):
        await retrieval.retriever.verify_corpus(
            embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
        )

    bundle = bundle_with_long_section(FIRST_MARKERS)
    await platform.corpus.importer(bundle, publish=True)

    await retrieval.retriever.verify_corpus(
        embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
    )
    releases = await retrieval.reader.current_releases([COLLECTION_KEY])
    assert len(releases) == 1

    [top] = with_marker(await search(retrieval, "kxdoan4"), "kxdoan4")
    assert releases[top.collection_id] == top.release_id
    assert top.section_key.endswith(LONG_SECTION_SUFFIX)
    assert top.hydrate_strategy is HydrateStrategy.CHUNK_WINDOW

    section = next(s for s in bundle.sections if s.key == top.section_key)
    document = next(d for d in bundle.documents if d.key == section.document_key)
    drafts = chunk_section(
        document, section, bundle.glossary, bundle.colloquial_mappings
    )
    draft = next(d for d in drafts if d.chunk_version_id == top.chunk_version_id)
    assert (top.context_header, top.embedding_text) == (
        draft.context_header,
        draft.embedding_text,
    )

    full = await platform.hydrator.hydrate(top, HydrateStrategy.FULL_SECTION)
    assert [c.chunk_version_id for c in full] == [d.chunk_version_id for d in drafts]
    window = await platform.hydrator.hydrate(top, HydrateStrategy.CHUNK_WINDOW)
    assert [c.ordinal for c in window] == [
        d.ordinal for d in drafts if abs(d.ordinal - top.ordinal) <= 1
    ]
    assert len(window) == 3
    assert await platform.hydrator.hydrate(top, HydrateStrategy.SEARCH_ONLY) == []

    dosage = next(s for s in bundle.sections if s.key == DOSAGE_SECTION)
    dosage_hits = await search(retrieval, dosage.blocks[0].markdown[:200])
    assert DOSAGE_SECTION in {hit.section_key for hit in dosage_hits}

    # The composed RetrievalService hydrates through the same PostgresHydrator.
    result = await retrieval.service.search(
        [Query(text="kxdoan4", origin=QueryOrigin.INITIAL)], rerank_query="kxdoan4"
    )
    item = next(
        i for i in result.items if i.hit.chunk_version_id == top.chunk_version_id
    )
    assert [c.ordinal for c in item.chunks] == [c.ordinal for c in window]


async def test_bm25_mode_searches_the_imported_release_without_embedding(
    platform: Platform,
) -> None:
    await platform.corpus.importer(
        bundle_with_long_section(FIRST_MARKERS), publish=True
    )
    embedder = FakeEmbedder()
    stack = build_retrieval_service(
        settings_for(platform.dsn, platform.qdrant_url, "bm25"), embedder=embedder
    )
    try:
        hits = await search(stack, "kxdoan6")
    finally:
        await stack.aclose()

    assert with_marker(hits[:1], "kxdoan6")
    assert embedder.batches == []


async def test_custom_qdrant_collection_is_used_for_import_and_search(
    platform: Platform, qdrant_client: AsyncQdrantClient
) -> None:
    settings = settings_for(
        platform.dsn, platform.qdrant_url, qdrant_collection="p3_custom_alias"
    )
    async with open_corpus_services(settings, embedder=FakeEmbedder()) as corpus:
        await corpus.importer(bundle_with_long_section(FIRST_MARKERS), publish=True)

    aliases = (await qdrant_client.get_aliases()).aliases
    assert {a.alias_name: a.collection_name for a in aliases} == {
        "p3_custom_alias": "chunks_fake_embedding_4d"
    }
    stack = build_retrieval_service(settings, embedder=FakeEmbedder())
    try:
        await stack.retriever.verify_corpus(
            embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
        )
        assert with_marker(await search(stack, "kxdoan4"), "kxdoan4")
    finally:
        await stack.aclose()
    # Nothing was written under the default alias, so retrieval on defaults is not ready.
    with pytest.raises(RetrievalError):
        await platform.retrieval.retriever.verify_corpus(
            embedding_model=FAKE_EMBEDDING_MODEL, dimension=FAKE_EMBEDDING_DIMENSION
        )


async def test_publish_and_rollback_switch_search_results(platform: Platform) -> None:
    retrieval = platform.retrieval
    await platform.corpus.importer(
        bundle_with_long_section(FIRST_MARKERS), publish=True
    )
    first = await retrieval.reader.current_releases([COLLECTION_KEY])

    await platform.corpus.importer(
        bundle_with_long_section(EDITED_MARKERS), publish=True
    )
    second = await retrieval.reader.current_releases([COLLECTION_KEY])
    assert second.keys() == first.keys() and second != first

    edited = await search(retrieval, "kxmoi1")
    assert len(with_marker(edited, "kxmoi1")) == 1
    assert {hit.release_id for hit in edited} == set(second.values())
    assert with_marker(await search(retrieval, "kxdoan1"), "kxdoan1") == []
    [unchanged] = with_marker(await search(retrieval, "kxdoan4"), "kxdoan4")
    assert unchanged.release_id in second.values()

    await platform.corpus.releases.rollback(COLLECTION_KEY)
    assert await retrieval.reader.current_releases([COLLECTION_KEY]) == first

    restored = await search(retrieval, "kxdoan1")
    [original] = with_marker(restored, "kxdoan1")
    assert {hit.release_id for hit in restored} == set(first.values())
    assert with_marker(await search(retrieval, "kxmoi1"), "kxmoi1") == []
    window = await platform.hydrator.hydrate(original, HydrateStrategy.CHUNK_WINDOW)
    assert window and all("kxmoi1" not in chunk.text for chunk in window)
