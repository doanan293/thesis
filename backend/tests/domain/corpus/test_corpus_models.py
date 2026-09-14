import uuid

import pytest

from pharma_agent.domain.corpus.bundle import BlockKind
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION
from pharma_agent.domain.corpus.identity import CORPUS_NAMESPACE
from pharma_agent.domain.corpus.models import (
    CorpusImportError,
    Release,
    ReleaseStatus,
    Visibility,
    build_snapshot,
    bundle_digest,
    collection_id_for,
    document_id_for,
    releases_to_retire,
    section_id_for,
)
from pharma_agent.domain.corpus.ports import Embedder
from pharma_agent.domain.retrieval.models import HydrateStrategy
from tests.corpus_fixtures import (
    BRANDS_SECTION,
    DOSAGE_SECTION,
    INTERACTIONS_SECTION,
    LEAFLET,
    LEAFLET_SECTION,
    PARACETAMOL,
    PHARMACOLOGY_SECTION,
    small_bundle,
    small_bundle_embeddings,
    with_section_text,
)
from tests.fakes import (
    FAKE_EMBEDDING_DIMENSION,
    FAKE_EMBEDDING_MODEL,
    NOW,
    FakeEmbedder,
)


def test_snapshot_maps_the_fixture_to_rows() -> None:
    bundle = small_bundle()
    snapshot = build_snapshot(bundle)

    assert snapshot.collection.id == collection_id_for("formulary")
    assert snapshot.collection.visibility is Visibility.PRIVATE
    assert snapshot.collection.current_release_id is None
    assert [document.key for document in snapshot.documents] == [PARACETAMOL, LEAFLET]
    assert len(snapshot.sections) == len(snapshot.revisions) == 5
    assert len(snapshot.chunks) == len(snapshot.release_chunks) > 5
    assert {chunk.chunker_version for chunk in snapshot.chunks} == {CHUNKER_VERSION}
    assert snapshot.chunker_version == CHUNKER_VERSION
    dosage_section_id = next(s.id for s in snapshot.sections if s.key == DOSAGE_SECTION)
    dosage_revision_id = next(
        r.id for r in snapshot.revisions if r.section_id == dosage_section_id
    )
    # P1 build_context_header: title and context_path lines joined with "\n> ".
    assert {
        chunk.context_header
        for chunk in snapshot.chunks
        if chunk.section_revision_id == dosage_revision_id
    } == {"Paracetamol\n> Liều lượng và cách dùng"}
    assert snapshot.bundle_digest == bundle_digest(bundle)

    section_keys = {section.id: section.key for section in snapshot.sections}
    strategies: dict[str, set[HydrateStrategy]] = {}
    for placement in snapshot.release_chunks:
        strategies.setdefault(section_keys[placement.section_id], set()).add(
            placement.hydrate_strategy
        )
    assert strategies[BRANDS_SECTION] == {HydrateStrategy.SEARCH_ONLY}
    assert strategies[PHARMACOLOGY_SECTION] == {HydrateStrategy.CHUNK_WINDOW}
    for key in (DOSAGE_SECTION, INTERACTIONS_SECTION, LEAFLET_SECTION):
        assert strategies[key] == {HydrateStrategy.FULL_SECTION}

    pharmacology_id = next(
        section.id
        for section in snapshot.sections
        if section.key == PHARMACOLOGY_SECTION
    )
    ordinals = [
        placement.ordinal
        for placement in snapshot.release_chunks
        if placement.section_id == pharmacology_id
    ]
    assert len(ordinals) > 1 and ordinals == sorted(ordinals) and ordinals[0] == 1

    chunks = {chunk.id: chunk for chunk in snapshot.chunks}
    kinds = {chunks[p.chunk_version_id].kind for p in snapshot.release_chunks}
    assert {BlockKind.PROSE, BlockKind.TABLE, BlockKind.INDEX_ENTRIES} <= kinds
    leaflet_id = next(s.id for s in snapshot.sections if s.key == LEAFLET_SECTION)
    leaflet_chunks = [
        chunks[p.chunk_version_id]
        for p in snapshot.release_chunks
        if p.section_id == leaflet_id
    ]
    assert all(
        chunk.colloquial is not None and chunk.colloquial.key == "panadol-extra"
        for chunk in leaflet_chunks
    )

    assert set(snapshot.embedding_texts()) == set(
        small_bundle_embeddings(FAKE_EMBEDDING_MODEL, FAKE_EMBEDDING_DIMENSION)
    )
    stats = snapshot.stats()
    assert (stats.documents, stats.sections, stats.section_revisions) == (2, 5, 5)
    assert (stats.glossary_entries, stats.colloquial_mappings) == (2, 1)
    assert stats.chunks == len(snapshot.release_chunks)
    assert stats.embeddings_computed == stats.points_upserted == 0


def test_snapshot_ids_are_deterministic_and_follow_content() -> None:
    first = build_snapshot(small_bundle())
    assert build_snapshot(small_bundle()) == first

    edited = build_snapshot(
        with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú mới.")
    )
    assert {s.id for s in edited.sections} == {s.id for s in first.sections}
    assert len({r.id for r in first.revisions} - {r.id for r in edited.revisions}) == 1
    dosage_id = next(s.id for s in first.sections if s.key == DOSAGE_SECTION)
    untouched_before = {
        p.chunk_version_id for p in first.release_chunks if p.section_id != dosage_id
    }
    untouched_after = {
        p.chunk_version_id for p in edited.release_chunks if p.section_id != dosage_id
    }
    assert untouched_before == untouched_after
    assert {p.chunk_version_id for p in first.release_chunks} != {
        p.chunk_version_id for p in edited.release_chunks
    }


def test_bundle_digest_tracks_content_not_embeddings() -> None:
    bundle = small_bundle()
    digest = bundle_digest(bundle)
    assert len(digest) == 64
    assert bundle_digest(bundle.model_copy(update={"embeddings": {}})) == digest
    assert (
        bundle_digest(with_section_text(bundle, DOSAGE_SECTION, "Ghi chú.")) != digest
    )


def test_snapshot_rejects_unknown_document_reference() -> None:
    bundle = small_bundle()
    broken = bundle.model_copy(
        update={
            "sections": [
                bundle.sections[0].model_copy(update={"document_key": "drug:missing"})
            ]
        }
    )
    with pytest.raises(CorpusImportError, match="drug:missing"):
        build_snapshot(broken)


def test_identity_helpers_are_uuid5_in_the_corpus_namespace() -> None:
    collection_id = collection_id_for("formulary")
    document_id = document_id_for(collection_id, PARACETAMOL)
    assert collection_id == uuid.uuid5(CORPUS_NAMESPACE, "collection\x1fformulary")
    assert document_id == uuid.uuid5(
        CORPUS_NAMESPACE, f"document\x1f{collection_id}\x1f{PARACETAMOL}"
    )
    assert section_id_for(document_id, DOSAGE_SECTION) == uuid.uuid5(
        CORPUS_NAMESPACE, f"section\x1f{document_id}\x1f{DOSAGE_SECTION}"
    )


def _release(number: int, status: ReleaseStatus = ReleaseStatus.READY) -> Release:
    return Release(
        id=uuid.uuid4(),
        collection_id=collection_id_for("formulary"),
        number=number,
        status=status,
        bundle_digest="0" * 64,
        chunker_version=CHUNKER_VERSION,
        embedding_model=FAKE_EMBEDDING_MODEL,
        created_at=NOW,
    )


def test_releases_to_retire_keeps_current_and_newest() -> None:
    r1, r3, r4, r5 = _release(1), _release(3), _release(4), _release(5)
    r2 = _release(2, ReleaseStatus.RETIRED)
    releases = [r1, r2, r3, r4, r5]
    assert releases_to_retire(releases, r1.id, keep=2) == [r3]
    assert releases_to_retire(releases, r1.id, keep=0) == [r5, r4, r3]
    assert releases_to_retire(releases, None, keep=10) == []
    with pytest.raises(ValueError, match="keep"):
        releases_to_retire(releases, None, keep=-1)


async def test_fake_embedder_satisfies_the_embedder_port() -> None:
    embedder: Embedder = FakeEmbedder()
    assert (embedder.model, embedder.dimension) == (FAKE_EMBEDDING_MODEL, 4)
    assert len((await embedder.embed(["x"]))[0]) == embedder.dimension
