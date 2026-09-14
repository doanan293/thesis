import uuid
from dataclasses import dataclass

import pytest

from pharma_agent.application.corpus.import_bundle import ImportKnowledgeBundle
from pharma_agent.application.corpus.releases import ReleaseService
from pharma_agent.domain.corpus.bundle import KnowledgeBundle
from pharma_agent.domain.corpus.models import (
    CollectionNotFound,
    NoEarlierRelease,
    Release,
    ReleaseNotFound,
    ReleaseNotPublishable,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    LEAFLET_SECTION,
    small_bundle,
    with_renamed_documents,
    with_section_text,
)
from tests.corpus_memory import (
    CorpusAdapters,
    SteppingClock,
    build_importer,
    build_release_service,
)
from tests.fakes import FakeEmbedder


@dataclass
class Services:
    adapters: CorpusAdapters
    embedder: FakeEmbedder
    clock: SteppingClock
    importer: ImportKnowledgeBundle
    releases: ReleaseService

    async def publish_bundle(
        self, bundle: KnowledgeBundle, *, publish: bool = True
    ) -> Release:
        return (await self.importer(bundle, publish=publish)).release


def services() -> Services:
    adapters, embedder, clock = CorpusAdapters(), FakeEmbedder(), SteppingClock()
    return Services(
        adapters=adapters,
        embedder=embedder,
        clock=clock,
        importer=build_importer(adapters, embedder, clock=clock),
        releases=build_release_service(adapters, embedder, clock=clock),
    )


def v2() -> KnowledgeBundle:
    return with_section_text(small_bundle(), DOSAGE_SECTION, "Ghi chú liều v2.")


def v3() -> KnowledgeBundle:
    return with_section_text(v2(), LEAFLET_SECTION, "Ghi chú tờ HDSD v3.")


def chunk_ids(bundle: KnowledgeBundle) -> set[uuid.UUID]:
    return {chunk.id for chunk in build_snapshot(bundle).chunks}


async def test_list_and_publish_follow_release_status() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2(), publish=False)

    summaries = await s.releases.list_releases("formulary")
    assert [(x.release.number, x.current) for x in summaries] == [(2, False), (1, True)]

    published = await s.releases.publish(r2.id)
    assert published.id == r2.id and published.published_at is not None
    collection = await s.adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r2.id

    with pytest.raises(RetrievalError):
        await build_importer(s.adapters, FakeEmbedder(fail_on_batch=1), clock=s.clock)(
            v3(), publish=False
        )
    building = (await s.releases.list_releases("formulary"))[0].release
    assert building.status is ReleaseStatus.BUILDING
    with pytest.raises(ReleaseNotPublishable, match="building"):
        await s.releases.publish(building.id)
    with pytest.raises(ReleaseNotFound):
        await s.releases.publish(uuid.uuid4())
    assert len(await s.releases.list_releases(None)) == 3
    assert r1.number == 1


async def test_rollback_points_back_to_the_previously_published_release() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2())

    rolled = await s.releases.rollback("formulary")
    assert rolled.id == r1.id
    collection = await s.adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == r1.id
    with pytest.raises(NoEarlierRelease):
        await s.releases.rollback("formulary")

    await s.releases.publish(r2.id)
    assert (await s.releases.rollback("formulary")).id == r1.id
    with pytest.raises(CollectionNotFound):
        await s.releases.rollback("missing")


async def test_gc_retires_old_releases_cleans_points_and_keeps_protected_chunks() -> (
    None
):
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2())
    r3 = await s.publish_bundle(v3())
    ids1, ids2, ids3 = chunk_ids(small_bundle()), chunk_ids(v2()), chunk_ids(v3())
    gone = (ids1 | ids2) - ids3
    assert len(gone) >= 2
    protected = sorted(ids1 - ids2 - ids3)[0]
    s.adapters.repository.protected_chunk_ids.add(protected)

    report = await s.releases.gc("formulary", keep=1)

    assert report.retired == [r2.id, r1.id]
    assert set(s.adapters.index.points) == ids3
    assert all(
        item.release_ids == [r3.id] for item, _ in s.adapters.index.points.values()
    )
    assert report.points_deleted == len(gone)
    assert report.points_updated == len((ids1 | ids2) & ids3)
    assert report.purge.chunk_versions_kept == 1
    assert report.purge.chunk_versions_deleted == len(gone) - 1
    assert protected in s.adapters.repository.chunks
    statuses = {
        x.release.id: (x.release.status, x.chunk_count, x.current)
        for x in await s.releases.list_releases("formulary")
    }
    assert statuses[r3.id][0] is ReleaseStatus.READY and statuses[r3.id][2]
    assert statuses[r1.id][:2] == (ReleaseStatus.RETIRED, 0)
    assert statuses[r2.id][:2] == (ReleaseStatus.RETIRED, 0)

    again = await s.releases.gc("formulary", keep=1)
    assert (again.retired, again.points_updated, again.points_deleted) == ([], 0, 0)
    assert again.purge.chunk_versions_kept == 1


async def test_reindex_rebuilds_points_from_repository_and_cache() -> None:
    s = services()
    r1 = await s.publish_bundle(small_bundle())
    r2 = await s.publish_bundle(v2())
    ids1, ids2 = chunk_ids(small_bundle()), chunk_ids(v2())
    snapshot1, snapshot2 = build_snapshot(small_bundle()), build_snapshot(v2())

    foreign = r1.model_copy(
        update={"id": uuid.uuid4(), "number": 9, "embedding_model": "other-model"}
    )
    s.adapters.repository.releases[foreign.id] = foreign
    s.adapters.repository.placements[foreign.id] = list(snapshot1.release_chunks)

    s.adapters.index.points.clear()
    dropped = next(iter(snapshot2.embedding_texts()))
    del s.adapters.cache.vectors[(s.embedder.model, dropped)]
    s.embedder.batches.clear()

    report = await s.releases.reindex("formulary")

    assert report.points_upserted == len(ids1 | ids2)
    assert report.release_points == {
        r1.id: len(snapshot1.release_chunks),
        r2.id: len(snapshot2.release_chunks),
    }
    assert report.skipped == [foreign.id]
    shared = next(iter(ids1 & ids2))
    assert s.adapters.index.release_ids_of(shared) == sorted([r1.id, r2.id])
    assert s.embedder.batches == [[snapshot2.embedding_texts()[dropped]]]


async def test_gc_deletes_documents_and_sections_no_release_uses() -> None:
    s = services()
    old = await s.publish_bundle(small_bundle())
    renamed = with_renamed_documents(small_bundle(), "-v2")
    await s.publish_bundle(renamed)

    report = await s.releases.gc("formulary", keep=1)

    repository = s.adapters.repository
    assert report.retired == [old.id]
    assert {document.key for document in repository.documents.values()} == {
        document.key for document in renamed.documents
    }
    assert {section.key for section in repository.sections.values()} == {
        section.key for section in renamed.sections
    }
    assert report.purge.documents_deleted == len(small_bundle().documents)
    assert report.purge.sections_deleted == len(small_bundle().sections)
