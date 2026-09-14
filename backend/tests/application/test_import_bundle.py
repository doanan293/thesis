import asyncio
from collections.abc import Sequence
from functools import partial

import pytest

from pharma_agent.application.corpus.import_bundle import ImportOutcome
from pharma_agent.application.corpus.indexing import EmbeddingCounts, EmbeddingResolver
from pharma_agent.domain.corpus.bundle import read_bundle_embeddings
from pharma_agent.domain.corpus.models import (
    CorpusImportError,
    ReleaseStatus,
    build_snapshot,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from tests.corpus_fixtures import (
    DOSAGE_SECTION,
    FIXTURE_DIR,
    no_bundle_embeddings,
    small_bundle,
    small_bundle_embeddings,
    with_section_text,
)
from tests.corpus_memory import (
    CorpusAdapters,
    InMemoryEmbeddingCache,
    InMemoryVectorIndex,
    build_importer,
)
from tests.fakes import (
    FAKE_EMBEDDING_DIMENSION,
    FAKE_EMBEDDING_MODEL,
    NOW,
    FakeEmbedder,
    fake_vector,
)


def unchanged_bundle_embeddings(model: str, dims: int) -> dict[str, list[float]]:
    raise AssertionError("an unchanged bundle must not load its vectors")


async def test_import_takes_vectors_from_cache_then_bundle_then_embedder() -> None:
    adapters = CorpusAdapters()
    bundle = small_bundle()
    snapshot = build_snapshot(bundle)
    texts = snapshot.embedding_texts()
    hashes = sorted(texts)
    cached, bundled = hashes[0], hashes[1:3]
    await adapters.cache.put_many(
        FAKE_EMBEDDING_MODEL, 4, {cached: fake_vector(texts[cached])}
    )
    shipped = small_bundle_embeddings(FAKE_EMBEDDING_MODEL, FAKE_EMBEDDING_DIMENSION)
    embedder = FakeEmbedder()

    report = await build_importer(adapters, embedder, batch_size=2)(
        bundle,
        embeddings=lambda _model, _dims: {sha: shipped[sha] for sha in bundled},
        publish=False,
    )

    assert (report.outcome, report.published, report.collection_key) == (
        ImportOutcome.IMPORTED,
        False,
        "formulary",
    )
    release = report.release
    assert (release.number, release.status, release.ready_at, release.published_at) == (
        1,
        ReleaseStatus.READY,
        NOW,
        None,
    )
    assert release.stats is not None
    assert (
        release.stats.embeddings_cached,
        release.stats.embeddings_from_bundle,
        release.stats.embeddings_computed,
    ) == (1, 2, len(hashes) - 3)
    assert (release.stats.points_upserted, release.stats.points_updated) == (
        len(snapshot.release_chunks),
        0,
    )
    assert release.stats.chunks == len(snapshot.release_chunks)
    assert sorted(text for batch in embedder.batches for text in batch) == sorted(
        texts[sha] for sha in hashes[3:]
    )
    assert all(len(batch) <= 2 for batch in embedder.batches)
    assert set(adapters.cache.vectors) == {
        (FAKE_EMBEDDING_MODEL, sha) for sha in hashes
    }
    assert len(adapters.index.points) == len(snapshot.release_chunks)
    assert all(
        item.release_ids == [release.id] for item, _ in adapters.index.points.values()
    )
    collection = await adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id is None


async def test_publish_then_same_bundle_is_no_change() -> None:
    adapters = CorpusAdapters()
    embedder = FakeEmbedder()
    first = await build_importer(adapters, embedder)(
        small_bundle(), embeddings=small_bundle_embeddings, publish=True
    )
    assert first.outcome is ImportOutcome.IMPORTED and first.published
    assert first.release.published_at == NOW
    assert embedder.batches == []  # the fixture ships every vector

    again_embedder = FakeEmbedder()
    again = await build_importer(adapters, again_embedder)(
        small_bundle(), embeddings=unchanged_bundle_embeddings, publish=True
    )

    assert again.outcome is ImportOutcome.NO_CHANGE
    assert again.release == first.release
    assert adapters.repository.stage_calls == 1
    assert adapters.index.ensure_calls == 1
    assert again_embedder.batches == []
    collection = await adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == first.release.id


async def test_ready_unpublished_release_is_reused() -> None:
    adapters = CorpusAdapters()
    first = await build_importer(adapters, FakeEmbedder())(
        small_bundle(), embeddings=small_bundle_embeddings, publish=False
    )
    second = await build_importer(adapters, FakeEmbedder())(
        small_bundle(), embeddings=small_bundle_embeddings, publish=True
    )

    assert second.outcome is ImportOutcome.REUSED
    assert second.release.id == first.release.id and second.published
    assert second.release.published_at == NOW
    assert adapters.repository.stage_calls == 1
    assert len(await adapters.repository.list_releases("formulary")) == 1


async def test_failed_embedding_leaves_building_release_and_rerun_resumes() -> None:
    adapters = CorpusAdapters()
    bundle = small_bundle()
    texts = build_snapshot(bundle).embedding_texts()

    with pytest.raises(RetrievalError):
        await build_importer(adapters, FakeEmbedder(fail_on_batch=2), batch_size=2)(
            bundle, embeddings=no_bundle_embeddings, publish=True
        )

    (summary,) = await adapters.repository.list_releases("formulary")
    assert summary.release.status is ReleaseStatus.BUILDING and not summary.current
    assert len(adapters.cache.vectors) == 2

    retry = FakeEmbedder()
    report = await build_importer(adapters, retry, batch_size=2)(
        bundle, embeddings=no_bundle_embeddings, publish=True
    )

    assert report.outcome is ImportOutcome.IMPORTED
    assert report.release.id == summary.release.id
    assert report.release.status is ReleaseStatus.READY and report.published
    assert sum(len(batch) for batch in retry.batches) == len(texts) - 2
    assert report.release.stats is not None
    assert report.release.stats.embeddings_cached == 2
    assert len(await adapters.repository.list_releases("formulary")) == 1


async def test_changed_bundle_adds_release_and_rewrites_release_ids() -> None:
    adapters = CorpusAdapters()
    original = small_bundle()
    first = await build_importer(adapters, FakeEmbedder())(
        original, embeddings=small_bundle_embeddings, publish=True
    )
    edited = with_section_text(original, DOSAGE_SECTION, "Ghi chú liều mới.")
    embedder = FakeEmbedder()

    second = await build_importer(adapters, embedder)(
        edited, embeddings=small_bundle_embeddings, publish=True
    )

    old = build_snapshot(original)
    new = build_snapshot(edited)
    old_ids, new_ids = {c.id for c in old.chunks}, {c.id for c in new.chunks}
    assert second.outcome is ImportOutcome.IMPORTED and second.release.number == 2
    assert second.release.stats is not None
    assert second.release.stats.points_upserted == len(new_ids - old_ids)
    assert second.release.stats.points_updated == len(new_ids & old_ids)
    ids = sorted([first.release.id, second.release.id])
    for chunk_id in new_ids & old_ids:
        assert adapters.index.release_ids_of(chunk_id) == ids
    for chunk_id in new_ids - old_ids:
        assert adapters.index.release_ids_of(chunk_id) == [second.release.id]
    for chunk_id in old_ids - new_ids:
        assert adapters.index.release_ids_of(chunk_id) == [first.release.id]
    new_hashes = set(new.embedding_texts()) - set(old.embedding_texts())
    assert sum(len(batch) for batch in embedder.batches) == len(new_hashes)
    collection = await adapters.repository.get_collection("formulary")
    assert collection is not None and collection.current_release_id == second.release.id


async def test_bundle_vectors_declared_with_other_dims_are_ignored() -> None:
    adapters = CorpusAdapters()
    bundle = small_bundle()
    manifest = bundle.manifest.model_copy(
        update={
            "embeddings": [
                entry.model_copy(update={"dims": 8})
                for entry in bundle.manifest.embeddings
            ]
        }
    )
    report = await build_importer(adapters, FakeEmbedder())(
        bundle,
        embeddings=partial(read_bundle_embeddings, FIXTURE_DIR, manifest),
        publish=False,
    )
    assert report.release.stats is not None
    assert report.release.stats.embeddings_from_bundle == 0
    assert report.release.stats.embeddings_computed == len(
        build_snapshot(bundle).embedding_texts()
    )


async def test_wrong_vector_length_is_rejected_before_any_write() -> None:
    adapters = CorpusAdapters()
    vectors = small_bundle_embeddings(FAKE_EMBEDDING_MODEL, FAKE_EMBEDDING_DIMENSION)
    vectors[next(iter(vectors))] = [0.5]

    with pytest.raises(CorpusImportError, match="dims"):
        await build_importer(adapters, FakeEmbedder())(
            small_bundle(), embeddings=lambda _model, _dims: vectors, publish=True
        )

    assert adapters.repository.stage_calls == 0
    assert adapters.index.ensure_calls == 0


async def test_missing_points_keep_the_release_building() -> None:
    snapshot = build_snapshot(small_bundle())
    adapters = CorpusAdapters(
        index=InMemoryVectorIndex(drop_ids={snapshot.chunks[0].id})
    )

    with pytest.raises(CorpusImportError, match="expected"):
        await build_importer(adapters, FakeEmbedder())(
            small_bundle(), embeddings=small_bundle_embeddings, publish=True
        )

    (summary,) = await adapters.repository.list_releases("formulary")
    assert summary.release.status is ReleaseStatus.BUILDING and not summary.current


class TrackingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.peak = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return await super().embed(texts)


async def test_embedding_resolver_respects_the_concurrency_limit() -> None:
    embedder = TrackingEmbedder()
    resolver = EmbeddingResolver(
        InMemoryEmbeddingCache(), embedder, batch_size=1, max_concurrent=2
    )
    counts = await resolver.ensure({f"h{i}": f"text {i}" for i in range(6)}, {})
    assert counts == EmbeddingCounts(cached=0, from_bundle=0, computed=6)
    assert embedder.peak == 2 and len(embedder.batches) == 6
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingResolver(
            InMemoryEmbeddingCache(), embedder, batch_size=0, max_concurrent=1
        )
