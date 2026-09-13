import struct
from pathlib import Path

import pytest

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    DocumentKind,
    RetrievalMode,
    read_bundle,
    write_bundle,
)
from pharma_agent.domain.corpus.hydrate import (
    FULL_SECTION_MAX_CHARS,
    section_char_count,
)
from pharma_agent.domain.retrieval.ports import RetrievalError
from tests.corpus_fixtures import (
    BRANDS_SECTION,
    COLLECTION_KEY,
    DOSAGE_SECTION,
    FIXTURE_DIR,
    INTERACTIONS_SECTION,
    LEAFLET_SECTION,
    PHARMACOLOGY_SECTION,
    build_small_bundle,
    small_bundle,
    with_section_text,
)
from tests.fakes import (
    FAKE_EMBEDDING_DIMENSION,
    FAKE_EMBEDDING_MODEL,
    FakeEmbedder,
    fake_vector,
)


def _files(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


def test_committed_fixture_matches_the_generator(tmp_path: Path) -> None:
    target = tmp_path / "bundle"
    target.mkdir()
    write_bundle(build_small_bundle(), target)
    assert read_bundle(target) == small_bundle(), (
        "stale fixture: run `uv run python -m tests.corpus_fixtures` and commit it"
    )
    assert _files(target) == _files(FIXTURE_DIR)


def test_fixture_covers_every_case_the_corpus_tests_need() -> None:
    bundle = small_bundle()
    sections = {section.key: section for section in bundle.sections}
    assert bundle.manifest.collection.key == COLLECTION_KEY
    assert [document.kind for document in bundle.documents] == [
        DocumentKind.DRUG_MONOGRAPH,
        DocumentKind.LEAFLET,
    ]
    assert {block.kind for block in sections[DOSAGE_SECTION].blocks} == {
        BlockKind.PROSE
    }
    assert section_char_count(sections[PHARMACOLOGY_SECTION]) > FULL_SECTION_MAX_CHARS
    assert sections[INTERACTIONS_SECTION].blocks[0].kind is BlockKind.TABLE
    assert sections[BRANDS_SECTION].retrieval is RetrievalMode.INDEX_ONLY
    assert sections[BRANDS_SECTION].blocks[0].kind is BlockKind.INDEX_ENTRIES
    assert [mapping.section_keys for mapping in bundle.colloquial_mappings] == [
        [LEAFLET_SECTION]
    ]
    assert [entry.term for entry in bundle.glossary] == ["NSAID", "G6PD"]
    (embedding_file,) = bundle.manifest.embeddings
    assert (embedding_file.model, embedding_file.dims) == (
        FAKE_EMBEDDING_MODEL,
        FAKE_EMBEDDING_DIMENSION,
    )
    assert embedding_file.file == "embeddings/fake_embedding_4d.jsonl"
    assert len(bundle.embeddings[FAKE_EMBEDDING_MODEL]) >= len(bundle.sections)


def test_with_section_text_changes_only_that_section() -> None:
    bundle = small_bundle()
    edited = with_section_text(bundle, DOSAGE_SECTION, "Ghi chú mới.")
    before = {section.key: section for section in bundle.sections}
    after = {section.key: section for section in edited.sections}
    assert after[DOSAGE_SECTION].blocks[0].markdown.endswith("\n\nGhi chú mới.")
    assert all(after[key] == before[key] for key in before if key != DOSAGE_SECTION)
    with pytest.raises(KeyError):
        with_section_text(bundle, "drug:missing:section", "x")


async def test_fake_embedder_is_deterministic_and_float32_exact() -> None:
    embedder = FakeEmbedder()
    first = await embedder.embed(["a", "b"])
    second = await embedder.embed(["a"])
    assert first[0] == second[0] == fake_vector("a")
    assert len(first[1]) == FAKE_EMBEDDING_DIMENSION
    packed = struct.pack("<4f", *first[1])
    assert list(struct.unpack("<4f", packed)) == first[1]
    assert embedder.batches == [["a", "b"], ["a"]]
    assert (embedder.model, embedder.dimension) == ("fake-embedding-4d", 4)


async def test_fake_embedder_fails_on_the_requested_batch() -> None:
    embedder = FakeEmbedder(fail_on_batch=2)
    await embedder.embed(["a"])
    with pytest.raises(RetrievalError):
        await embedder.embed(["b"])
    assert embedder.batches == [["a"], ["b"]]
