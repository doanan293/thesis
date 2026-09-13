import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import TypeAdapter

from pharma_agent.domain.corpus import bundle, chunking, enrichment, hydrate, identity
from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    ColloquialMappingRecord,
    DocumentRecord,
    GlossaryEntry,
    SectionRecord,
)
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION, chunk_section
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for
from pharma_agent.domain.corpus.identity import chunk_version_id, sha256_hex

GOLDEN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "golden" / "chunking_v1.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = GOLDEN["cases"]
GLOSSARY = TypeAdapter(list[GlossaryEntry]).validate_json(
    json.dumps(GOLDEN["glossary"])
)
MAPPINGS = TypeAdapter(list[ColloquialMappingRecord]).validate_json(
    json.dumps(GOLDEN["colloquial_mappings"])
)
PUBLIC_API: dict[ModuleType, list[str]] = {
    bundle: [
        "BUNDLE_SCHEMA_VERSION",
        "KnowledgeBundle",
        "BundleValidationError",
        "read_bundle",
        "write_bundle",
        "model_slug",
        "encode_vector",
        "decode_vector",
    ],
    identity: [
        "CORPUS_NAMESPACE",
        "canonical_json",
        "sha256_hex",
        "section_revision_id",
        "chunk_version_id",
    ],
    chunking: ["CHUNKER_VERSION", "MAX_CHUNK_CHARS", "ChunkDraft", "chunk_section"],
    enrichment: [
        "build_context_header",
        "detect_terms",
        "mapping_for_section",
        "compose_embedding_text",
    ],
    hydrate: ["FULL_SECTION_MAX_CHARS", "section_char_count", "hydrate_strategy_for"],
}


def load_case(name: str) -> tuple[DocumentRecord, SectionRecord, int]:
    case = next(case for case in CASES if case["name"] == name)
    return (
        DocumentRecord.model_validate_json(json.dumps(case["document"])),
        SectionRecord.model_validate_json(json.dumps(case["section"])),
        case["max_chars"],
    )


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_chunk_section_matches_the_old_pipeline(case: dict[str, Any]) -> None:
    document, section, max_chars = load_case(case["name"])

    drafts = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)

    assert [
        {
            "ordinal": draft.ordinal,
            "kind": draft.kind.value,
            "chunk_text": draft.chunk_text,
            "start_page": draft.start_page,
            "end_page": draft.end_page,
            "table_key": draft.table_key,
            "context_header": draft.context_header,
            "embedding_text": draft.embedding_text,
            "term_annotations": [a.model_dump() for a in draft.term_annotations],
            "colloquial": draft.colloquial.model_dump(exclude_defaults=True)
            if draft.colloquial is not None
            else None,
        }
        for draft in drafts
    ] == case["expected"]["chunks"]
    assert hydrate_strategy_for(section).value == case["expected"]["hydrate_strategy"]


def test_chunk_drafts_carry_their_identities() -> None:
    for case in CASES:
        document, section, max_chars = load_case(case["name"])
        for draft in chunk_section(
            document, section, GLOSSARY, MAPPINGS, max_chars=max_chars
        ):
            assert draft.section_key == section.key
            assert draft.embedding_text_sha256 == sha256_hex(draft.embedding_text)
            assert draft.chunk_version_id == chunk_version_id(
                section.key, draft.chunk_text, draft.embedding_text, CHUNKER_VERSION
            )


def test_chunk_section_is_deterministic() -> None:
    document, section, _ = load_case("table")

    first = chunk_section(document, section, GLOSSARY, MAPPINGS)
    second = chunk_section(
        document.model_copy(deep=True),
        section.model_copy(deep=True),
        list(GLOSSARY),
        list(MAPPINGS),
    )

    assert first == second
    assert len({draft.chunk_version_id for draft in first}) == len(first) == 2


def test_glossary_change_creates_new_versions_only_for_affected_chunks() -> None:
    document, section, max_chars = load_case("prose")
    without_adr = [entry for entry in GLOSSARY if entry.term != "ADR"]

    before = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    after = chunk_section(document, section, without_adr, MAPPINGS, max_chars=max_chars)

    assert [draft.chunk_text for draft in after] == [d.chunk_text for d in before]
    assert [
        old.ordinal
        for old, new in zip(before, after, strict=True)
        if old.chunk_version_id != new.chunk_version_id
    ] == [2]
    assert "ADR = " not in after[1].embedding_text


def test_colloquial_mapping_change_creates_new_versions() -> None:
    document, section, max_chars = load_case("leaflet")
    edited = [MAPPINGS[0].model_copy(update={"visual_sign": "Vỉ thuốc màu đỏ"})]

    before = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    after = chunk_section(document, section, GLOSSARY, edited, max_chars=max_chars)

    assert all(
        old.chunk_version_id != new.chunk_version_id
        for old, new in zip(before, after, strict=True)
    )


def test_colloquial_key_changes_nothing_but_the_colloquial_key() -> None:
    document, section, max_chars = load_case("leaflet")
    unkeyed = [MAPPINGS[0].model_copy(update={"key": ""})]

    keyed = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    without_key = chunk_section(
        document, section, GLOSSARY, unkeyed, max_chars=max_chars
    )

    assert [d.colloquial.key if d.colloquial else None for d in keyed] == [
        "panadol-extra-gsk-150-vien-11440",
        "panadol-extra-gsk-150-vien-11440",
    ]
    assert [d.colloquial.key if d.colloquial else None for d in without_key] == [
        "",
        "",
    ]
    assert [d.model_dump(exclude={"colloquial": {"key"}}) for d in without_key] == [
        d.model_dump(exclude={"colloquial": {"key"}}) for d in keyed
    ]


def test_chunker_version_change_creates_new_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, section, max_chars = load_case("list")

    before = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    monkeypatch.setattr(chunking, "CHUNKER_VERSION", "chunker-v2")
    after = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)

    assert [draft.chunk_text for draft in after] == [d.chunk_text for d in before]
    assert not {d.chunk_version_id for d in before} & {
        d.chunk_version_id for d in after
    }


def test_a_section_whose_blocks_are_all_blank_has_no_chunks() -> None:
    document, section, _ = load_case("prose")
    blank = section.model_copy(
        update={
            "blocks": [
                BlockRecord(kind=BlockKind.PROSE, markdown=" \n "),
                BlockRecord(kind=BlockKind.TABLE, markdown=""),
            ]
        }
    )

    assert chunk_section(document, blank, GLOSSARY, MAPPINGS) == []


def test_context_header_falls_back_like_the_old_metadata_builder() -> None:
    document, section, _ = load_case("prose")
    no_path = section.model_copy(update={"context_path": []})
    label_title = document.model_copy(update={"title": "Tên gọi khác: Panadol đỏ"})
    untitled = document.model_copy(update={"title": ""})

    assert chunk_section(document, no_path, [], [])[0].context_header == "PARACETAMOL"
    assert (
        chunk_section(label_title, no_path, [], [])[0].context_header
        == "Tên gọi khác: Panadol đỏ > Liều lượng và cách dùng"
    )
    assert (
        chunk_section(untitled, no_path, [], [])[0].context_header
        == "Liều lượng và cách dùng"
    )


def test_table_key_is_kept_only_for_table_blocks() -> None:
    document, section, _ = load_case("table")
    prose, table = section.blocks
    tagged = section.model_copy(
        update={"blocks": [prose.model_copy(update={"table_key": "tbl-9999"}), table]}
    )

    drafts = chunk_section(document, tagged, GLOSSARY, MAPPINGS)

    assert [draft.table_key for draft in drafts] == [None, "tbl-0042"]


@pytest.mark.parametrize(
    "module", list(PUBLIC_API), ids=[module.__name__ for module in PUBLIC_API]
)
def test_public_api_for_seed_pipeline_is_documented(module: ModuleType) -> None:
    assert "Public API" in (module.__doc__ or "")
    for name in PUBLIC_API[module]:
        assert hasattr(module, name), f"{module.__name__}.{name}"
