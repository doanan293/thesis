import hashlib
import unicodedata
import uuid

from pharma_agent.domain.corpus.bundle import BlockKind, BlockRecord
from pharma_agent.domain.corpus.identity import (
    CORPUS_NAMESPACE,
    canonical_json,
    chunk_version_id,
    section_revision_id,
    sha256_hex,
)

SECTION_KEY = "drug:paracetamol:lieu-luong-va-cach-dung"
BLOCKS = [
    BlockRecord(
        kind=BlockKind.PROSE,
        markdown="Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ, tối đa 4 g/ngày.",
        start_page=1203,
        end_page=1203,
    ),
    BlockRecord(
        kind=BlockKind.TABLE,
        markdown="| Tuổi | Liều |\n| --- | --- |\n| Trẻ em | 10 - 15 mg/kg |",
        start_page=1204,
        end_page=1204,
        table_key="tbl-0007",
    ),
]
CHUNK_TEXT = "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ, tối đa 4 g/ngày."
EMBEDDING_TEXT = f"PARACETAMOL\n> Liều lượng và cách dùng\n\n{CHUNK_TEXT}"


def test_namespace_is_pinned() -> None:
    assert uuid.uuid5(uuid.NAMESPACE_URL, "pharma-agent:corpus:v1") == CORPUS_NAMESPACE


def test_canonical_json_sorts_keys_compacts_and_normalizes_strings() -> None:
    value = {
        "b": [unicodedata.normalize("NFD", "Liều") + "  \r\n", 2.5],
        "a": {"z": 1, "y": None},
    }

    assert canonical_json(value) == '{"a":{"y":null,"z":1},"b":["Liều",2.5]}'


def test_sha256_hex_hashes_utf8_text() -> None:
    assert sha256_hex("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_hex("Liều") == hashlib.sha256("Liều".encode()).hexdigest()


def test_section_revision_id_follows_the_spec_formula() -> None:
    blocks_json = canonical_json([block.model_dump(mode="json") for block in BLOCKS])
    expected = uuid.uuid5(
        CORPUS_NAMESPACE,
        "section-revision\x1f" + SECTION_KEY + "\x1f" + sha256_hex(blocks_json),
    )

    assert section_revision_id(SECTION_KEY, BLOCKS) == expected


def test_section_revision_id_is_deterministic_and_content_sensitive() -> None:
    base = section_revision_id(SECTION_KEY, BLOCKS)
    with_crlf = [
        BLOCKS[0].model_copy(update={"markdown": BLOCKS[0].markdown + "  \r\n"}),
        BLOCKS[1],
    ]
    with_other_page = [BLOCKS[0].model_copy(update={"start_page": 1202}), BLOCKS[1]]

    assert section_revision_id(SECTION_KEY, [b.model_copy() for b in BLOCKS]) == base
    assert section_revision_id(SECTION_KEY, with_crlf) == base
    assert section_revision_id(SECTION_KEY, with_other_page) != base
    assert section_revision_id(SECTION_KEY, list(reversed(BLOCKS))) != base
    assert section_revision_id("drug:paracetamol:chong-chi-dinh", BLOCKS) != base


def test_chunk_version_id_follows_the_spec_formula() -> None:
    digest = sha256_hex(
        "\x1f".join((SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1"))
    )

    assert chunk_version_id(
        SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1"
    ) == uuid.uuid5(CORPUS_NAMESPACE, "chunk\x1f" + digest)


def test_chunk_version_id_changes_with_every_input_but_not_with_whitespace() -> None:
    base = chunk_version_id(SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1")
    enriched = (
        f"{EMBEDDING_TEXT}\n\nThuật ngữ: G6PD = glucose-6-phosphate dehydrogenase"
    )

    assert (
        chunk_version_id(SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v2") != base
    )
    assert chunk_version_id(SECTION_KEY, CHUNK_TEXT, enriched, "chunker-v1") != base
    assert (
        chunk_version_id(
            SECTION_KEY, f"{CHUNK_TEXT} Trẻ em", EMBEDDING_TEXT, "chunker-v1"
        )
        != base
    )
    assert (
        chunk_version_id(
            "drug:ibuprofen:lieu-dung", CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1"
        )
        != base
    )
    assert (
        chunk_version_id(
            SECTION_KEY,
            f"{CHUNK_TEXT}  \n",
            EMBEDDING_TEXT.replace("\n", "\r\n"),
            "chunker-v1",
        )
        == base
    )
