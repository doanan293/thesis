import pytest
from pydantic import ValidationError

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    BundleValidationError,
    DocumentKind,
    DocumentRecord,
    RetrievalMode,
    SectionRecord,
    decode_vector,
    encode_vector,
    model_slug,
)
from pharma_agent.domain.shared.errors import DomainError

SECTION_LINE = (
    '{"key": "drug:paracetamol:tuong-tac-thuoc", "document_key": "drug:paracetamol",'
    ' "heading": "Tương tác thuốc", "context_path": ["Tương tác thuốc"],'
    ' "ordinal": 2, "start_page": 1204, "end_page": 1205, "retrieval": "default",'
    ' "blocks": [{"kind": "table",'
    ' "markdown": "| Thuốc | Hậu quả |\\n| --- | --- |\\n| Warfarin | Tăng INR |",'
    ' "start_page": 1205, "end_page": 1205, "table_key": "tbl-0042"}]}'
)


def test_section_line_parses_enum_strings_from_json() -> None:
    section = SectionRecord.model_validate_json(SECTION_LINE)

    assert section.retrieval is RetrievalMode.DEFAULT
    assert section.blocks[0].kind is BlockKind.TABLE
    assert section.blocks[0].table_key == "tbl-0042"
    assert section.blocks[0].markdown.splitlines()[2] == "| Warfarin | Tăng INR |"


def test_document_line_uses_defaults() -> None:
    document = DocumentRecord.model_validate_json(
        '{"key": "drug:paracetamol", "kind": "drug_monograph", "title": "PARACETAMOL",'
        ' "source": {"title": "Dược thư Quốc gia Việt Nam 2022"}}'
    )

    assert document.kind is DocumentKind.DRUG_MONOGRAPH
    assert document.source.url is None
    assert document.attributes == {}


def test_bundle_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="chunk_id"):
        BlockRecord.model_validate_json(
            '{"kind": "prose", "markdown": "Hạ sốt", "chunk_id": "x"}'
        )


def test_bundle_models_reject_unknown_enum_values_and_loose_types() -> None:
    with pytest.raises(ValidationError, match="kind"):
        DocumentRecord.model_validate_json(
            '{"key": "drug:x", "kind": "brand_page", "title": "X",'
            ' "source": {"title": "Dược thư"}}'
        )
    with pytest.raises(ValidationError, match="start_page"):
        BlockRecord.model_validate_json(
            '{"kind": "prose", "markdown": "Hạ sốt", "start_page": "12"}'
        )


@pytest.mark.parametrize(
    ("model", "slug"),
    [
        ("qwen3-embedding:4b-fp16", "qwen3_embedding_4b_fp16"),
        ("Qwen/Qwen3-Embedding-4B", "qwen_qwen3_embedding_4b"),
        ("fake-embedding-4d", "fake_embedding_4d"),
        ("--", ""),
    ],
)
def test_model_slug(model: str, slug: str) -> None:
    assert model_slug(model) == slug


def test_vector_round_trip_is_float32_little_endian_base64() -> None:
    values = [0.5, -1.25, 3.0, 0.0]

    assert encode_vector([1.0]) == "AACAPw=="
    assert decode_vector(encode_vector(values), 4) == values


def test_decode_vector_rejects_wrong_length_and_bad_base64() -> None:
    with pytest.raises(ValueError, match="expected 16"):
        decode_vector(encode_vector([0.5, 0.5, 0.5]), 4)
    with pytest.raises(ValueError, match="base64"):
        decode_vector("@@@@", 1)


def test_bundle_validation_error_carries_every_problem() -> None:
    problems = [
        "sections.jsonl:3: ordinal: duplicate ordinal 2 in document 'drug:paracetamol'",
        "glossary.json: [1].term: must not be empty",
    ]

    error = BundleValidationError(problems)

    assert isinstance(error, DomainError)
    assert error.code == "BUNDLE_INVALID"
    assert error.problems == problems
    assert "sections.jsonl:3: ordinal" in str(error)
