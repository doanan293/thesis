import hashlib
import json
from pathlib import Path

import pytest

from pharma_agent.domain.corpus.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BlockKind,
    BlockRecord,
    BundleCollection,
    BundleEmbeddingFile,
    BundleFile,
    BundleGenerator,
    BundleManifest,
    BundleValidationError,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    KnowledgeBundle,
    SectionRecord,
    SourceInfo,
    encode_vector,
    read_bundle,
    read_bundle_embeddings,
    write_bundle,
    write_bundle_embeddings,
)

MODEL = "fake-embedding-4d"
EMBEDDINGS_FILE = "embeddings/fake_embedding_4d.jsonl"
SHA_DOSE = hashlib.sha256("PARACETAMOL\n> Liều lượng".encode()).hexdigest()
SHA_TABLE = hashlib.sha256("PARACETAMOL\n> Tương tác thuốc".encode()).hexdigest()
VECTORS = {SHA_DOSE: [0.5, -1.25, 3.0, 0.0], SHA_TABLE: [1.0, 0.0, 0.0, 0.0]}
LEAFLET_SECTION = "leaflet:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440"


def make_bundle() -> KnowledgeBundle:
    paracetamol = DocumentRecord(
        key="drug:paracetamol",
        kind=DocumentKind.DRUG_MONOGRAPH,
        title="PARACETAMOL",
        source=SourceInfo(title="Dược thư Quốc gia Việt Nam 2022"),
    )
    leaflet = DocumentRecord(
        key="leaflet:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440",
        kind=DocumentKind.LEAFLET,
        title="Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)",
        source=SourceInfo(title="Tờ hướng dẫn sử dụng", url=None),
        attributes={"category": "thuoc-giam-dau-ha-sot"},
    )
    return KnowledgeBundle(
        manifest=BundleManifest(
            schema_version=BUNDLE_SCHEMA_VERSION,
            collection=BundleCollection(key="formulary", title="Dược thư Quốc gia"),
            generator=BundleGenerator(
                name="pharma-lab", version="0.1.0", build_id="build-test"
            ),
            source_digests={"source_pdf_sha256": "0" * 64},
            document_count=0,
            section_count=0,
            files={},
        ),
        documents=[paracetamol, leaflet],
        sections=[
            SectionRecord(
                key="drug:paracetamol:lieu-luong-va-cach-dung",
                document_key=paracetamol.key,
                heading="Liều lượng và cách dùng",
                context_path=["Liều lượng và cách dùng"],
                ordinal=1,
                start_page=1203,
                end_page=1204,
                blocks=[
                    BlockRecord(
                        kind=BlockKind.PROSE,
                        markdown="Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần.",
                    )
                ],
            ),
            SectionRecord(
                key="drug:paracetamol:tuong-tac-thuoc",
                document_key=paracetamol.key,
                heading="Tương tác thuốc",
                context_path=["Tương tác thuốc"],
                ordinal=2,
                blocks=[
                    BlockRecord(
                        kind=BlockKind.TABLE,
                        markdown="| Thuốc | Hậu quả |\n| --- | --- |\n| Warfarin | Tăng INR |",
                        start_page=1205,
                        end_page=1205,
                        table_key="tbl-0042",
                    )
                ],
            ),
            SectionRecord(
                key=LEAFLET_SECTION,
                document_key=leaflet.key,
                heading="Thông tin chi tiết",
                context_path=["Thông tin chi tiết"],
                ordinal=1,
                blocks=[
                    BlockRecord(
                        kind=BlockKind.PROSE,
                        markdown="Panadol đỏ chứa paracetamol 500 mg và cafein 65 mg.",
                    )
                ],
            ),
        ],
        glossary=[
            GlossaryEntry(
                term="INR",
                case_sensitive=True,
                vietnamese_expansions=["tỷ số chuẩn hóa quốc tế"],
                english_expansions=["International Normalized Ratio"],
                category="laboratory",
                confidence="high",
                source="curated",
            )
        ],
        colloquial_mappings=[
            ColloquialMappingRecord(
                key="panadol-extra-gsk-150-vien-11440",
                aliases=["Panadol đỏ", "Panadol hộp đỏ"],
                visual_sign="Hộp màu đỏ, vỉ thuốc màu đỏ",
                product_names=["Panadol Extra GSK"],
                section_keys=[LEAFLET_SECTION],
            )
        ],
    )


def write_full_bundle(directory: Path) -> BundleManifest:
    write_bundle(make_bundle(), directory)
    return write_bundle_embeddings(directory, MODEL, sorted(VECTORS.items()))


def rewrite(directory: Path, name: str, content: str) -> None:
    """Replace a bundle file and refresh its manifest digest."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data = path.read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def edit_manifest(directory: Path, **changes: object) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(changes)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def problems_of(directory: Path) -> list[str]:
    with pytest.raises(BundleValidationError) as caught:
        read_bundle(directory)
    return caught.value.problems


def embedding_problems_of(directory: Path, manifest: BundleManifest) -> list[str]:
    with pytest.raises(BundleValidationError) as caught:
        read_bundle_embeddings(directory, manifest, MODEL, 4)
    return caught.value.problems


def jsonl(*records: object) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def test_write_then_read_round_trips_with_computed_manifest(tmp_path: Path) -> None:
    bundle = make_bundle()

    manifest = write_full_bundle(tmp_path)

    assert manifest.document_count == 2
    assert manifest.section_count == 3
    assert sorted(manifest.files) == [
        "colloquial_mappings.json",
        "documents.jsonl",
        EMBEDDINGS_FILE,
        "glossary.json",
        "sections.jsonl",
    ]
    assert manifest.embeddings == [
        BundleEmbeddingFile(model=MODEL, dims=4, file=EMBEDDINGS_FILE)
    ]
    sections_text = (tmp_path / "sections.jsonl").read_text(encoding="utf-8")
    assert len(sections_text.splitlines()) == 3
    assert "Liều lượng và cách dùng" in sections_text
    assert read_bundle(tmp_path) == bundle.model_copy(update={"manifest": manifest})
    assert read_bundle_embeddings(tmp_path, manifest, MODEL, 4) == VECTORS


def test_write_bundle_writes_content_and_replaces_placeholder_manifest_fields(
    tmp_path: Path,
) -> None:
    bundle = make_bundle()
    placeholder = bundle.manifest.model_copy(
        update={
            "document_count": 99,
            "section_count": 99,
            "files": {"documents.jsonl": BundleFile(sha256="0" * 64, bytes=0)},
            "embeddings": [
                BundleEmbeddingFile(
                    model="stale-model", dims=8, file="embeddings/stale_model.jsonl"
                )
            ],
        }
    )

    manifest = write_bundle(
        bundle.model_copy(update={"manifest": placeholder}), tmp_path
    )

    documents = (tmp_path / "documents.jsonl").read_bytes()
    assert manifest.document_count == 2
    assert manifest.section_count == 3
    assert manifest.files["documents.jsonl"] == BundleFile(
        sha256=hashlib.sha256(documents).hexdigest(), bytes=len(documents)
    )
    assert sorted(manifest.files) == [
        "colloquial_mappings.json",
        "documents.jsonl",
        "glossary.json",
        "sections.jsonl",
    ]
    assert manifest.embeddings == []
    assert not (tmp_path / "embeddings").exists()
    assert read_bundle(tmp_path).manifest == manifest


@pytest.mark.parametrize("key", ["", "panadol-extra-gsk-150-vien-11440"])
def test_colloquial_mapping_key_may_be_empty_or_a_slug(
    tmp_path: Path, key: str
) -> None:
    bundle = make_bundle()
    bundle.colloquial_mappings[0].key = key

    write_bundle(bundle, tmp_path)

    assert read_bundle(tmp_path).colloquial_mappings[0].key == key


def test_bundle_files_do_not_depend_on_the_order_models_are_embedded(
    tmp_path: Path,
) -> None:
    other = {SHA_DOSE: [0.25, 0.75]}
    write_bundle(make_bundle(), tmp_path / "a")
    write_bundle_embeddings(tmp_path / "a", MODEL, sorted(VECTORS.items()))
    first = write_bundle_embeddings(tmp_path / "a", "other-2d", other.items())
    write_bundle(make_bundle(), tmp_path / "b")
    write_bundle_embeddings(tmp_path / "b", "other-2d", other.items())
    second = write_bundle_embeddings(tmp_path / "b", MODEL, sorted(VECTORS.items()))

    assert first == second
    assert list(first.files) == [
        "documents.jsonl",
        "sections.jsonl",
        "glossary.json",
        "colloquial_mappings.json",
        EMBEDDINGS_FILE,
        "embeddings/other_2d.jsonl",
    ]
    assert (tmp_path / "a" / "manifest.json").read_bytes() == (
        tmp_path / "b" / "manifest.json"
    ).read_bytes()
    for name in first.files:
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_write_bundle_embeddings_replaces_one_model_and_keeps_the_others(
    tmp_path: Path,
) -> None:
    write_full_bundle(tmp_path)
    write_bundle_embeddings(tmp_path, "other-2d", [(SHA_TABLE, [0.25, 0.75])])
    replaced = {SHA_DOSE: [2.0, 2.0, 2.0, 2.0]}

    manifest = write_bundle_embeddings(tmp_path, MODEL, replaced.items())

    assert manifest.embeddings == [
        BundleEmbeddingFile(model=MODEL, dims=4, file=EMBEDDINGS_FILE),
        BundleEmbeddingFile(model="other-2d", dims=2, file="embeddings/other_2d.jsonl"),
    ]
    assert read_bundle(tmp_path).manifest == manifest
    assert read_bundle_embeddings(tmp_path, manifest, MODEL, 4) == replaced
    assert read_bundle_embeddings(tmp_path, manifest, "other-2d", 2) == {
        SHA_TABLE: [0.25, 0.75]
    }
    assert sorted(path.name for path in (tmp_path / "embeddings").iterdir()) == [
        "fake_embedding_4d.jsonl",
        "other_2d.jsonl",
    ]


def test_write_bundle_rejects_invalid_content_before_writing(tmp_path: Path) -> None:
    bundle = make_bundle()
    bundle.documents.append(bundle.documents[0])

    with pytest.raises(BundleValidationError) as caught:
        write_bundle(bundle, tmp_path)

    assert caught.value.problems == [
        "documents.jsonl:3: key: duplicate document key 'drug:paracetamol' "
        "(first on line 1)",
    ]
    assert not (tmp_path / "manifest.json").exists()


def test_write_bundle_embeddings_rejects_invalid_vectors_and_keeps_the_bundle(
    tmp_path: Path,
) -> None:
    manifest = write_full_bundle(tmp_path)
    before = (tmp_path / EMBEDDINGS_FILE).read_bytes()
    first, second = sorted(VECTORS)

    with pytest.raises(BundleValidationError) as caught:
        write_bundle_embeddings(
            tmp_path,
            MODEL,
            [
                (second, [1.0] * 4),
                (first, [1.0] * 3),
                (first, [1.0] * 4),
                ("abc", [1.0] * 4),
            ],
        )

    where = f"embeddings[{MODEL!r}]"
    assert caught.value.problems == [
        f"{where}: {first!r} is not sorted after {second!r}",
        f"{where}: {first!r} is a duplicate",
        f"{where}: 'abc' is not a sha256 hex digest",
        f"{where}: vectors must share one non-zero length, got [3, 4]",
    ]
    assert (tmp_path / EMBEDDINGS_FILE).read_bytes() == before
    assert read_bundle(tmp_path).manifest == manifest
    assert sorted(path.name for path in (tmp_path / "embeddings").iterdir()) == [
        "fake_embedding_4d.jsonl"
    ]


def test_write_bundle_embeddings_needs_a_manifest_vectors_and_a_free_slug(
    tmp_path: Path,
) -> None:
    with pytest.raises(BundleValidationError) as missing:
        write_bundle_embeddings(tmp_path, MODEL, sorted(VECTORS.items()))
    assert missing.value.problems == ["manifest.json: file is missing"]

    write_full_bundle(tmp_path)
    cases = {
        "!!!": "embeddings['!!!']: model name must contain letters or digits",
        "Fake Embedding 4D": "embeddings['Fake Embedding 4D']: model slug collides "
        f"with {MODEL!r}",
    }
    for model, problem in cases.items():
        with pytest.raises(BundleValidationError) as caught:
            write_bundle_embeddings(tmp_path, model, sorted(VECTORS.items()))
        assert caught.value.problems == [problem]
    with pytest.raises(BundleValidationError) as empty:
        write_bundle_embeddings(tmp_path, "other-2d", [])
    assert empty.value.problems == [
        "embeddings['other-2d']: vectors must share one non-zero length, got []"
    ]


def test_missing_manifest(tmp_path: Path) -> None:
    assert problems_of(tmp_path) == ["manifest.json: file is missing"]


def test_schema_version_must_match(tmp_path: Path) -> None:
    write_bundle(make_bundle(), tmp_path)
    edit_manifest(tmp_path, schema_version="knowledge-bundle/v2")

    assert problems_of(tmp_path) == [
        "manifest.json: schema_version: Input should be 'knowledge-bundle/v1'"
    ]


def test_file_digests_sizes_and_listing_must_match(tmp_path: Path) -> None:
    manifest_written = write_full_bundle(tmp_path)
    written_bytes = manifest_written.files["documents.jsonl"].bytes
    embedding_bytes = manifest_written.files[EMBEDDINGS_FILE].bytes
    for name in ("documents.jsonl", EMBEDDINGS_FILE):
        with (tmp_path / name).open("a", encoding="utf-8") as handle:
            handle.write("\n")
    (tmp_path / "glossary.json").unlink()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    edit_manifest(
        tmp_path,
        files={**manifest["files"], "notes.txt": {"sha256": "0" * 64, "bytes": 0}},
        document_count=5,
    )

    assert problems_of(tmp_path) == [
        f"documents.jsonl: size {written_bytes + 1} bytes does not match manifest "
        f"({written_bytes})",
        "documents.jsonl: sha256 does not match manifest",
        "glossary.json: file is missing",
        f"{EMBEDDINGS_FILE}: size {embedding_bytes + 1} bytes does not match manifest "
        f"({embedding_bytes})",
        f"{EMBEDDINGS_FILE}: sha256 does not match manifest",
        "manifest.json: files.notes.txt: not part of knowledge-bundle/v1",
        "manifest.json: document_count: 5 does not match documents.jsonl (2 records)",
    ]


def test_record_fields_are_reported_with_line_and_path(tmp_path: Path) -> None:
    bundle = make_bundle()
    write_bundle(bundle, tmp_path)
    rewrite(
        tmp_path,
        "documents.jsonl",
        jsonl(
            bundle.documents[0].model_dump(mode="json"),
            {
                "key": "drug:ibuprofen",
                "kind": "brand_page",
                "title": "IBUPROFEN",
                "source": {"title": "Dược thư Quốc gia Việt Nam 2022"},
            },
        ),
    )

    assert problems_of(tmp_path) == [
        "documents.jsonl:2: kind: Input should be 'drug_monograph', "
        "'general_monograph' or 'leaflet'",
        "sections.jsonl:3: document_key: unknown document "
        "'leaflet:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440'",
    ]


def test_section_keys_ordinals_blocks_and_pages_are_validated(tmp_path: Path) -> None:
    bundle = make_bundle()
    write_bundle(bundle, tmp_path)
    first = bundle.sections[0].model_dump(mode="json")
    duplicate = {
        **first,
        "start_page": 1206,
        "end_page": 1205,
        "blocks": [{"kind": "prose", "markdown": "   ", "start_page": 0}],
    }
    orphan = {
        **first,
        "key": "drug:ibuprofen:chi-dinh",
        "document_key": "drug:ibuprofen",
        "blocks": [],
    }
    rewrite(tmp_path, "sections.jsonl", jsonl(first, duplicate, orphan))

    assert problems_of(tmp_path) == [
        "sections.jsonl:2: key: duplicate section key "
        "'drug:paracetamol:lieu-luong-va-cach-dung' (first on line 1)",
        "sections.jsonl:2: ordinal: duplicate ordinal 1 in document "
        "'drug:paracetamol' (first on line 1)",
        "sections.jsonl:2: start_page: 1206 is after end_page 1205",
        "sections.jsonl:2: blocks[0].markdown: must not be empty",
        "sections.jsonl:2: blocks[0].start_page: must be >= 1, got 0",
        "sections.jsonl:3: document_key: unknown document 'drug:ibuprofen'",
        "sections.jsonl:3: blocks: must contain at least one block",
        f"colloquial_mappings.json: [0].section_keys[0]: unknown section "
        f"'{LEAFLET_SECTION}'",
    ]


def test_glossary_terms_must_be_present_and_unique_ignoring_case(
    tmp_path: Path,
) -> None:
    write_bundle(make_bundle(), tmp_path)
    rewrite(
        tmp_path,
        "glossary.json",
        json.dumps(
            [
                {"term": "INR", "vietnamese_expansions": ["tỷ số chuẩn hóa quốc tế"]},
                {
                    "term": "inr",
                    "english_expansions": ["International Normalized Ratio"],
                },
                {"term": " ", "aliases": ["x"]},
                {"case_sensitive": True},
            ],
            ensure_ascii=False,
        ),
    )

    assert problems_of(tmp_path) == ["glossary.json: [3].term: Field required"]

    rewrite(
        tmp_path,
        "glossary.json",
        json.dumps(
            [
                {"term": "INR", "vietnamese_expansions": ["tỷ số chuẩn hóa quốc tế"]},
                {
                    "term": "inr",
                    "english_expansions": ["International Normalized Ratio"],
                },
                {"term": " ", "aliases": ["x"]},
            ],
            ensure_ascii=False,
        ),
    )

    assert problems_of(tmp_path) == [
        "glossary.json: [1].term: duplicate term 'inr' (case-insensitive, first at [0])",
        "glossary.json: [2].term: must not be empty",
    ]


def test_a_section_belongs_to_at_most_one_colloquial_mapping(tmp_path: Path) -> None:
    write_bundle(make_bundle(), tmp_path)
    rewrite(
        tmp_path,
        "colloquial_mappings.json",
        json.dumps(
            [
                {
                    "key": "panadol-extra-gsk-150-vien-11440",
                    "section_keys": [LEAFLET_SECTION],
                },
                {
                    "key": "",
                    "product_names": ["Panadol Extra GSK"],
                    "section_keys": [LEAFLET_SECTION],
                },
            ],
            ensure_ascii=False,
        ),
    )

    assert problems_of(tmp_path) == [
        f"colloquial_mappings.json: [1].section_keys[0]: section '{LEAFLET_SECTION}' "
        "is already mapped by [0]"
    ]


def test_embedding_lines_are_validated(tmp_path: Path) -> None:
    write_full_bundle(tmp_path)
    rewrite(
        tmp_path,
        EMBEDDINGS_FILE,
        jsonl(
            {
                "embedding_text_sha256": SHA_DOSE,
                "dims": 3,
                "vector": encode_vector([1.0] * 3),
            },
            {
                "embedding_text_sha256": SHA_TABLE,
                "dims": 4,
                "vector": encode_vector([1.0] * 3),
            },
            {
                "embedding_text_sha256": SHA_TABLE,
                "dims": 4,
                "vector": encode_vector([1.0] * 4),
            },
            {
                "embedding_text_sha256": "abc",
                "dims": 4,
                "vector": encode_vector([1.0] * 4),
            },
        ),
    )
    expected = [
        f"{EMBEDDINGS_FILE}:1: dims: 3 does not match manifest dims 4",
        f"{EMBEDDINGS_FILE}:2: vector: vector has 12 bytes, expected 16 (dims * 4)",
        f"{EMBEDDINGS_FILE}:3: embedding_text_sha256: duplicate of line 2",
        f"{EMBEDDINGS_FILE}:4: embedding_text_sha256: String should match pattern "
        "'^[0-9a-f]{64}$'",
    ]

    assert problems_of(tmp_path) == expected
    manifest = BundleManifest.model_validate_json(
        (tmp_path / "manifest.json").read_bytes()
    )
    assert embedding_problems_of(tmp_path, manifest) == expected


def test_read_bundle_embeddings_loads_only_a_declared_model_and_rechecks_its_file(
    tmp_path: Path,
) -> None:
    manifest = write_full_bundle(tmp_path)

    assert read_bundle_embeddings(tmp_path, manifest, MODEL, 8) == {}
    assert read_bundle_embeddings(tmp_path, manifest, "other-2d", 2) == {}

    size = manifest.files[EMBEDDINGS_FILE].bytes
    with (tmp_path / EMBEDDINGS_FILE).open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert embedding_problems_of(tmp_path, manifest) == [
        f"{EMBEDDINGS_FILE}: size {size + 1} bytes does not match manifest ({size})",
        f"{EMBEDDINGS_FILE}: sha256 does not match manifest",
    ]
    (tmp_path / EMBEDDINGS_FILE).unlink()
    assert embedding_problems_of(tmp_path, manifest) == [
        f"{EMBEDDINGS_FILE}: file is missing"
    ]


def test_manifest_embedding_entries_must_name_their_file(tmp_path: Path) -> None:
    write_full_bundle(tmp_path)
    edit_manifest(
        tmp_path,
        embeddings=[{"model": MODEL, "dims": 0, "file": "embeddings/other.jsonl"}],
    )

    assert problems_of(tmp_path) == [
        "manifest.json: embeddings[0].dims: must be >= 1, got 0",
        "manifest.json: embeddings[0].file: expected "
        f"'{EMBEDDINGS_FILE}', got 'embeddings/other.jsonl'",
    ]
