import json
from pathlib import Path

import pytest
from pharma_agent.domain.corpus.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BlockKind,
    BundleCollection,
    ColloquialMappingRecord,
    DocumentKind,
    KnowledgeBundle,
    RetrievalMode,
    SourceInfo,
    read_bundle,
)

from seed_pipeline.bundle.chunks import gold_chunk_label, iter_section_chunks
from seed_pipeline.bundle.export import (
    COLLECTION_KEY,
    COLLECTION_TITLE,
    ExportRequest,
    ExportResult,
    block_kind_for,
    export_bundle,
    leaflet_blocks,
)
from seed_pipeline.corpus.canonical.build_canonical_rag import (
    ATC_SECTION_ID,
    BRAND_INDEX_SECTION_ID,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
DOSAGE = "drug:paracetamol:lieu-luong-va-cach-dung"
PANADOL = "brand:ankhang:thuoc:panadol-extra-gsk-150-vien-11440"
HAPACOL = "brand:ankhang:thuoc:hapacol-250-dhg-11500"


def _request(output_dir: Path, *, force: bool = False) -> ExportRequest:
    return ExportRequest(
        rag_final_dir=FIXTURE,
        glossary_path=FIXTURE / "term_glossary.json",
        mappings_path=FIXTURE / "colloquial_mappings.json",
        output_dir=output_dir,
        force=force,
    )


def _export(tmp_path: Path) -> tuple[ExportResult, KnowledgeBundle]:
    result = export_bundle(_request(tmp_path / "bundle"))
    return result, read_bundle(tmp_path / "bundle")


def test_export_groups_documents_and_keeps_section_keys(tmp_path: Path) -> None:
    result, bundle = _export(tmp_path)

    assert [document.key for document in bundle.documents] == [
        "drug:paracetamol",
        "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat",
        "leaflet:ankhang:thuoc:hapacol-250-dhg-11500",
        "leaflet:ankhang:thuoc:panadol-extra-gsk-150-vien-11440",
    ]
    assert [document.kind for document in bundle.documents] == [
        DocumentKind.DRUG_MONOGRAPH,
        DocumentKind.GENERAL_MONOGRAPH,
        DocumentKind.LEAFLET,
        DocumentKind.LEAFLET,
    ]
    assert bundle.documents[0].source.url is None
    assert bundle.documents[2].source == SourceInfo(
        title="Tờ hướng dẫn sử dụng",
        url="https://www.nhathuocankhang.com/thuoc/hapacol-250-dhg-11500",
    )
    assert [section.key for section in bundle.sections] == [
        DOSAGE,
        "drug:paracetamol:chong-chi-dinh",
        BRAND_INDEX_SECTION_ID,
        HAPACOL,
        PANADOL,
    ]
    assert [section.ordinal for section in bundle.sections] == [1, 2, 1, 1, 1]
    assert result.skipped_sections == ()
    assert result.manifest.schema_version == BUNDLE_SCHEMA_VERSION
    assert result.manifest.collection == BundleCollection(
        key=COLLECTION_KEY, title=COLLECTION_TITLE
    )
    assert (result.manifest.document_count, result.manifest.section_count) == (4, 5)
    assert result.manifest.generator.name == "seed-pipeline"
    assert result.manifest.generator.build_id == "fixture-build-0001"
    assert result.manifest.source_digests == {
        "source_pdf_sha256": "1" * 64,
        "leaflet_source_manifest_sha256": "2" * 64,
        "leaflet_source_file_count": "2",
        "curated_curated_tables_sha256": "5" * 64,
        "curated_glossary_sha256": "6" * 64,
        "curated_mappings_sha256": "7" * 64,
        "curated_table_overrides_sha256": "8" * 64,
    }


def test_export_maps_block_kinds_retrieval_and_pages(tmp_path: Path) -> None:
    _, bundle = _export(tmp_path)
    sections = {section.key: section for section in bundle.sections}
    old_chunks = [
        json.loads(line)
        for line in (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    ]

    dosage = sections[DOSAGE]
    assert [block.kind for block in dosage.blocks] == [BlockKind.PROSE, BlockKind.TABLE]
    assert dosage.blocks[1].table_key == "curated-table-1134-001"
    assert (dosage.blocks[1].start_page, dosage.blocks[1].end_page) == (1134, 1134)
    assert (dosage.start_page, dosage.end_page) == (1133, 1134)
    assert dosage.heading == "Liều lượng và cách dùng"
    assert dosage.context_path == ["Liều lượng và cách dùng"]

    index = sections[BRAND_INDEX_SECTION_ID]
    assert index.retrieval is RetrievalMode.INDEX_ONLY
    assert [block.kind for block in index.blocks] == [BlockKind.INDEX_ENTRIES]
    assert sections[DOSAGE].retrieval is RetrievalMode.DEFAULT

    leaflet = sections[PANADOL]
    assert (leaflet.start_page, leaflet.end_page) == (None, None)
    assert [block.kind for block in leaflet.blocks] == [BlockKind.PROSE]
    assert [block.start_page for block in leaflet.blocks] == [None]
    old_leaflet = next(chunk for chunk in old_chunks if chunk["section_id"] == PANADOL)
    assert leaflet.blocks[0].markdown == old_leaflet["chunk_text"]

    assert block_kind_for(ATC_SECTION_ID, "paragraph") is BlockKind.LIST
    assert block_kind_for(ATC_SECTION_ID, "table") is BlockKind.TABLE


def test_export_copies_glossary_and_attaches_mappings(tmp_path: Path) -> None:
    _, bundle = _export(tmp_path)
    mappings = {mapping.key: mapping for mapping in bundle.colloquial_mappings}

    assert len(mappings) == len(bundle.colloquial_mappings)
    assert [entry.term for entry in bundle.glossary] == ["ADR", "NSAID"]
    assert mappings["panadol-extra-gsk-150-vien-11440"] == ColloquialMappingRecord(
        key="panadol-extra-gsk-150-vien-11440",
        aliases=["Panadol đỏ", "Panadol extra đỏ", "Panadol vỉ đỏ", "Panadol hộp đỏ"],
        visual_sign="Hộp màu đỏ, vỉ thuốc màu đỏ",
        product_names=["Panadol Extra GSK"],
        section_keys=[PANADOL],
    )
    assert mappings["hapacol-250-dhg-11500"] == ColloquialMappingRecord(
        key="hapacol-250-dhg-11500",
        product_names=["Hapacol 250 DHG"],
        section_keys=[HAPACOL],
    )


def test_export_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    export_bundle(_request(tmp_path / "bundle"))

    with pytest.raises(FileExistsError, match="--force"):
        export_bundle(_request(tmp_path / "bundle"))
    replaced = export_bundle(_request(tmp_path / "bundle", force=True))
    assert replaced.manifest.section_count == 5


def test_leaflet_blocks_keep_long_pipe_tables_apart() -> None:
    rows = "\n".join(f"| thuốc {index} | {'x' * 40} |" for index in range(80))
    table = f"| Thành phần | Hàm lượng |\n| --- | --- |\n{rows}"

    blocks = leaflet_blocks(f"# Thuốc\n\nĐoạn một.\n\n{table}\n\n\nĐoạn hai.")

    assert [block.kind for block in blocks] == [
        BlockKind.PROSE,
        BlockKind.TABLE,
        BlockKind.PROSE,
    ]
    assert blocks[0].markdown == "# Thuốc\n\nĐoạn một."
    assert blocks[1].markdown == table
    assert blocks[2].markdown == "Đoạn hai."


def test_iter_section_chunks_runs_backend_chunker(tmp_path: Path) -> None:
    _, bundle = _export(tmp_path)

    items = list(iter_section_chunks(bundle))

    assert [item.section.key for item in items] == [s.key for s in bundle.sections]
    assert [draft.ordinal for draft in items[0].drafts] == [1, 2]
    assert items[0].document.key == "drug:paracetamol"
    assert gold_chunk_label(DOSAGE, 2) == f"{DOSAGE}:chunk-002"
