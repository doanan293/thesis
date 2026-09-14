import json
from pathlib import Path

from pharma_agent.domain.corpus.bundle import KnowledgeBundle, read_bundle

from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.bundle.parity import check_chunk_parity, legacy_chunk_view

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
HAPACOL = "leaflet:thuoc:hapacol-250-dhg-11500"


def _bundle(tmp_path: Path) -> KnowledgeBundle:
    export_bundle(
        ExportRequest(
            rag_final_dir=FIXTURE,
            glossary_path=FIXTURE / "term_glossary.json",
            mappings_path=FIXTURE / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    return read_bundle(tmp_path / "bundle")


def _old_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    ]


def test_backend_chunker_matches_the_old_fixture_build(tmp_path: Path) -> None:
    report = check_chunk_parity(_bundle(tmp_path), _old_chunks())

    assert report.ok, report.to_dict()
    assert (report.sections_checked, report.chunks_checked) == (5, 6)


def test_changed_embedding_text_is_reported(tmp_path: Path) -> None:
    old = _old_chunks()
    old[2]["embedding_text"] = old[2]["embedding_text"] + " thay đổi"

    report = check_chunk_parity(_bundle(tmp_path), old)

    assert not report.ok
    assert report.mismatch_count == 1
    mismatch = report.mismatches[0]
    assert (mismatch.section_key, mismatch.ordinal, mismatch.field) == (
        "drug:paracetamol:chong-chi-dinh",
        1,
        "embedding_text",
    )


def test_sections_missing_from_the_bundle_are_reported(tmp_path: Path) -> None:
    old = _old_chunks()
    old.append({**old[0], "section_id": "drug:ghost:lieu-dung"})

    report = check_chunk_parity(_bundle(tmp_path), old)

    assert [(m.section_key, m.field) for m in report.mismatches] == [
        ("drug:ghost:lieu-dung", "missing_section")
    ]


def test_legacy_view_maps_page_zero_to_none_and_keeps_the_old_mapping() -> None:
    view = legacy_chunk_view(
        {
            "section_id": HAPACOL,
            "chunk_index": 1,
            "chunk_text": "Hạ sốt",
            "embedding_text": "Hạ sốt",
            "hydrate_strategy": "full_section",
            "start_page": 0,
            "end_page": 0,
            "colloquial_mapping": {"product_names": ["Hapacol 250 DHG"]},
        }
    )

    assert (view["start_page"], view["end_page"]) == (None, None)
    assert view["colloquial_mapping"] == {"product_names": ["Hapacol 250 DHG"]}
    assert view["term_annotations"] == []
