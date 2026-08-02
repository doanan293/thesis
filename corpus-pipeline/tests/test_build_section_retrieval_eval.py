import json
from pathlib import Path

from corpus_pipeline.evaluation.build_section_retrieval_eval import (
    DEFAULT_OUTPUT_PATH,
    make_row,
    save_jsonl,
)
from corpus_pipeline.evaluation.section_eval_schema import validate_query_rows


def test_default_output_path_is_jsonl():
    assert DEFAULT_OUTPUT_PATH.name == "section_retrieval_eval.jsonl"


def test_make_row_returns_native_json_types():
    section = {
        "id": "sec_test_1",
        "title": "Test Title",
        "section": "Chỉ định",
    }
    row = make_row(
        query_id="q1",
        query="Test query?",
        eval_group="formulary",
        source_family="drug_formulary",
        query_form="natural",
        intent_category="indication",
        source_subcategory="formulary_indication",
        section=section,
        expected_chunk_role="prose",
        difficulty="easy",
        notes="test note",
        expected_section_ids=["sec_test_1"],
        expected_chunk_index=0,
        eval_tags=["tag1", "tag2"],
    )

    assert isinstance(row["expected_section_ids"], list)
    assert row["expected_section_ids"] == ["sec_test_1"]
    assert isinstance(row["eval_tags"], list)
    assert sorted(row["eval_tags"]) == sorted(
        [
            "formulary",
            "drug_formulary",
            "natural",
            "indication",
            "formulary_indication",
            "tag1",
            "tag2",
        ]
    )
    assert isinstance(row["query_intent_count"], int)
    assert row["query_intent_count"] == 1
    assert isinstance(row["expected_chunk_index"], int)
    assert row["expected_chunk_index"] == 0

    # Validate against schema
    validate_query_rows([row])


def test_save_jsonl_writes_valid_jsonl(tmp_path: Path):
    out_file = tmp_path / "test_eval.jsonl"
    rows = [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]
    save_jsonl(out_file, rows)

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1, "name": "a"}
    assert json.loads(lines[1]) == {"id": 2, "name": "b"}


def test_extract_section_entity_anchors():
    from corpus_pipeline.evaluation.build_section_retrieval_eval import (
        build_hybrid_entity_intent_query,
        extract_section_entity_anchors,
    )

    section = {
        "id": "brand:ankhang:100",
        "title": "Thuốc Hapacol 500mg viên sủi",
        "section": "Chống chỉ định",
        "colloquial_names": ["Hapacol", "Paracetamol sủi"],
    }
    chunks = [{"chunk_text": "Hapacol 500mg chứa Paracetamol 500mg"}]
    anchors = extract_section_entity_anchors(section, chunks)
    assert "Hapacol 500mg" in anchors or "Hapacol" in anchors
    assert "Paracetamol" in anchors or "Paracetamol sủi" in anchors

    query = build_hybrid_entity_intent_query(section, "contraindication", anchors)
    assert any(anchor.lower() in query.lower() for anchor in anchors)
    assert "chống chỉ định" in query.lower() or "không nên dùng" in query.lower()


def test_multi_intent_rows_have_entity_anchors():
    from corpus_pipeline.evaluation.build_section_retrieval_eval import (
        build_multi_intent_eval_rows,
    )

    section_a = {
        "id": "formulary:1:sec1",
        "title": "Amoxicillin 500mg",
        "section": "Chỉ định",
    }
    section_b = {
        "id": "formulary:1:sec2",
        "title": "Amoxicillin 500mg",
        "section": "Liều dùng",
    }
    sections = [section_a, section_b]
    chunks_by_section = {
        "formulary:1:sec1": [
            {"chunk_id": "c1", "chunk_text": "Amoxicillin chỉ định nhiễm khuẩn"}
        ],
        "formulary:1:sec2": [
            {"chunk_id": "c2", "chunk_text": "Amoxicillin liều 500mg 3 lần/ngày"}
        ],
    }
    rows = build_multi_intent_eval_rows(sections, chunks_by_section, target_count=1)
    assert len(rows) == 1
    assert "Amoxicillin" in rows[0]["query"]


def test_generated_rows_have_entity_anchors():
    from corpus_pipeline.evaluation.build_section_retrieval_eval import generate_rows

    section = {
        "id": "formulary:1:sec1",
        "title": "Aspirin 500mg",
        "section": "Chỉ định",
    }
    chunks = [
        {
            "chunk_id": "c1",
            "section_id": "formulary:1:sec1",
            "chunk_text": "Aspirin 500mg chỉ định giảm đau",
            "chunk_role": "prose",
        }
    ]

    rows = generate_rows([section], chunks, target_rows=20)
    assert len(rows) > 0
    for row in rows:
        query = row["query"].lower()
        assert (
            "aspirin" in query
            or "c1" in query
            or "dược" in query
            or "phân loại" in query
            or "bảng" in query
        ), f"Expected entity anchor in query, got: {row['query']}"


def test_category_for_section_expanded_monograph_titles():
    from corpus_pipeline.evaluation.build_section_retrieval_eval import (
        category_for_section,
    )

    sec_overdose = {"title": "ABACAVIR - Quá liều và xử trí"}
    cat, _diff, role = category_for_section(sec_overdose, [])
    assert cat == "overdose"
    assert role == "prose"

    sec_storage = {"title": "ACETAZOLAMID - Độ ổn định và bảo quản"}
    cat, _diff, role = category_for_section(sec_storage, [])
    assert cat == "storage"
    assert role == "prose"

    sec_precaution = {"title": "ACENOCOUMAROL - Thận trọng"}
    cat, _diff, role = category_for_section(sec_precaution, [])
    assert cat == "precaution"
    assert role == "prose"

    sec_pharm = {"title": "AMOXICILLIN - Dược lý và cơ chế tác dụng"}
    cat, _diff, role = category_for_section(sec_pharm, [])
    assert cat == "pharmacology"
    assert role == "prose"

    sec_form = {"title": "AUGMENTIN - Dạng thuốc và hàm lượng"}
    cat, _diff, role = category_for_section(sec_form, [])
    assert cat == "dosage_form"
    assert role == "prose"


def test_curated_query_for_section_expanded_categories():
    from corpus_pipeline.evaluation.build_section_retrieval_eval import (
        curated_query_for_section,
    )

    sec_overdose = {
        "title": "ABACAVIR - Quá liều và xử trí",
        "id": "drug:abacavir:qua-lieu-va-xu-tri",
    }
    q_overdose = curated_query_for_section(sec_overdose, "overdose")
    assert "ABACAVIR" in q_overdose
    assert "quá liều" in q_overdose.lower() or "xử trí" in q_overdose.lower()

    sec_storage = {
        "title": "ACETAZOLAMID - Độ ổn định và bảo quản",
        "id": "drug:acetazolamid:do-on-dinh-va-bao-quan",
    }
    q_storage = curated_query_for_section(sec_storage, "storage")
    assert "ACETAZOLAMID" in q_storage
    assert "bảo quản" in q_storage.lower()
