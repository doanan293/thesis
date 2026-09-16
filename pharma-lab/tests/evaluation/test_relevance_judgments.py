import json
from pathlib import Path

import pytest

from pharma_lab.evaluation.artifact_contracts import (
    ArtifactContractError,
    sha256_file,
)
from pharma_lab.evaluation.relevance_judgments import (
    active_ingredients,
    build_judgments,
    chunk_intents,
    formulary_intent,
    judgments_path,
    load_judgments,
    same_active_ingredients,
    write_judgments,
)

LEAFLET_PAGE = [
    "# Acarbose A 50 mg\n\n### 1. Thành phần\n\nMỗi viên chứa:\n\nAcarbose 50 mg.\n\n"
    "Tá dược: lactose 10 mg.\n\n### 2. Công dụng (Chỉ định)\n\nĐiều trị đái tháo đường.\n\n"
    "### 3. Cách dùng - Liều dùng\n\nUống 50 mg mỗi ngày.\n\n#### - Quá liều\n\n"
    "Có thể gây tiêu chảy.",
    "Không có thuốc giải độc đặc hiệu.\n\n### 4. Chống chỉ định\n\nQuá mẫn với acarbose.",
    "- Thận trọng khi sử dụng\n\nTheo dõi đường huyết.",
    "| Chướng khí | Thường gặp |\n\n### 6. Dược lý\n\nỨc chế alpha-glucosidase.",
]


@pytest.mark.parametrize(
    ("section_id", "intent"),
    [
        ("drug:acarbose:chi-dinh", "indication"),
        ("drug:acarbose:lieu-luong-va-cach-dung", "dosage"),
        ("drug:acarbose:thoi-ky-cho-con-bu", "pregnancy_lactation"),
        ("drug:acarbose:ten-thuong-mai", None),
        ("leaflet:thuoc-tri-tieu-duong:acarbose-a", None),
    ],
)
def test_formulary_sections_map_to_the_intent_a_leaflet_can_answer(
    section_id: str, intent: str | None
) -> None:
    assert formulary_intent(section_id) == intent


def test_each_chunk_carries_the_intents_of_the_text_it_contains() -> None:
    assert chunk_intents(LEAFLET_PAGE) == [
        frozenset({"ingredient", "indication", "dosage", "overdose"}),
        frozenset({"overdose", "contraindication"}),
        frozenset({"precaution"}),
        frozenset({"precaution", "pharmacology"}),
    ]


def test_active_ingredients_come_from_the_composition_block() -> None:
    assert active_ingredients("\n\n".join(LEAFLET_PAGE)) == ("acarbose",)


def test_active_ingredients_read_composition_tables_and_labels() -> None:
    table = (
        "### 1. Thành phần\n\n| Thành phần | Hàm lượng |\n| --- | --- |\n"
        "| Thiamin mononitrat | 125 mg |\n| Pyridoxin hydroclorid | 125 mg |\n\n"
        "### 2. Công dụng (Chỉ định)\n\nBổ sung vitamin."
    )
    labelled = "### 1. Thành phần\n\nHoạt chất: Paracetamol 500 mg\n\n### 2. Công dụng"

    assert active_ingredients(table) == (
        "thiamin mononitrat",
        "pyridoxin hydroclorid",
    )
    assert active_ingredients(labelled) == ("paracetamol",)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "- Thành phần hoạt chất: Telmisartan 40 mg và amlodipine 5 mg dưới dạng "
            "besilate.",
            ("telmisartan", "amlodipine"),
        ),
        ("Telmisartan 40 mg + Amlodipin 5 mg", ("telmisartan", "amlodipin")),
        (
            "Hoạt chất: Amoxicilin 875 mg (dưới dạng amoxicilin trihydrat compacted "
            "1004,5 mg) bù hàm lượng 1077 mg. Acid clavulanic 125 mg (dưới dạng Kali "
            "clavulanat/Avicel 297,5 mg) bù hàm lượng 332,7 mg.",
            ("amoxicilin", "acid clavulanic"),
        ),
    ],
)
def test_every_dosed_ingredient_of_a_combination_line_is_read(
    line: str, expected: tuple[str, ...]
) -> None:
    text = f"### 1. Thành phần\n\n{line}\n\n### 2. Công dụng (Chỉ định)"

    assert active_ingredients(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "### 2. Công dụng (Chỉ định)\n\nKhông có khối thành phần.",
        "### 1. Thành phần\n\nAcarbose 50 mg\n\n500 mg\n\n### 2. Công dụng",
    ],
)
def test_unreadable_compositions_have_no_active_ingredients(text: str) -> None:
    assert active_ingredients(text) is None


@pytest.mark.parametrize(
    ("ingredients", "title", "expected"),
    [
        (("metformin hydroclorid",), "METFORMIN", True),
        (("acetaminophen",), "PARACETAMOL", True),
        (
            ("amoxicillin trihydrat", "kali clavulanat"),
            "AMOXICILIN VÀ KALI CLAVULANAT",
            True,
        ),
        (("natri clorid",), "NATRI CLORID", True),
        (("esomeprazole magnesium",), "ESOMEPRAZOL", True),
        (("paracetamol", "codein phosphat"), "PARACETAMOL", False),
        (("glucose khan duoi dang glucose monohydrat",), "GLUCOSE", False),
        (("acarbose",), "METFORMIN", False),
    ],
)
def test_a_leaflet_matches_a_monograph_only_with_the_same_active_ingredients(
    ingredients: tuple[str, ...], title: str, expected: bool
) -> None:
    assert same_active_ingredients(ingredients, title) is expected


def _chunk(section_id: str, index: int, title: str, text: str) -> dict:
    return {
        "chunk_id": f"{section_id}:chunk-{index:03d}",
        "section_id": section_id,
        "chunk_index": index,
        "title": title,
        "chunk_text": text,
    }


def test_judgments_accept_same_ingredient_leaflet_chunks_of_the_asked_intent() -> None:
    acarbose = "leaflet:thuoc-tri-tieu-duong:acarbose-a"
    combination = "leaflet:thuoc-tri-tieu-duong:acarbose-metformin"
    chunks = [
        _chunk("drug:acarbose:chi-dinh", 1, "ACARBOSE", "Điều trị đái tháo đường."),
        _chunk(acarbose, 1, "Acarbose A 50 mg", LEAFLET_PAGE[0]),
        _chunk(acarbose, 2, "Acarbose A 50 mg", LEAFLET_PAGE[1]),
        _chunk(
            combination,
            1,
            "Acarbose Metformin",
            "### 1. Thành phần\n\nAcarbose 50 mg\n\nMetformin 500 mg\n\n"
            "### 2. Công dụng (Chỉ định)\n\nĐiều trị đái tháo đường.",
        ),
    ]
    rows = [
        {
            "query_id": "q-dosage",
            "eval_group": "formulary",
            "answer_mode": "single",
            "retrieval_granularity": "section",
            "expected_section_id": "drug:acarbose:lieu-luong-va-cach-dung",
        },
        {
            "query_id": "q-multi",
            "eval_group": "multi_intent",
            "answer_mode": "multi_required",
            "retrieval_granularity": "multi_section",
            "expected_section_id": "drug:acarbose:chi-dinh",
            "expected_section_ids": [
                "drug:acarbose:chi-dinh",
                "drug:acarbose:tuong-tac-thuoc",
            ],
        },
        {
            "query_id": "q-brand",
            "eval_group": "formulary",
            "answer_mode": "single",
            "retrieval_granularity": "chunk_exact",
            "expected_section_id": "drug:acarbose:ten-thuong-mai",
        },
        {
            "query_id": "q-leaflet",
            "eval_group": "leaflet",
            "answer_mode": "single",
            "retrieval_granularity": "section",
            "expected_section_id": acarbose,
        },
    ]

    judgments = build_judgments(rows, chunks)

    assert judgments == [
        {
            "query_id": "q-dosage",
            "intents": [
                {
                    "section_id": "drug:acarbose:lieu-luong-va-cach-dung",
                    "intent": "dosage",
                    "accepted_chunk_ids": [f"{acarbose}:chunk-001"],
                }
            ],
        },
        {
            "query_id": "q-multi",
            "intents": [
                {
                    "section_id": "drug:acarbose:chi-dinh",
                    "intent": "indication",
                    "accepted_chunk_ids": [f"{acarbose}:chunk-001"],
                }
            ],
        },
    ]


ACCEPTED = "leaflet:thuoc-tri-tieu-duong:acarbose-a:chunk-001"
JUDGMENT = {
    "query_id": "q-dosage",
    "intents": [
        {
            "section_id": "drug:acarbose:lieu-luong-va-cach-dung",
            "intent": "dosage",
            "accepted_chunk_ids": [ACCEPTED],
        }
    ],
}


def _gold_file(tmp_path: Path) -> Path:
    path = tmp_path / "section_retrieval_eval.jsonl"
    path.write_text(
        json.dumps({"query_id": "q-dosage", "query": "liều acarbose"}) + "\n",
        encoding="utf-8",
    )
    return path


def test_judgments_live_next_to_the_gold_file_and_round_trip(tmp_path: Path) -> None:
    evaluation = _gold_file(tmp_path)
    path = judgments_path(evaluation)

    write_judgments([JUDGMENT], evaluation_path=evaluation, output_path=path)
    loaded = load_judgments(path, evaluation_sha256=sha256_file(evaluation))

    assert path == tmp_path / "section_retrieval_eval.judgments.jsonl"
    assert loaded.sha256 == sha256_file(path)
    assert loaded.for_query("q-dosage") == {
        "drug:acarbose:lieu-luong-va-cach-dung": frozenset({ACCEPTED})
    }
    assert loaded.for_query("q-other") == {}


def test_judgments_of_another_gold_file_are_rejected(tmp_path: Path) -> None:
    evaluation = _gold_file(tmp_path)
    path = judgments_path(evaluation)
    write_judgments([JUDGMENT], evaluation_path=evaluation, output_path=path)

    with pytest.raises(ArtifactContractError, match="different evaluation file"):
        load_judgments(path, evaluation_sha256="0" * 64)


def test_edited_judgments_are_rejected(tmp_path: Path) -> None:
    evaluation = _gold_file(tmp_path)
    path = judgments_path(evaluation)
    write_judgments([JUDGMENT], evaluation_path=evaluation, output_path=path)
    path.write_text("", encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="checksum"):
        load_judgments(path, evaluation_sha256=sha256_file(evaluation))


def test_missing_judgments_name_the_command_that_builds_them(tmp_path: Path) -> None:
    with pytest.raises(ArtifactContractError, match="pharma-lab evaluation judgments"):
        load_judgments(tmp_path / "missing.judgments.jsonl", evaluation_sha256="e" * 64)
