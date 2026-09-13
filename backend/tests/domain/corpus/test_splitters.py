import pytest

from pharma_agent.domain.corpus.chunking import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    split_lines_without_breaking_entries,
    split_long_text,
    split_oversized_paragraph,
    split_table_markdown,
)

PHARMACOKINETICS = (
    "Paracetamol được hấp thu nhanh qua đường tiêu hóa. Nồng độ đỉnh trong huyết "
    "tương đạt sau 30 - 60 phút; thời gian bán thải khoảng 2 giờ và kéo dài khi "
    "suy gan."
)
PHARMACOKINETICS_PARTS = [
    "Paracetamol được hấp thu nhanh qua đường tiêu hóa.",
    "Nồng độ đỉnh trong huyết tương đạt sau 30 - 60 phút;",
    "thời gian bán thải khoảng 2 giờ và kéo dài khi suy gan.",
]
TABLE_HEADER = "| Thuốc phối hợp | Hậu quả |\n| --- | --- |"


def test_chunker_constants() -> None:
    assert CHUNKER_VERSION == "chunker-v1"
    assert MAX_CHUNK_CHARS == 3000


def test_split_long_text_returns_fitting_text_unchanged() -> None:
    text = "Người lớn: uống 500 mg.\n \nTrẻ em: 10 mg/kg."

    assert split_long_text(text, 100) == [text]


def test_split_long_text_packs_paragraphs_and_normalizes_breaks() -> None:
    text = (
        "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần.\n\n"
        "Trẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ.\n\n\n"
        "Không dùng quá 5 lần trong 24 giờ."
    )

    assert split_long_text(text, 90) == [
        "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần.\n\n"
        "Trẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ.",
        "Không dùng quá 5 lần trong 24 giờ.",
    ]


def test_split_long_text_isolates_an_oversized_paragraph() -> None:
    text = f"Hạ sốt.\n\n{PHARMACOKINETICS}\n\nGiảm đau."

    assert split_long_text(text, 80) == [
        "Hạ sốt.",
        *PHARMACOKINETICS_PARTS,
        "Giảm đau.",
    ]


@pytest.mark.parametrize(
    ("paragraph", "max_chars", "parts"),
    [
        (PHARMACOKINETICS, 80, PHARMACOKINETICS_PARTS),
        (
            "Liều dùng cho người lớn\nuống mỗi lần một viên sau ăn\n"
            "không quá tám viên mỗi ngày",
            40,
            [
                "Liều dùng cho người lớn",
                "uống mỗi lần một viên sau ăn",
                "không quá tám viên mỗi ngày",
            ],
        ),
        (
            "Acetylcystein hòa tan trong nước uống sau bữa ăn tối",
            20,
            ["Acetylcystein hòa", "tan trong nước uống", "sau bữa ăn tối"],
        ),
        (
            "Natriphenylbutyratglycerolphenylbutyrat",
            10,
            ["Natripheny", "lbutyratgl", "ycerolphen", "ylbutyrat"],
        ),
    ],
    ids=["sentence-end", "newline", "whitespace", "hard-cut"],
)
def test_split_oversized_paragraph_prefers_sentence_then_newline_then_space(
    paragraph: str, max_chars: int, parts: list[str]
) -> None:
    assert split_oversized_paragraph(paragraph, max_chars) == parts


def test_split_table_markdown_strips_a_fitting_table() -> None:
    table = "\n| Thuốc | Hậu quả |\n| --- | --- |\n| Warfarin | Tăng INR |\n"

    assert split_table_markdown(table, 100) == [table.strip()]


def test_split_table_markdown_repeats_header_and_separator_in_every_part() -> None:
    table = (
        f"{TABLE_HEADER}\n"
        "| Warfarin | Tăng INR |\n"
        "|  Rượu | Tăng độc tính trên gan |\n\n"
        "| Isoniazid | Tăng nguy cơ độc gan |\n"
        "| Cholestyramin | Giảm hấp thu |"
    )

    assert split_table_markdown(table, 70) == [
        f"{TABLE_HEADER}\n| Warfarin | Tăng INR |",
        f"{TABLE_HEADER}\n|  Rượu | Tăng độc tính trên gan |",
        f"{TABLE_HEADER}\n| Isoniazid | Tăng nguy cơ độc gan |",
        f"{TABLE_HEADER}\n| Cholestyramin | Giảm hấp thu |",
    ]


def test_split_table_markdown_without_separator_falls_back_to_prose() -> None:
    table = (
        "| Thuốc | Hậu quả |\n| Warfarin | Tăng INR khi dùng kéo dài |\n\n"
        "| Rượu | Tăng độc tính trên gan |"
    )

    assert split_table_markdown(table, 40) == [
        "| Thuốc | Hậu quả |",
        "| Warfarin | Tăng INR khi dùng kéo dài |",
        "| Rượu | Tăng độc tính trên gan |",
    ]


def test_split_table_markdown_keeps_a_wide_row_whole() -> None:
    table = (
        "| Thuốc | Hậu quả |\n|---|---|\n"
        "| Warfarin | Tăng INR khi dùng kéo dài, cần theo dõi INR hằng tuần |\n"
        "| Rượu | Độc gan |"
    )

    assert split_table_markdown(table, 40) == [
        "| Thuốc | Hậu quả |\n|---|---|\n"
        "| Warfarin | Tăng INR khi dùng kéo dài, cần theo dõi INR hằng tuần |",
        "| Thuốc | Hậu quả |\n|---|---|\n| Rượu | Độc gan |",
    ]


def test_split_lines_keeps_lowercase_continuations_with_their_entry() -> None:
    text = (
        "N02BE01 Paracetamol\n"
        "M01AB05 Diclofenac, dùng đường uống\n"
        "hoặc đặt trực tràng\n"
        "M01AC01 Piroxicam"
    )

    assert split_lines_without_breaking_entries(text, 60) == [
        "N02BE01 Paracetamol",
        "M01AB05 Diclofenac, dùng đường uống\nhoặc đặt trực tràng",
        "M01AC01 Piroxicam",
    ]


def test_split_lines_drops_blank_lines_and_keeps_long_entries_whole() -> None:
    text = (
        "  - **Efferalgan**: Paracetamol   \n\n"
        "- **Panadol Extra**: Paracetamol, cafein, dùng giảm đau hạ sốt\n"
        "- **Voltaren**: Diclofenac"
    )

    assert split_lines_without_breaking_entries(text, 30) == [
        "- **Efferalgan**: Paracetamol",
        "- **Panadol Extra**: Paracetamol, cafein, dùng giảm đau hạ sốt",
        "- **Voltaren**: Diclofenac",
    ]


def test_split_lines_never_splits_a_run_of_continuation_lines() -> None:
    assert split_lines_without_breaking_entries(
        "dạng uống\nhoặc tiêm\nhoặc đặt trực tràng", 12
    ) == ["dạng uống\nhoặc tiêm\nhoặc đặt trực tràng"]
    assert split_lines_without_breaking_entries(" \n\n", 12) == []
