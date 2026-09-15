import pytest

from seed_pipeline.corpus.processing.clean_markdown_corpus import (
    clean_markdown_text,
    repair_split_syllables,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("đ ường uống", "đường uống"),
        ("Dư ợc thư quốc gia", "Dược thư quốc gia"),
        ("theo D ược thư quốc gia", "theo Dược thư quốc gia"),
        ("H ướng dẫn sử dụng", "Hướng dẫn sử dụng"),
        ("T hường gặp", "Thường gặp"),
        ("người lớn và ng ười cao tuổi", "người lớn và người cao tuổi"),
        ("uống trư ớc bữa ăn", "uống trước bữa ăn"),
        ("tăng c ường tác dụng", "tăng cường tác dụng"),
    ],
)
def test_split_syllables_are_joined(text: str, expected: str) -> None:
    assert repair_split_syllables(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Words starting with an uppercase accented letter.
        "Ít hơn 1% liều dùng thải trừ qua phân",
        "Ám ảnh sợ xã hội",
        "Ăn ít hẳn nhưng vẫn ăn thành bữa",
        "LIỀU DÙNG CHO NGƯỜI LỚN",
        # A letter that starts a symbol, unit or gene name.
        "các thuốc có t1/2 dài",
        "ví dụ t1/2 của ceftriaxon",
        "Đơn vị của BSA là m2",
        "diện tích bề mặt thân thể tính ra m2.",
        "bệnh nhân có c-Kit dương tính",
        # A single uppercase letter naming a vitamin, protein, cell or wave.
        "chế phẩm chứa vitamin C uống có thể gây ăn mòn men răng",
        "liều vitamin D uống hằng ngày",
        "thiếu protein C hay đồng yếu tố là protein S",
        "lympho bào T hoạt hóa",
        "các sóng R, T hạ thấp",
        # Two separate words.
        "chỉ được sử dụng trong thú y.",
    ],
)
def test_separate_words_are_not_joined(text: str) -> None:
    assert repair_split_syllables(text) == text


def test_explicit_vitamin_c_repair_is_not_undone() -> None:
    assert "vitamin C uống có thể" in clean_markdown_text(
        "các chế phẩm chứa vitamin Cuống có thể gây ăn mòn"
    )
