import unicodedata

import pytest

from pharma_agent.domain.shared.text import make_snippet, normalize_text


def test_normalize_text_unifies_unicode_line_endings_and_blank_edges() -> None:
    decomposed = unicodedata.normalize("NFD", "Liều dùng")
    raw = f"\n  \r\n{decomposed}  \r\nTrẻ em:\t\r\r\n  - 10 mg/kg  \n\n"

    assert normalize_text(raw) == "Liều dùng\nTrẻ em:\n\n  - 10 mg/kg"


def test_normalize_text_is_idempotent_and_empties_blank_input() -> None:
    text = "PARACETAMOL\n> Liều lượng và cách dùng"

    assert normalize_text(normalize_text(text)) == text
    assert normalize_text(" \n\t\n") == ""
    assert normalize_text("") == ""


def test_make_snippet_drops_table_pipes_and_rules() -> None:
    table = (
        "| Thuốc phối hợp | Hậu quả |\n"
        "| :--- | ---: |\n"
        "| Warfarin | Tăng INR khi dùng kéo dài |"
    )

    assert make_snippet(table, 300) == (
        "Thuốc phối hợp Hậu quả Warfarin Tăng INR khi dùng kéo dài"
    )


def test_make_snippet_cuts_at_word_boundary_and_appends_ellipsis() -> None:
    text = "Người lớn:   uống 500 mg - 1 g\nmỗi 4 - 6 giờ khi cần."

    snippet = make_snippet(text, 24)

    assert snippet == "Người lớn: uống 500 mg…"
    assert len(snippet) <= 24


def test_make_snippet_keeps_word_that_ends_exactly_at_the_cut() -> None:
    assert make_snippet("Paracetamol giảm đau hạ sốt", 12) == "Paracetamol…"


def test_make_snippet_hard_cuts_a_single_long_word() -> None:
    assert make_snippet("Acetylsalicylic", 8) == "Acetyls…"


def test_make_snippet_returns_short_text_unchanged() -> None:
    assert make_snippet("  Hạ sốt  ", 7) == "Hạ sốt"


def test_make_snippet_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        make_snippet("Hạ sốt", 0)
