from pharma_agent.domain.skill.slug import slugify


def test_slugify_folds_vietnamese_and_punctuation() -> None:
    assert slugify("Tương tác thuốc & rượu (bia)") == "tuong-tac-thuoc-ruou-bia"
    assert slugify("  Liều  Đặc Biệt  ") == "lieu-dac-biet"
    assert slugify("!!!") == "skill"
    assert len(slugify("x" * 200)) <= 60
