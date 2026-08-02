from corpus_pipeline.cli.options import Backend, positive_int


def test_backend_values_are_stable():
    assert [item.value for item in Backend] == ["local", "kaggle"]


def test_positive_int_rejects_zero():
    try:
        positive_int("0")
    except ValueError as exc:
        assert str(exc) == "value must be >= 1"
    else:
        raise AssertionError("positive_int accepted zero")
