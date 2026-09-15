import pytest

from seed_pipeline.runtime.catalog import MODEL_CATALOG, require_model

VIMED_FILES = {
    "qwen3-reranker:0.6b-fp16": (
        "qwen3-reranker-0.6b-f16-vimed.gguf",
        1_197_634_336,
        "fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851",
    ),
    "qwen3-reranker:4b-fp16": (
        "qwen3-reranker-4b-f16-vimed.gguf",
        8_049_922_912,
        "4b428e981efc9a209c674a0af9f1ec527b5570896e0bcc431382c16f1e839c7a",
    ),
    "qwen3-reranker:8b-fp16": (
        "qwen3-reranker-8b-f16-vimed.gguf",
        15_141_207_776,
        "c6516333e32d4f1d8aad8325d5d5bdb8165f9778e892eed4ca8b0eaf3339a810",
    ),
}


@pytest.mark.parametrize(("model", "expected"), sorted(VIMED_FILES.items()))
def test_qwen3_rerankers_serve_the_vietnamese_medical_template(
    model: str, expected: tuple[str, int, str]
) -> None:
    spec = require_model(model)

    assert (spec.canonical_filename, spec.byte_size, spec.sha256) == expected


def test_the_instruction_experiment_model_is_gone() -> None:
    assert "qwen3-reranker:0.6b-fp16-vimed" not in MODEL_CATALOG


def test_every_gguf_dataset_slug_fits_the_kaggle_limit() -> None:
    too_long = {
        name: spec.gguf_dataset_slug
        for name, spec in MODEL_CATALOG.items()
        if len(spec.gguf_dataset_slug) > 50
    }

    assert too_long == {}
