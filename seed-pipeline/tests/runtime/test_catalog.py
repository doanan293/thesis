from seed_pipeline.runtime.catalog import MODEL_CATALOG, require_model

VIMED = "qwen3-reranker:0.6b-fp16-vimed"


def test_vimed_reranker_is_the_0_6b_runtime_with_its_own_file() -> None:
    base = require_model("qwen3-reranker:0.6b-fp16")
    variant = require_model(VIMED)

    assert variant.name == VIMED
    assert variant.slug == "qwen3_reranker_0_6b_fp16_vimed"
    assert variant.canonical_filename == "qwen3-reranker-0.6b-f16-vimed.gguf"
    assert (variant.byte_size, variant.sha256) == (
        1_197_634_336,
        "fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851",
    )
    assert variant.kind is base.kind
    assert variant.topology is base.topology
    assert variant.rerank_contract == base.rerank_contract
    assert variant.rerank_search_space == base.rerank_search_space


def test_every_gguf_dataset_slug_fits_the_kaggle_limit() -> None:
    too_long = {
        name: spec.gguf_dataset_slug
        for name, spec in MODEL_CATALOG.items()
        if len(spec.gguf_dataset_slug) > 50
    }

    assert too_long == {}
