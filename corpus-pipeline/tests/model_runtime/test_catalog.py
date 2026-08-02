from corpus_pipeline.runtime.catalog import require_model


def test_reranker_models_have_custom_runtime_parameters():
    spec_06b = require_model("qwen3-reranker:0.6b-fp16")
    assert spec_06b.kaggle_parallel == 4
    assert spec_06b.kaggle_request_batch_size == 16
    assert spec_06b.kaggle_context_per_slot == 4096

    spec_4b = require_model("qwen3-reranker:4b-fp16")
    assert spec_4b.kaggle_parallel == 2
    assert spec_4b.kaggle_request_batch_size == 8
    assert spec_4b.kaggle_context_per_slot == 4096

    spec_8b = require_model("qwen3-reranker:8b-fp16")
    assert spec_8b.kaggle_parallel == 2
    assert spec_8b.kaggle_request_batch_size == 4
    assert spec_8b.kaggle_context_per_slot == 4096

    spec_bge_m3 = require_model("bge-reranker-v2-m3:f16")
    assert spec_bge_m3.kaggle_parallel == 2
    assert spec_bge_m3.kaggle_request_batch_size == 16
    assert spec_bge_m3.kaggle_context_per_slot == 4096

    spec_bge_gemma = require_model("bge-reranker-v2-gemma:f16")
    assert spec_bge_gemma.kaggle_parallel == 2
    assert spec_bge_gemma.kaggle_request_batch_size == 8
    assert spec_bge_gemma.kaggle_context_per_slot == 4096
