from corpus_pipeline.config.paths import (
    chunk_embedding_bundle_dir,
    query_embedding_bundle_dir,
    resolve_workspace_root,
)


def test_workspace_root_is_checkout_root():
    root = resolve_workspace_root()
    assert (root / "pyproject.toml").is_file()
    assert root.name == "corpus-pipeline"


def test_embedding_bundle_paths_are_model_and_fingerprint_scoped():
    chunk_path = chunk_embedding_bundle_dir("qwen3-embedding:0.6b-fp16", "corpus-a")
    query_path = query_embedding_bundle_dir("qwen3-embedding:0.6b-fp16", "eval-a")

    assert chunk_path.name == "corpus-a"
    assert query_path.name == "eval-a"
    assert chunk_path != chunk_embedding_bundle_dir(
        "qwen3-embedding:0.6b-fp16", "corpus-b"
    )
    assert query_path != query_embedding_bundle_dir("bge-m3:567m-fp16", "eval-a")
