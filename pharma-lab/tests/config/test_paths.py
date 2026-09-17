from pharma_lab.config import paths


def test_evaluation_cache_corpus_and_work_folders_follow_the_layout() -> None:
    assert paths.RAG_FINAL_DIR == paths.DATA_DIR / "corpus" / "rag-final"
    assert paths.RAG_FINAL_SECTIONS_PATH == paths.RAG_FINAL_DIR / "sections.jsonl"
    assert paths.BUNDLE_DIR == paths.DATA_DIR / "corpus" / "formulary"
    assert paths.GOLD_DIR == paths.DATA_DIR / "evaluation" / "gold"
    assert paths.RUNS_DIR == paths.DATA_DIR / "evaluation" / "runs"
    assert paths.run_dir("trial") == paths.RUNS_DIR / "trial"
    assert paths.TEXT_EMBEDDING_CACHE_DIR == paths.CACHE_DIR / "text_embeddings"
    assert paths.QUERY_EMBEDDING_CACHE_DIR == paths.CACHE_DIR / "query_embeddings"
    assert paths.RERANK_SCORE_CACHE_DIR == paths.CACHE_DIR / "rerank_scores"
    assert paths.KAGGLE_PROFILE_DIR == paths.CACHE_DIR / "kaggle_profiles"
    assert paths.LOCAL_PROFILE_DIR == paths.CACHE_DIR / "local_profiles"
    assert paths.CACHE_DIR == paths.DATA_DIR / "cache"
    assert paths.WORK_DIR == paths.DATA_DIR / "work"
    assert paths.BUILD_WORK_DIR == paths.WORK_DIR / "build"
    assert paths.LOCK_DIR == paths.WORK_DIR / "locks"
    assert paths.TEXT_INTERIM_DIR == paths.BUILD_WORK_DIR / "text"
    assert paths.BUNDLE_EMBED_WORK_DIR == paths.WORK_DIR / "bundle-embed"
    assert paths.EVALUATION_CHUNKS_PATH == (
        paths.WORK_DIR / "evaluation-chunks" / "chunks.jsonl"
    )


def test_build_inputs_live_in_sources() -> None:
    assert paths.SOURCES_DIR == paths.DATA_DIR / "sources"
    assert paths.FORMULARY_PDF_PATH == (
        paths.SOURCES_DIR / "duoc-thu-quoc-gia-viet-nam.pdf"
    )
    assert paths.LEAFLETS_DIR == paths.SOURCES_DIR / "leaflets"
    assert paths.LEAFLETS_MANIFEST_PATH == paths.LEAFLETS_DIR / "manifest.json"
    assert paths.SOURCES_CURATION_DIR == paths.SOURCES_DIR / "curation"


def test_rerank_logs_live_under_work(isolated_logs_dir) -> None:
    # tests/conftest.py redirects LOGS_DIR for every test; the original is the real value.
    assert isolated_logs_dir.original == paths.WORK_DIR / "logs"
    assert paths.rerank_log_path("qwen3-reranker:0.6b-fp16") == (
        isolated_logs_dir.root / "rerank" / "qwen3_reranker_0_6b_fp16.log"
    )


def test_source_root_is_the_imported_package_tree() -> None:
    import pharma_lab

    assert paths.Path(pharma_lab.__file__).resolve().parent.parent == paths.SOURCE_ROOT
    assert (paths.SOURCE_ROOT / "pharma_lab" / "integrations" / "kaggle").is_dir()
