from seed_pipeline.config import paths


def test_data_layout_has_one_heavy_root_and_split_retrieval_roots():
    assert paths.HEAVY_DATA_DIR == paths.DATA_DIR / "heavy"
    assert paths.RESOURCES_DIR == paths.DATA_DIR / "resources"
    assert paths.MANIFESTS_DIR == paths.DATA_DIR / "manifests"
    assert paths.RAW_DIR == paths.HEAVY_DATA_DIR / "raw"
    assert paths.PROCESSED_DIR == paths.HEAVY_DATA_DIR / "processed"
    assert paths.DATA_CACHE_DIR == paths.HEAVY_DATA_DIR / "cache"
    assert paths.WORK_DIR == paths.HEAVY_DATA_DIR / ".work"
    assert paths.INTERIM_DIR == paths.WORK_DIR / "manual"
    assert paths.RUNTIME_PROFILE_DIR == (
        paths.HEAVY_DATA_DIR / "runtime_kaggle_profiles"
    )
    assert paths.retrieval_run_roots("trial") == (
        paths.DATA_DIR / "retrieval_eval" / "trial",
        paths.HEAVY_DATA_DIR / "retrieval_eval" / "trial",
    )
