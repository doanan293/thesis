from pathlib import Path

import pytest

from seed_pipeline.config.paths import RUNS_DIR
from seed_pipeline.evaluation.run_workspace import load_run_record
from seed_pipeline.runtime.catalog import RERANKER_MODELS

TRACKED_RUN_RECORDS = sorted(RUNS_DIR.glob("*/run.json"))


@pytest.mark.parametrize(
    "record_path", TRACKED_RUN_RECORDS, ids=lambda path: path.parent.name
)
def test_tracked_runs_register_only_catalog_rerankers(record_path: Path) -> None:
    record = load_run_record(record_path)

    assert {variant.model for variant in record.rerank_variants.values()} <= set(
        RERANKER_MODELS
    )
