from pathlib import Path

import pytest

from seed_pipeline.config.paths import RUNS_DIR
from seed_pipeline.evaluation.run_workspace import load_run_record
from seed_pipeline.runtime.catalog import RERANKER_MODELS

# Evidence of the rerank instruction decision (docs/guides/evaluation.md). Its variants
# name the retired experiment model and the original-template 0.6b file, and the run is
# never recomputed.
FROZEN_EXPERIMENT_RUNS = frozenset({"hybrid-qwen4b-p50-k30-rrf2-sample1000"})
TRACKED_RUN_RECORDS = sorted(
    path
    for path in RUNS_DIR.glob("*/run.json")
    if path.parent.name not in FROZEN_EXPERIMENT_RUNS
)


@pytest.mark.parametrize(
    "record_path", TRACKED_RUN_RECORDS, ids=lambda path: path.parent.name
)
def test_tracked_runs_register_only_catalog_rerankers(record_path: Path) -> None:
    record = load_run_record(record_path)

    assert {variant.model for variant in record.rerank_variants.values()} <= set(
        RERANKER_MODELS
    )


@pytest.mark.parametrize("run", sorted(FROZEN_EXPERIMENT_RUNS))
def test_frozen_experiment_runs_still_exist(run: str) -> None:
    assert (RUNS_DIR / run / "run.json").is_file()
