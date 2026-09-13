import pytest
from pharma_agent.domain.corpus.bundle import read_bundle

from seed_pipeline.bundle.parity import check_chunk_parity
from seed_pipeline.config.paths import DEFAULT_BUNDLE_DIR, MIGRATION_DIR
from seed_pipeline.evaluation.artifact_contracts import iter_jsonl_objects

pytestmark = pytest.mark.data


def test_backend_chunker_reproduces_the_pre_migration_build() -> None:
    report = check_chunk_parity(
        read_bundle(DEFAULT_BUNDLE_DIR),
        iter_jsonl_objects(MIGRATION_DIR / "rag-final-chunks.jsonl"),
    )

    assert report.chunks_checked > 0
    assert report.ok, report.to_dict()
