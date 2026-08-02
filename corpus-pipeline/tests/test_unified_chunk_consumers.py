from pathlib import Path

from corpus_pipeline.config.paths import RAG_FINAL_CHUNKS_PATH
from corpus_pipeline.evaluation.build_section_retrieval_eval import (
    chunk_body_text,
    chunk_identifier,
)
from corpus_pipeline.integrations.kaggle.models import StageName
from corpus_pipeline.integrations.kaggle.stages import get_stage_adapter
from corpus_pipeline.vector_store.ingest_vectors import CANONICAL_METADATA_PATH


def test_consumers_use_unified_chunks(tmp_path: Path) -> None:
    chunk = {
        "chunk_id": "c1",
        "section_id": "s1",
        "chunk_text": "visible",
        "embedding_text": "searchable",
    }
    assert chunk_identifier(chunk) == "c1"
    assert chunk_body_text(chunk) == "visible"
    assert Path(CANONICAL_METADATA_PATH) == RAG_FINAL_CHUNKS_PATH

    assert get_stage_adapter(StageName.CORPUS_EMBED).name is StageName.CORPUS_EMBED
