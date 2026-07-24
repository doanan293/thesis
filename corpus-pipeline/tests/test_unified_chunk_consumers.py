from pathlib import Path

from config.paths import RAG_FINAL_CHUNKS_PATH
from evaluation.build_section_retrieval_eval import chunk_body_text, chunk_identifier
from kaggle_vector_cache.orchestrator import KaggleVectorCacheOrchestrator
from vector_store.ingest_vectors import CANONICAL_METADATA_PATH


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

    orchestrator = object.__new__(KaggleVectorCacheOrchestrator)
    orchestrator.project_root = tmp_path
    assert orchestrator.corpus_path == (
        tmp_path / "data/processed/rag-final/chunks.jsonl"
    )
