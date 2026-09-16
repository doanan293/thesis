import json
from pathlib import Path

from pharma_agent.domain.corpus.bundle import read_bundle

from pharma_lab.bundle.evaluation_chunks import (
    evaluation_chunk_rows,
    write_evaluation_chunks,
)
from pharma_lab.bundle.export import ExportRequest, export_bundle

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
FIELDS_READ_BY_THE_BUILDERS = (
    "chunk_id",
    "section_id",
    "chunk_index",
    "title",
    "section",
    "source",
    "content_type",
    "context_header",
    "context_path",
    "chunk_text",
    "embedding_text",
    "hydrate_strategy",
    "chunk_role",
    "chunk_content_type",
    "term_annotations",
)


def _bundle_dir(tmp_path: Path) -> Path:
    export_bundle(
        ExportRequest(
            rag_final_dir=FIXTURE,
            glossary_path=FIXTURE / "term_glossary.json",
            mappings_path=FIXTURE / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    return tmp_path / "bundle"


def test_rows_reproduce_the_old_unified_chunk_fields(tmp_path: Path) -> None:
    rows = evaluation_chunk_rows(read_bundle(_bundle_dir(tmp_path)))
    old = [
        json.loads(line)
        for line in (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    ]

    by_id = {row["chunk_id"]: row for row in rows}
    assert sorted(by_id) == sorted(chunk["chunk_id"] for chunk in old)
    for chunk in old:
        row = by_id[chunk["chunk_id"]]
        for field in FIELDS_READ_BY_THE_BUILDERS:
            assert row[field] == chunk.get(field, []), (chunk["chunk_id"], field)
        assert row["table_id"] == chunk.get("table_id", "")


def test_write_evaluation_chunks_writes_one_row_per_chunk(tmp_path: Path) -> None:
    path = tmp_path / "work" / "chunks.jsonl"

    count = write_evaluation_chunks(read_bundle(_bundle_dir(tmp_path)), path)

    assert count == 6
    assert len(path.read_text("utf-8").splitlines()) == 6
