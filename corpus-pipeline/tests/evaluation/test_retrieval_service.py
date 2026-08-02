import json

import pytest

from corpus_pipeline.evaluation.dump_retrieval_candidates import load_query_rows
from corpus_pipeline.evaluation.retrieval_service import RetrieveRequest


def test_retrieve_request_has_shared_fields(tmp_path):
    request = RetrieveRequest(
        tmp_path / "eval.jsonl",
        None,
        tmp_path / "run",
        "model",
        "http://localhost:6333",
        "bm25",
        50,
        60,
        None,
        False,
    )
    assert request.candidate_k == 50


def test_load_query_rows_applies_limit(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "".join(
            json.dumps({"query_id": f"q{i}", "query": f"query {i}"}) + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )
    assert [row["query_id"] for row in load_query_rows(path, 2)] == ["q0", "q1"]


def test_load_query_rows_reports_missing_input(tmp_path):
    with pytest.raises(ValueError, match="Evaluation JSONL is missing"):
        load_query_rows(tmp_path / "missing.jsonl")
