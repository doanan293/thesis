from pathlib import Path

from corpus_pipeline.evaluation.build_dataset import (
    EvaluationBuildRequest,
    EvaluationBuildResult,
    build_evaluation_dataset,
)


def test_build_evaluation_dataset_chains_patient_queries_into_eval(
    monkeypatch, tmp_path: Path
):
    observed_patient = []
    observed_eval = {}

    def fake_patient(sections, chunks, output, target):
        observed_patient.append((sections, chunks, output, target))
        output.write_text("[]\n", encoding="utf-8")
        return [{"query": "one"}]

    def fake_eval(**kwargs):
        observed_eval.update(kwargs)
        kwargs["output_path"].write_text("{}\n", encoding="utf-8")
        return [{"query_id": "q1"}, {"query_id": "q2"}]

    monkeypatch.setattr(
        "corpus_pipeline.evaluation.build_dataset.build_patient_queries", fake_patient
    )
    monkeypatch.setattr(
        "corpus_pipeline.evaluation.build_dataset.build_jsonl", fake_eval
    )

    result = build_evaluation_dataset(
        EvaluationBuildRequest(
            sections_path=tmp_path / "sections.jsonl",
            chunks_path=tmp_path / "chunks.jsonl",
            output_dir=tmp_path / "evaluation",
            patient_query_count=1,
            evaluation_row_count=2,
        )
    )

    assert isinstance(result, EvaluationBuildResult)
    assert observed_patient
    assert observed_eval["patient_queries_path"] == result.patient_queries_path
    assert result.patient_query_count == 1
    assert result.evaluation_row_count == 2
