from typer.testing import CliRunner

from corpus_pipeline.cli.app import app
from corpus_pipeline.cli.commands.retrieve import resolve_query_embeddings_dir
from corpus_pipeline.config.paths import query_embedding_bundle_dir
from corpus_pipeline.evaluation.artifact_contracts import sha256_file


def test_retrieve_help_exposes_named_run():
    result = CliRunner().invoke(app, ["retrieve", "--help"])
    assert result.exit_code == 0
    assert "--run" in result.stdout
    assert "--qdrant-url" in result.stdout


def test_dense_retrieval_resolves_canonical_query_bundle(tmp_path):
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text('{"query_id":"q1","query":"query"}\n', encoding="utf-8")
    expected = query_embedding_bundle_dir(
        "qwen3-embedding:0.6b-fp16", sha256_file(evaluation)
    )
    assert (
        resolve_query_embeddings_dir(
            evaluation,
            "qwen3-embedding:0.6b-fp16",
            "hybrid",
            None,
        )
        == expected
    )
