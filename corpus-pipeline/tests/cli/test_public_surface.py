import ast
from pathlib import Path


def test_guides_document_corpus_entrypoint_only():
    paths = [
        Path("README.md"),
        Path("docs/guides/workflow-local-only.md"),
        Path("docs/guides/workflow-local-kaggle.md"),
        Path("docs/guides/downstream.md"),
        Path("docs/guides/cli-reference.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "python -m cli." not in text
    assert "python -m evaluation." not in text
    assert "uv run corpus" in text


def test_legacy_cli_package_is_absent():
    assert not Path("src/cli").exists()
    assert not Path("src/evaluation/run_retrieval_eval.py").exists()


def test_domain_modules_do_not_define_legacy_parsers_or_main():
    allowed = {
        Path("src/corpus_pipeline/cli/app.py"),
        Path("src/corpus_pipeline/integrations/kaggle/workers/corpus_embed.py"),
        Path("src/corpus_pipeline/integrations/kaggle/workers/query_embed.py"),
        Path("src/corpus_pipeline/integrations/kaggle/workers/rerank.py"),
    }
    legacy = []
    for path in Path("src").rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        forbidden = names & {"main", "parse_args", "build_parser", "build_arg_parser"}
        if forbidden:
            legacy.append((str(path), sorted(forbidden)))
    assert legacy == []
