import ast
import tomllib
from pathlib import Path

LEGACY_ROOTS = {
    "artifact_lifecycle",
    "canonical",
    "config",
    "corpus",
    "crawler",
    "evaluation",
    "kaggle_infra",
    "kaggle_pipeline",
    "model_runtime",
    "pipeline",
    "postgres_store",
    "rag_metadata",
    "tables",
    "validation",
    "vector_store",
}


def test_all_python_sources_use_one_namespace():
    offenders = [
        path
        for path in Path("src").rglob("*.py")
        if Path("src/corpus_pipeline") not in path.parents
    ]
    assert offenders == []


def test_source_does_not_import_former_top_level_namespaces():
    offenders: list[tuple[str, str]] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module.split(".", 1)[0]] if node.module else []
            else:
                continue
            for name in names:
                if name in LEGACY_ROOTS:
                    offenders.append((str(path), name))
    assert offenders == []


def test_wheel_contains_one_package_root():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/corpus_pipeline"
    ]
