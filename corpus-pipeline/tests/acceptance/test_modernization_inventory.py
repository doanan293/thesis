import ast
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
DEAD_SYMBOLS = {
    "NoopReranker",
    "PrecomputedJsonRetriever",
    "PrecomputedJsonlRetriever",
    "runtime_artifacts",
    "qdrant_hydration",
}


def test_former_import_roots_are_absent_from_application_code():
    offenders: list[tuple[str, str]] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            else:
                names = []
            offenders.extend(
                (str(path), name) for name in names if name in LEGACY_ROOTS
            )
    assert offenders == []


def test_dead_symbols_and_modules_are_absent():
    offenders: list[tuple[str, str]] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in DEAD_SYMBOLS:
                    offenders.append((str(path), node.name))
            if path.name == "ingest_vectors.py" and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "Document",
                        "PointStruct",
                        "QdrantClientHelper",
                    }:
                        offenders.append((str(path), target.id))
    offenders.extend(
        (str(path), path.stem)
        for path in Path("src").rglob("*.py")
        if path.stem in DEAD_SYMBOLS
    )
    assert offenders == []


def test_no_test_gate_is_skipped_or_expected_failure():
    offenders = []
    for path in Path("tests").rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if "pytest.mark.xfail" in text or "pytest.mark.skip" in text:
            offenders.append(str(path))
    assert offenders == []
