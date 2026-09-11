import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "pharma_agent"
FORBIDDEN_IN_DOMAIN = {
    "langgraph",
    "langchain",
    "langchain_core",
    "openai",
    "qdrant_client",
    "sqlalchemy",
    "fastapi",
    "langfuse",
    "httpx",
    "fastembed",
}
OUTER_LAYERS = {
    "pharma_agent.application",
    "pharma_agent.infrastructure",
    "pharma_agent.api",
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _py_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.py")) if folder.exists() else []


def test_domain_is_framework_free() -> None:
    for file in _py_files(SRC / "domain"):
        for name in _imports(file):
            root = name.split(".")[0]
            assert root not in FORBIDDEN_IN_DOMAIN, (
                f"{file.relative_to(SRC)} imports {name}"
            )


def test_domain_does_not_import_outer_layers() -> None:
    for file in _py_files(SRC / "domain"):
        for name in _imports(file):
            assert not any(name.startswith(layer) for layer in OUTER_LAYERS), (
                f"{file.relative_to(SRC)} imports outer layer {name}"
            )


def test_api_does_not_import_domain_directly() -> None:
    for file in _py_files(SRC / "api"):
        for name in _imports(file):
            assert not name.startswith("pharma_agent.domain"), (
                f"{file.relative_to(SRC)} must go through application, not domain: {name}"
            )
