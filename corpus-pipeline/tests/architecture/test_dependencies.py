import ast
from pathlib import Path


def test_cli_does_not_leak_into_domain_modules():
    offenders: list[tuple[str, str]] = []
    for path in Path("src").rglob("*.py"):
        if "/cli/" in path.as_posix():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            else:
                continue
            if any(
                module == "corpus_pipeline.cli"
                or module.startswith("corpus_pipeline.cli.")
                for module in modules
            ):
                offenders.extend((str(path), module or "") for module in modules)
    assert offenders == []
