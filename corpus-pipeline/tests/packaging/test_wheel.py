from pathlib import Path
from zipfile import ZipFile


def test_built_wheel_contains_only_canonical_namespace():
    wheels = sorted(Path("dist").glob("*.whl"))
    assert wheels, "run uv build before packaging tests"
    with ZipFile(wheels[-1]) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    payload = [name for name in names if ".dist-info/" not in name]
    assert payload
    assert all(name.startswith("corpus_pipeline/") for name in payload)
    assert "corpus_pipeline/integrations/postgres/schema/rag_app_schema.sql" in names
    assert not any(name.startswith("evaluation/") for name in names)
