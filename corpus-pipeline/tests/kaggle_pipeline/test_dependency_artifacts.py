from types import SimpleNamespace

from corpus_pipeline.integrations.kaggle.model_artifacts import (
    resolve_publishable_artifact,
    stage_model_dataset,
)


def test_model_artifact_validates_size_and_checksum(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"model")
    spec = SimpleNamespace(
        name="fake:embed",
        canonical_filename="model.gguf",
        byte_size=5,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        gguf_dataset_slug="fake-model",
        vector_dimension=3,
        kind=SimpleNamespace(value="embedding"),
        reranker_protocol=None,
    )
    spec.sha256 = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "corpus_pipeline.integrations.kaggle.model_artifacts.require_model",
        lambda name: spec,
    )
    artifact = resolve_publishable_artifact(tmp_path, "fake:embed")
    assert artifact.sha256 == spec.sha256


def test_stage_model_dataset_writes_manifest_and_file(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"model")
    spec = SimpleNamespace(
        name="fake:embed",
        canonical_filename="model.gguf",
        byte_size=5,
        sha256=__import__("hashlib").sha256(b"model").hexdigest(),
        gguf_dataset_slug="fake-model",
        vector_dimension=3,
        kind=SimpleNamespace(value="embedding"),
        reranker_protocol=None,
    )
    monkeypatch.setattr(
        "corpus_pipeline.integrations.kaggle.model_artifacts.require_model",
        lambda name: spec,
    )
    artifact = resolve_publishable_artifact(tmp_path, "fake:embed")
    target = stage_model_dataset(artifact, tmp_path / "staging", "owner")
    assert (target / "model_manifest.json").is_file()
    assert (target / "model.gguf").is_symlink()
