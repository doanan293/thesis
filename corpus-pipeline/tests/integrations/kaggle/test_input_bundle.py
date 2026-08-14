from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.models import InputBundle, InputFile


def _file(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_input_bundle_identity_is_order_independent_and_content_aware(tmp_path):
    candidates = InputFile.create(
        "candidates", _file(tmp_path / "candidates.jsonl", "{}\n")
    )
    manifest = InputFile.create(
        "candidate_manifest", _file(tmp_path / "manifest.json", "{}\n")
    )

    first = InputBundle.create((candidates, manifest))
    reordered = InputBundle.create((manifest, candidates))

    assert first.sha256 == reordered.sha256
    assert first.sha256 != candidates.sha256
    assert first.descriptors()["candidates"] == {
        "filename": "candidates.jsonl",
        "sha256": candidates.sha256,
    }


def test_input_bundle_identity_changes_when_manifest_content_changes(tmp_path):
    candidates = InputFile.create(
        "candidates", _file(tmp_path / "candidates.jsonl", "{}\n")
    )
    manifest_path = _file(tmp_path / "manifest.json", '{"version": 1}\n')
    before = InputBundle.create(
        (candidates, InputFile.create("candidate_manifest", manifest_path))
    )
    manifest_path.write_text('{"version": 2}\n', encoding="utf-8")
    after = InputBundle.create(
        (candidates, InputFile.create("candidate_manifest", manifest_path))
    )

    assert before.sha256 != after.sha256


def test_input_bundle_rejects_duplicate_logical_keys(tmp_path):
    first = InputFile.create("input", _file(tmp_path / "one.jsonl", "{}\n"))
    second = InputFile.create("input", _file(tmp_path / "two.jsonl", "{}\n"))

    with pytest.raises(ValueError, match="duplicate input key: input"):
        InputBundle.create((first, second))


def test_input_file_rejects_stale_direct_descriptor(tmp_path):
    source = _file(tmp_path / "input.jsonl", "{}\n")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        InputFile("input", source, "input.jsonl", "0" * 64)


def test_input_bundle_rejects_filename_collisions_and_reserved_manifest(tmp_path):
    first = InputFile.create("first", _file(tmp_path / "one.jsonl", "one\n"))
    second = InputFile.create(
        "second", _file(tmp_path / "two.jsonl", "two\n"), filename=first.filename
    )

    with pytest.raises(ValueError, match="duplicate mounted filename"):
        InputBundle.create((first, second))

    with pytest.raises(ValueError, match="reserved mounted filename"):
        InputFile.create(
            "input",
            _file(tmp_path / "input.jsonl", "{}\n"),
            filename="dependency_manifest.json",
        )


def test_input_file_rejects_explicit_empty_filename(tmp_path):
    with pytest.raises(ValueError, match="invalid mounted input filename"):
        InputFile.create("input", _file(tmp_path / "input.jsonl", "{}\n"), filename="")
