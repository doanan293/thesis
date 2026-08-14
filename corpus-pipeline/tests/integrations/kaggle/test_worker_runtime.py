from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.artifacts import sha256_file
from corpus_pipeline.integrations.kaggle.workers.runtime import resolve_input_file


def _config(key: str, expected_path: Path, source: Path) -> dict:
    return {
        "input_files": {
            key: {
                "path": str(expected_path),
                "filename": source.name,
                "sha256": sha256_file(source),
            }
        }
    }


def test_resolve_input_file_accepts_exact_path_with_file_checksum(tmp_path):
    mounted = tmp_path / "mounted" / "candidates.jsonl"
    mounted.parent.mkdir()
    mounted.write_text("{}\n", encoding="utf-8")

    assert (
        resolve_input_file(
            _config("candidates", mounted, mounted),
            "candidates",
            input_root=tmp_path,
        )
        == mounted
    )


def test_resolve_input_file_fallback_selects_checksum_match(tmp_path):
    input_root = tmp_path / "input"
    expected = tmp_path / "source" / "candidates.jsonl"
    expected.parent.mkdir()
    expected.write_text("correct\n", encoding="utf-8")
    wrong = input_root / "first" / "candidates.jsonl"
    right = input_root / "second" / "candidates.jsonl"
    wrong.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    wrong.write_text("wrong\n", encoding="utf-8")
    right.write_text("correct\n", encoding="utf-8")
    config = _config("candidates", input_root / "missing" / expected.name, expected)

    assert resolve_input_file(config, "candidates", input_root=input_root) == right


def test_resolve_input_file_reports_checksum_mismatch(tmp_path):
    input_root = tmp_path / "input"
    mounted = input_root / "mounted" / "candidates.jsonl"
    mounted.parent.mkdir(parents=True)
    mounted.write_text("wrong\n", encoding="utf-8")
    expected = tmp_path / "expected" / "candidates.jsonl"
    expected.parent.mkdir()
    expected.write_text("correct\n", encoding="utf-8")
    config = _config("candidates", mounted, expected)

    with pytest.raises(RuntimeError, match=r"input 'candidates'.*checksum mismatch"):
        resolve_input_file(config, "candidates", input_root=input_root)


def test_resolve_input_file_ignores_composite_identity_hash(tmp_path):
    mounted = tmp_path / "mounted" / "candidates.jsonl"
    mounted.parent.mkdir()
    mounted.write_text("{}\n", encoding="utf-8")
    config = _config("candidates", mounted, mounted)
    config["identity"] = {"input_sha256": "f" * 64}

    assert resolve_input_file(config, "candidates", input_root=tmp_path) == mounted


def test_resolve_input_file_does_not_accept_exact_path_with_wrong_filename(tmp_path):
    mounted = tmp_path / "mounted" / "wrong-name.jsonl"
    mounted.parent.mkdir()
    mounted.write_text("{}\n", encoding="utf-8")
    expected = tmp_path / "expected" / "candidates.jsonl"
    expected.parent.mkdir()
    expected.write_text("{}\n", encoding="utf-8")
    config = _config("candidates", mounted, expected)

    with pytest.raises(RuntimeError, match="no mounted file found"):
        resolve_input_file(config, "candidates", input_root=tmp_path / "mounted")
