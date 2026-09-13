from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

CONTRACT_FILES = frozenset(
    {
        "sections.jsonl",
        "blocks.jsonl",
        "chunks.jsonl",
        "manifest.json",
        "validation_report.json",
    }
)
TRACKED_FILES = (
    "sections.jsonl",
    "blocks.jsonl",
    "chunks.jsonl",
    "validation_report.json",
)


class ContractError(RuntimeError):
    """Raised when a source or final contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_materialized_file(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"Required file is missing: {path}")
    prefix = path.read_bytes()[:128]
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ContractError(f"{path} is a Git LFS pointer; run `git lfs pull`")


def require_materialized_pdf(path: Path) -> None:
    require_materialized_file(path)
    if not Path(path).read_bytes()[:128].startswith(b"%PDF-"):
        raise ContractError(f"{path} does not have a PDF signature")


def require_workspace_capacity(path: Path, *, required_bytes: int) -> None:
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        raise ContractError(
            f"Insufficient free disk space: need {required_bytes} bytes, "
            f"have {usage.free} bytes"
        )


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    require_materialized_file(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ContractError(f"JSONL row is not an object at {path}:{line_number}")
        records.append(record)
    return records


def build_manifest(
    final_dir: Path,
    *,
    build_id: str,
    source_pdf_sha256: str,
    snapshot_id: str,
    snapshot_sha256: str,
    curated_input_digests: dict[str, str],
    config_digest: str,
) -> dict[str, Any]:
    final_dir = Path(final_dir)
    sections = _jsonl_records(final_dir / "sections.jsonl")
    blocks = _jsonl_records(final_dir / "blocks.jsonl")
    chunks = _jsonl_records(final_dir / "chunks.jsonl")
    report = json.loads((final_dir / "validation_report.json").read_text("utf-8"))
    tracked_files = {
        name: {
            "sha256": sha256_file(final_dir / name),
            "bytes": (final_dir / name).stat().st_size,
        }
        for name in TRACKED_FILES
    }
    return {
        "schema_version": "rag-final-v2",
        "build_id": build_id,
        "source_pdf_sha256": source_pdf_sha256,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "curated_input_digests": dict(sorted(curated_input_digests.items())),
        "config_digest": config_digest,
        "section_count": len(sections),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "validation_ok": bool(report.get("ok")),
        "files": tracked_files,
    }


def validate_contract_directory(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        raise ContractError(f"Final contract directory is missing: {path}")
    actual = {entry.name for entry in path.iterdir()}
    if actual != CONTRACT_FILES:
        raise ContractError(
            f"Final contract must contain exactly {sorted(CONTRACT_FILES)}; "
            f"found {sorted(actual)}"
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rag-final-v2":
        raise ContractError("Unsupported final contract schema")
    report = json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("ok") is not True or manifest.get("validation_ok") is not True:
        raise ContractError("Final contract validation report is not successful")
    sections = _jsonl_records(path / "sections.jsonl")
    blocks = _jsonl_records(path / "blocks.jsonl")
    chunks = _jsonl_records(path / "chunks.jsonl")
    if not sections or not blocks or not chunks:
        raise ContractError("Final contract JSONL files must not be empty")
    if any(not record.get("id") or not record.get("text") for record in sections):
        raise ContractError("Every section requires id and text")
    if any(
        not record.get("block_id") or not record.get("section_id") for record in blocks
    ):
        raise ContractError("Every block requires block_id and section_id")
    if any(
        not record.get("chunk_id")
        or not record.get("section_id")
        or not record.get("chunk_text")
        or not record.get("embedding_text")
        for record in chunks
    ):
        raise ContractError(
            "Every unified chunk requires chunk_id, section_id, chunk_text, and embedding_text"
        )
    section_ids = {record["id"] for record in sections}
    if len(section_ids) != len(sections):
        raise ContractError("Final sections contain duplicate IDs")
    if len({record["block_id"] for record in blocks}) != len(blocks):
        raise ContractError("Final blocks contain duplicate IDs")
    orphans = sorted({str(record["section_id"]) for record in blocks} - section_ids)
    if orphans:
        raise ContractError(f"Final blocks reference unknown sections: {orphans[:5]}")
    if len({record["chunk_id"] for record in chunks}) != len(chunks):
        raise ContractError("Final chunks contain duplicate IDs")
    for name in TRACKED_FILES:
        expected = manifest.get("files", {}).get(name, {})
        file_path = path / name
        if expected.get("sha256") != sha256_file(file_path):
            raise ContractError(f"Final contract checksum mismatch: {name}")
        if int(expected.get("bytes", -1)) != file_path.stat().st_size:
            raise ContractError(f"Final contract size mismatch: {name}")
    if int(manifest.get("section_count", -1)) != len(sections):
        raise ContractError("Final section count mismatch")
    if int(manifest.get("block_count", -1)) != len(blocks):
        raise ContractError("Final block count mismatch")
    if int(manifest.get("chunk_count", -1)) != len(chunks):
        raise ContractError("Final chunk count mismatch")
    return manifest
