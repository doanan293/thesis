from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_lifecycle.snapshot import (
    SnapshotError,
    extract_snapshot,
    pack_snapshot,
    verify_snapshot,
)


def _manifest_archive(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path.parent / str(manifest["archive_name"])


def _snapshot_pairs(snapshot_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for manifest in sorted(Path(snapshot_dir).glob("*.manifest.json")):
        archive = _manifest_archive(manifest)
        if not archive.is_file():
            raise SnapshotError(f"Snapshot archive is missing: {archive}")
        pairs.append((archive, manifest))
    if not pairs:
        raise SnapshotError(f"No snapshots found in {snapshot_dir}")
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage immutable An Khang snapshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--urls", type=Path)
    pack.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)

    verify_all = subparsers.add_parser("verify-all")
    verify_all.add_argument("--snapshot-dir", type=Path, required=True)

    extract_latest = subparsers.add_parser("extract-latest")
    extract_latest.add_argument("--snapshot-dir", type=Path, required=True)
    extract_latest.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pack":
            result = pack_snapshot(args.source, args.output_dir, urls_path=args.urls)
            print(result.archive_path)
            print(result.manifest_path)
            print(f"snapshot_id={result.snapshot_id}")
            print(f"archive_sha256={result.archive_sha256}")
            print(f"file_count={result.file_count}")
        elif args.command == "verify":
            manifest = verify_snapshot(args.archive, args.manifest)
            print(f"verified snapshot_id={manifest['snapshot_id']}")
        elif args.command == "extract":
            extract_snapshot(args.archive, args.manifest, args.output_dir)
            print(f"extracted={args.output_dir}")
        elif args.command == "verify-all":
            file_count = 0
            for archive, manifest in _snapshot_pairs(args.snapshot_dir):
                verified = verify_snapshot(archive, manifest)
                file_count += int(verified["file_count"])
            print(f"verified_snapshots={len(_snapshot_pairs(args.snapshot_dir))}")
            print(f"verified_files={file_count}")
        else:
            archive, manifest = _snapshot_pairs(args.snapshot_dir)[-1]
            extract_snapshot(archive, manifest, args.output_dir)
            print(f"extracted={args.output_dir}")
    except SnapshotError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
