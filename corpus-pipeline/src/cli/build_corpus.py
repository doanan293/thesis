from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pipeline.build_corpus import default_config, run_build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and publish the RAG corpus.")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--curated-tables", type=Path)
    parser.add_argument("--table-overrides", type=Path)
    parser.add_argument("--mappings", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--final-dir", type=Path)
    parser.add_argument("--max-chars", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = default_config()
    overrides = {
        "pdf_path": args.pdf,
        "snapshot_archive": args.snapshot_archive,
        "snapshot_manifest": args.snapshot_manifest,
        "curated_tables_path": args.curated_tables,
        "table_overrides_path": args.table_overrides,
        "mappings_path": args.mappings,
        "glossary_path": args.glossary,
        "work_root": args.work_root,
        "final_dir": args.final_dir,
        "max_chars": args.max_chars,
    }
    config = replace(
        config, **{key: value for key, value in overrides.items() if value is not None}
    )
    result = run_build(config)
    print(f"build_id={result.build_id}")
    print(f"published={result.final_dir}")
    print("workspace_removed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
