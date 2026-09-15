from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from seed_pipeline.artifacts.contract import (
    build_manifest,
    require_materialized_file,
    require_materialized_pdf,
    require_workspace_capacity,
    sha256_file,
    validate_contract_directory,
)
from seed_pipeline.artifacts.paths import ArtifactPaths, retain_failed_workspace
from seed_pipeline.artifacts.publisher import publish_contract
from seed_pipeline.config.chunking import DEFAULT_CHUNK_MAX_CHARS
from seed_pipeline.config.paths import (
    BUILD_WORK_DIR,
    FORMULARY_PDF_PATH,
    LEAFLETS_DIR,
    RAG_FINAL_DIR,
    SOURCES_CURATION_DIR,
    SOURCES_DIR,
)
from seed_pipeline.corpus.canonical.build_canonical_rag import process_canonical_rag
from seed_pipeline.corpus.crawling.integrate_leaflets import integrate_leaflet_corpus
from seed_pipeline.corpus.crawling.parse_html import parse_html_tree
from seed_pipeline.corpus.processing.clean_markdown_corpus import (
    process_corpus as clean_corpus,
)
from seed_pipeline.corpus.processing.extract_pymupdf_text import process_pdf
from seed_pipeline.corpus.processing.preprocess_rag_corpus import (
    process_corpus as preprocess_corpus,
)
from seed_pipeline.corpus.sources.leaflet_source import verify_leaflet_source
from seed_pipeline.corpus.validation.validate_final_rag import (
    combine_validation_reports,
    read_jsonl,
    run_deep_audit,
    validate_final_rag,
)

# Part of the build id, so a release whose code changes the output rebuilds the corpus.
PIPELINE_VERSION = version("seed-pipeline")


@dataclass(frozen=True)
class BuildConfig:
    pdf_path: Path
    leaflets_dir: Path
    curated_tables_path: Path
    table_overrides_path: Path
    mappings_path: Path
    glossary_path: Path
    work_root: Path = BUILD_WORK_DIR
    final_dir: Path = RAG_FINAL_DIR
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS


@dataclass(frozen=True)
class BuildResult:
    build_id: str
    final_dir: Path


@dataclass(frozen=True)
class BuildHooks:
    build_candidate: Callable[[BuildConfig, ArtifactPaths], None]


def _digest_payload(config: BuildConfig) -> dict[str, Any]:
    return {
        "pdf": sha256_file(config.pdf_path),
        "leaflet_manifest": sha256_file(config.leaflets_dir / "manifest.json"),
        "curated_tables": sha256_file(config.curated_tables_path),
        "table_overrides": sha256_file(config.table_overrides_path),
        "mappings": sha256_file(config.mappings_path),
        "glossary": sha256_file(config.glossary_path),
        "valid_syllables": sha256_file(SOURCES_DIR / "vietnamese_valid_syllables.json"),
        "max_chars": config.max_chars,
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": "rag-final-v3",
    }


def build_id_for(config_payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        config_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_state(paths: ArtifactPaths, **values: Any) -> None:
    state: dict[str, Any] = {}
    if paths.state_path.exists():
        state = json.loads(paths.state_path.read_text(encoding="utf-8"))
    state.update(values)
    paths.state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _update_source_audit_counts(source_final_dir: Path) -> None:
    sections = read_jsonl(source_final_dir / "sections.jsonl")
    chunks = read_jsonl(source_final_dir / "chunks.jsonl")
    audit_path = source_final_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["section_count"] = len(sections)
    audit["chunk_count"] = len(chunks)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_candidate(config: BuildConfig, paths: ArtifactPaths) -> None:
    require_materialized_pdf(config.pdf_path)
    for input_path in (
        config.curated_tables_path,
        config.table_overrides_path,
        config.mappings_path,
        config.glossary_path,
        SOURCES_DIR / "vietnamese_valid_syllables.json",
    ):
        require_materialized_file(input_path)
    leaflet_source = verify_leaflet_source(config.leaflets_dir)
    # The HTML is read in place, so only the PDF stages need room in the workspace.
    require_workspace_capacity(
        paths.root,
        required_bytes=3 * config.pdf_path.stat().st_size
        + sum(path.stat().st_size for path in leaflet_source.html_dir.rglob("*.html")),
    )
    process_pdf(config.pdf_path, paths.raw_text)
    clean_corpus(input_path=paths.raw_text, output_path=paths.cleaned_text)
    preprocess_corpus(
        input_path=paths.cleaned_text,
        output_dir=paths.rag_dir,
        max_chars=config.max_chars,
    )
    process_canonical_rag(
        sections_path=paths.rag_dir / "sections.jsonl",
        chunks_path=paths.rag_dir / "chunks.jsonl",
        tables_path=config.curated_tables_path,
        table_overrides_path=config.table_overrides_path,
        canonical_dir=paths.canonical_dir,
        final_dir=paths.source_final_dir,
        max_chars=config.max_chars,
    )
    parse_html_tree(leaflet_source.html_dir, paths.leaflet_markdown_dir)
    integrate_leaflet_corpus(
        markdown_dir=paths.leaflet_markdown_dir,
        sections_path=paths.source_final_dir / "sections.jsonl",
        chunks_path=paths.source_final_dir / "chunks.jsonl",
        mappings_path=config.mappings_path,
        output_sections_path=paths.source_final_dir / "sections.jsonl",
        output_chunks_path=paths.source_final_dir / "chunks.jsonl",
    )
    _update_source_audit_counts(paths.source_final_dir)

    source_report = validate_final_rag(
        final_dir=paths.source_final_dir,
        canonical_dir=paths.canonical_dir,
        rag_interim_dir=paths.rag_dir,
        docling_dir=config.curated_tables_path.parent,
        leaflet_markdown_dir=paths.leaflet_markdown_dir,
        final_sections_path=paths.source_final_dir / "sections.jsonl",
        final_chunks_path=paths.source_final_dir / "chunks.jsonl",
        curated_tables_path=config.curated_tables_path,
        include_deep_audit=False,
    )
    deep_report = run_deep_audit(
        final_dir=paths.source_final_dir,
        canonical_dir=paths.canonical_dir,
        rag_interim_dir=paths.rag_dir,
        docling_dir=config.curated_tables_path.parent,
        leaflet_markdown_dir=paths.leaflet_markdown_dir,
        final_sections_path=paths.source_final_dir / "sections.jsonl",
        final_chunks_path=paths.source_final_dir / "chunks.jsonl",
        curated_tables_path=config.curated_tables_path,
    )
    shutil.copy2(
        paths.source_final_dir / "sections.jsonl",
        paths.candidate_final_dir / "sections.jsonl",
    )
    shutil.copy2(
        paths.canonical_dir / "blocks.jsonl",
        paths.candidate_final_dir / "blocks.jsonl",
    )
    combined = combine_validation_reports(
        source_report=source_report.to_dict(),
        deep_report=deep_report.to_dict(),
    )
    (paths.candidate_final_dir / "validation_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_digests = _digest_payload(config)
    manifest = build_manifest(
        paths.candidate_final_dir,
        build_id=build_id_for(input_digests),
        source_pdf_sha256=input_digests["pdf"],
        leaflet_source={
            "manifest_sha256": leaflet_source.manifest_sha256,
            "file_count": leaflet_source.file_count,
        },
        curated_input_digests={
            key: input_digests[key]
            for key in ("curated_tables", "table_overrides", "mappings", "glossary")
        },
        config_digest=hashlib.sha256(
            json.dumps(input_digests, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )
    (paths.candidate_final_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


DEFAULT_BUILD_HOOKS = BuildHooks(build_candidate=build_candidate)


def run_build(
    config: BuildConfig,
    *,
    hooks: BuildHooks = DEFAULT_BUILD_HOOKS,
) -> BuildResult:
    # The build id hashes only the leaflet manifest, so the HTML is checked first; an
    # edited file would otherwise pass the "already built" shortcut below.
    verify_leaflet_source(config.leaflets_dir)
    payload = _digest_payload(config)
    build_id = build_id_for(payload)
    if config.final_dir.is_dir():
        try:
            manifest = validate_contract_directory(config.final_dir)
        except Exception:
            manifest = None
        if manifest is not None and manifest.get("build_id") == build_id:
            return BuildResult(build_id, config.final_dir)
    paths = ArtifactPaths.create(config.work_root)
    _write_state(paths, build_id=build_id, stage="created")
    try:
        hooks.build_candidate(config, paths)
        _write_state(paths, stage="candidate-ready")
        publish_contract(paths.candidate_final_dir, config.final_dir)
        _write_state(paths, stage="published")
        paths.cleanup()
    except BaseException as exc:
        _write_state(paths, stage="failed", error=f"{type(exc).__name__}: {exc}")
        retain_failed_workspace(paths)
        raise
    return BuildResult(build_id=build_id, final_dir=config.final_dir)


def default_config() -> BuildConfig:
    return BuildConfig(
        pdf_path=FORMULARY_PDF_PATH,
        leaflets_dir=LEAFLETS_DIR,
        curated_tables_path=SOURCES_CURATION_DIR / "docling_tables.jsonl",
        table_overrides_path=SOURCES_CURATION_DIR / "table_duplicate_overrides.json",
        mappings_path=SOURCES_DIR / "colloquial_mappings.json",
        glossary_path=SOURCES_DIR / "term_glossary.json",
    )
