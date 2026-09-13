"""Map a published corpus build (`rag-final/`) to a knowledge-bundle/v1 directory."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    BundleCollection,
    BundleGenerator,
    BundleManifest,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    KnowledgeBundle,
    RetrievalMode,
    SectionRecord,
    SourceInfo,
)
from pharma_agent.domain.corpus.chunking import MAX_CHUNK_CHARS

from seed_pipeline.bundle.io import write_validated_bundle
from seed_pipeline.corpus.canonical.build_canonical_rag import (
    APPENDIX_LIST_SECTION_IDS,
    BRAND_INDEX_SECTION_ID,
)
from seed_pipeline.corpus.crawling.integrate_ankhang import (
    product_names_from_title,
    resolve_colloquial_mapping,
)
from seed_pipeline.corpus.metadata.payload_layers import compact_colloquial_mapping
from seed_pipeline.corpus.processing.preprocess_rag_corpus import slugify
from seed_pipeline.evaluation.artifact_contracts import iter_jsonl_objects

COLLECTION_KEY = "formulary"
COLLECTION_TITLE = "Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng An Khang"
ANKHANG_SECTION_PREFIX = "brand:ankhang:"
ANKHANG_BASE_URL = "https://www.nhathuocankhang.com"
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ExportRequest:
    rag_final_dir: Path
    glossary_path: Path
    mappings_path: Path
    output_dir: Path
    force: bool = False


@dataclass(frozen=True)
class ExportResult:
    manifest: BundleManifest
    skipped_sections: tuple[str, ...]


@dataclass
class _MappingDraft:
    aliases: list[str]
    visual_sign: str
    product_names: list[str] = field(default_factory=list)
    section_keys: list[str] = field(default_factory=list)


def block_kind_for(section_key: str, content_type: str) -> BlockKind:
    if content_type == "table":
        return BlockKind.TABLE
    if section_key == BRAND_INDEX_SECTION_ID:
        return BlockKind.INDEX_ENTRIES
    if section_key in APPENDIX_LIST_SECTION_IDS:
        return BlockKind.LIST
    return BlockKind.PROSE


def document_for_section(row: Mapping[str, Any]) -> DocumentRecord:
    section_key = str(row["id"])
    title = str(row["title"])
    source_title = str(row.get("source") or "")
    if section_key.startswith(ANKHANG_SECTION_PREFIX):
        category, slug = _leaflet_path(section_key)
        return DocumentRecord(
            key=f"leaflet:ankhang:{category}:{slug}",
            kind=DocumentKind.LEAFLET,
            title=title,
            source=SourceInfo(
                title=source_title, url=f"{ANKHANG_BASE_URL}/{category}/{slug}"
            ),
        )
    content_type = str(row.get("content_type") or "")
    if content_type == "drug_monograph":
        return DocumentRecord(
            key=f"drug:{slugify(title)}",
            kind=DocumentKind.DRUG_MONOGRAPH,
            title=title,
            source=SourceInfo(title=source_title),
        )
    if content_type == "general_monograph":
        return DocumentRecord(
            key=f"general:{slugify(title)}",
            kind=DocumentKind.GENERAL_MONOGRAPH,
            title=title,
            source=SourceInfo(title=source_title),
        )
    raise ValueError(
        f"Section {section_key} has unsupported content_type {content_type!r}"
    )


def export_bundle(request: ExportRequest) -> ExportResult:
    output_dir = Path(request.output_dir)
    if output_dir.exists() and not request.force:
        raise FileExistsError(
            f"Bundle directory already exists: {output_dir}; use --force to replace it"
        )
    rag_final_dir = Path(request.rag_final_dir)
    blocks_by_section = _blocks_by_section(
        iter_jsonl_objects(rag_final_dir / "blocks.jsonl")
    )
    build_manifest = _read_json_object(rag_final_dir / "manifest.json")
    mappings = _MappingCollector(_read_json_object(request.mappings_path))
    documents: dict[str, DocumentRecord] = {}
    ordinals: dict[str, int] = {}
    sections: list[SectionRecord] = []
    skipped: list[str] = []
    for row in iter_jsonl_objects(rag_final_dir / "sections.jsonl"):
        section_key = str(row["id"])
        is_leaflet = section_key.startswith(ANKHANG_SECTION_PREFIX)
        blocks = (
            leaflet_blocks(str(row.get("text") or ""))
            if is_leaflet
            else _formulary_blocks(section_key, blocks_by_section.get(section_key, []))
        )
        if not blocks:
            skipped.append(section_key)
            continue
        document = document_for_section(row)
        existing = documents.setdefault(document.key, document)
        if existing.title != document.title:
            raise ValueError(
                f"Document key {document.key} is shared by titles "
                f"{existing.title!r} and {document.title!r}"
            )
        ordinals[document.key] = ordinals.get(document.key, 0) + 1
        heading = str(row["section"])
        sections.append(
            SectionRecord(
                key=section_key,
                document_key=document.key,
                heading=heading,
                context_path=[
                    str(part) for part in row.get("context_path") or [heading]
                ],
                ordinal=ordinals[document.key],
                start_page=_page(row.get("start_page")),
                end_page=_page(row.get("end_page")),
                retrieval=(
                    RetrievalMode.INDEX_ONLY
                    if section_key == BRAND_INDEX_SECTION_ID
                    else RetrievalMode.DEFAULT
                ),
                blocks=blocks,
            )
        )
        if is_leaflet:
            mappings.add_leaflet(section_key, document.title)

    bundle = KnowledgeBundle(
        manifest=BundleManifest(
            schema_version="knowledge-bundle/v1",
            collection=BundleCollection(key=COLLECTION_KEY, title=COLLECTION_TITLE),
            generator=BundleGenerator(
                name="seed-pipeline",
                version=version("seed-pipeline"),
                build_id=str(build_manifest["build_id"]),
            ),
            source_digests=_source_digests(build_manifest),
            document_count=len(documents),
            section_count=len(sections),
            files={},
        ),
        documents=list(documents.values()),
        sections=sections,
        glossary=_glossary(request.glossary_path),
        colloquial_mappings=mappings.records(),
    )
    manifest = write_validated_bundle(bundle, output_dir)
    return ExportResult(manifest=manifest, skipped_sections=tuple(skipped))


class _MappingCollector:
    def __init__(self, curated: dict[str, Any]) -> None:
        self._curated = curated
        self._drafts: dict[str, _MappingDraft] = {}
        for key, value in curated.items():
            entry = value if isinstance(value, dict) else {}
            self._drafts[key] = _MappingDraft(
                aliases=[str(alias) for alias in entry.get("aliases") or []],
                visual_sign=str(entry.get("visual_sign") or ""),
            )

    def add_leaflet(self, section_key: str, title: str) -> None:
        _, slug = _leaflet_path(section_key)
        resolved = resolve_colloquial_mapping(slug, self._curated)
        key = str(resolved["mapping_key"]) or slug
        draft = self._drafts.setdefault(key, _MappingDraft(aliases=[], visual_sign=""))
        for name in product_names_from_title(title) or [title]:
            if name not in draft.product_names:
                draft.product_names.append(name)
        draft.section_keys.append(section_key)

    def records(self) -> list[ColloquialMappingRecord]:
        records: list[ColloquialMappingRecord] = []
        for key, draft in self._drafts.items():
            compact = compact_colloquial_mapping(
                {
                    "product_aliases": draft.aliases,
                    "visual_sign": draft.visual_sign,
                    "product_names": draft.product_names,
                }
            )
            records.append(
                ColloquialMappingRecord(
                    key=key,
                    aliases=[str(alias) for alias in compact.get("aliases", [])],
                    visual_sign=str(compact.get("visual_sign", "")),
                    product_names=[
                        str(name) for name in compact.get("product_names", [])
                    ],
                    section_keys=list(draft.section_keys),
                )
            )
        return records


def _leaflet_path(section_key: str) -> tuple[str, str]:
    category, _, slug = section_key.removeprefix(ANKHANG_SECTION_PREFIX).rpartition(":")
    if not category or not slug:
        raise ValueError(f"An Khang section key is malformed: {section_key}")
    return category, slug


def _blocks_by_section(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["section_id"]), []).append(row)
    return grouped


def _formulary_blocks(
    section_key: str, rows: list[dict[str, Any]]
) -> list[BlockRecord]:
    blocks: list[BlockRecord] = []
    for row in rows:
        content_type = str(row.get("content_type") or "")
        is_table = content_type == "table"
        markdown = str((row.get("markdown") if is_table else row.get("text")) or "")
        if not markdown.strip():
            continue
        blocks.append(
            BlockRecord(
                kind=block_kind_for(section_key, content_type),
                markdown=markdown,
                start_page=_page(row.get("page_start")),
                end_page=_page(row.get("page_end")),
                table_key=str(row["table_id"])
                if is_table and row.get("table_id")
                else None,
            )
        )
    return blocks


def leaflet_blocks(text: str) -> list[BlockRecord]:
    """P1 decision 4: paragraphs joined into prose blocks, long pipe tables kept apart."""
    blocks: list[BlockRecord] = []
    prose: list[str] = []
    for paragraph in (part.strip() for part in _PARAGRAPH_BREAK.split(text)):
        if not paragraph:
            continue
        if paragraph.startswith("|") and len(paragraph) > MAX_CHUNK_CHARS:
            if prose:
                blocks.append(
                    BlockRecord(kind=BlockKind.PROSE, markdown="\n\n".join(prose))
                )
                prose = []
            blocks.append(BlockRecord(kind=BlockKind.TABLE, markdown=paragraph))
        else:
            prose.append(paragraph)
    if prose:
        blocks.append(BlockRecord(kind=BlockKind.PROSE, markdown="\n\n".join(prose)))
    return blocks


def _page(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _glossary(path: Path) -> list[GlossaryEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Glossary must be a JSON list: {path}")
    return [GlossaryEntry.model_validate(entry) for entry in payload]


def _source_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    digests = {
        "source_pdf_sha256": str(manifest["source_pdf_sha256"]),
        "snapshot_id": str(manifest["snapshot_id"]),
        "snapshot_sha256": str(manifest["snapshot_sha256"]),
    }
    curated = manifest.get("curated_input_digests") or {}
    for name, digest in sorted(curated.items()):
        digests[f"curated_{name}_sha256"] = str(digest)
    return digests
