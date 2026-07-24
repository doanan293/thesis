from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.paths import RAG_FINAL_DIR, RAW_DIR
from rag_metadata.payload_layers import (
    clean_context_header,
    compact_colloquial_mapping,
    format_colloquial_mapping,
    term_annotations_from_enrichments,
)
from rag_metadata.qdrant_payload_contract import (
    QDRANT_RUNTIME_PAYLOAD_FIELDS,
    compact_runtime_payload,
    compact_validated_runtime_payload,
    validate_runtime_payload,
)
from rag_metadata.term_enrichment import (
    build_term_enrichment_audit,
    detect_term_enrichments,
    format_term_search_text,
    load_term_glossary,
)

SECTIONS_PATH = str(RAG_FINAL_DIR / "sections.jsonl")
CHUNKS_PATH = str(RAG_FINAL_DIR / "chunks.jsonl")
OUTPUT_METADATA_CHUNKS_PATH = str(RAG_FINAL_DIR / "chunks.jsonl")
TERM_GLOSSARY_PATH = RAW_DIR / "term_glossary.json"
TERM_ENRICHMENT_AUDIT_PATH = RAG_FINAL_DIR / "term_enrichment_audit.json"

DEFAULT_CORPUS_VERSION = "rag-final-2026-07-05"


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _section_lookup(sections: list[dict]) -> dict[str, dict]:
    return {str(section.get("id") or ""): section for section in sections}


def _first_non_empty(*values) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _value(chunk: dict, section: dict, field: str, default=""):
    value = chunk.get(field)
    if value is None or value == "":
        value = section.get(field, default)
    return value


def _int_value(chunk: dict, section: dict, field: str, default: int = 0) -> int:
    value = _value(chunk, section, field, default)
    if value in (None, ""):
        return default
    return int(value)


def _context_header(chunk: dict, section: dict, title: str, section_name: str) -> str:
    header = clean_context_header(
        _first_non_empty(chunk.get("context_header"), section.get("context_header"))
    )
    if header:
        return header
    if title and section_name:
        return f"{title} > {section_name}"
    return _first_non_empty(title, section_name)


def _required_source_chunk_text(chunk: dict) -> str:
    chunk_text = str(chunk.get("text") or "").strip()
    if chunk_text:
        return chunk_text
    chunk_id = chunk.get("id") or chunk.get("chunk_id") or "<unknown>"
    raise ValueError(f"Chunk {chunk_id} missing required chunk text")


def _embedding_text(
    *,
    context_header: str,
    chunk_text: str,
    colloquial_mapping: dict,
    term_enrichments: list[dict],
) -> str:
    visible_text = f"{context_header}\n\n{chunk_text}" if context_header else chunk_text
    parts = []
    if context_header:
        parts.append(context_header)
    colloquial_text = format_colloquial_mapping(colloquial_mapping, visible_text)
    if colloquial_text:
        parts.append(colloquial_text)
    parts.append(chunk_text)
    term_text = format_term_search_text(term_enrichments)
    if term_text and "thuật ngữ:" not in "\n\n".join(parts).casefold():
        parts.append(term_text)
    return "\n\n".join(parts).strip()


def build_runtime_chunk_metadata(
    chunk: dict,
    section: dict,
    glossary_entries: list[dict] | None = None,
) -> dict:
    section_id = str(chunk.get("section_id") or section.get("id") or "")
    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    title = _first_non_empty(chunk.get("title"), section.get("title"))
    section_name = _first_non_empty(chunk.get("section"), section.get("section"))
    context_header = _context_header(chunk, section, title, section_name)
    chunk_text = _required_source_chunk_text(chunk)
    colloquial_mapping = compact_colloquial_mapping({**section, **chunk})
    searchable_text = (
        f"{context_header}\n\n{chunk_text}" if context_header else chunk_text
    )
    term_enrichments = detect_term_enrichments(searchable_text, glossary_entries or [])

    row = {
        "chunk_id": chunk_id,
        "section_id": section_id,
        "chunk_index": _int_value(chunk, section, "chunk_index", 0),
        "hydrate_strategy": str(_value(chunk, section, "hydrate_strategy", "")),
        "source": _first_non_empty(chunk.get("source"), section.get("source")),
        "title": title,
        "section": section_name,
        "start_page": _int_value(chunk, section, "start_page", 0),
        "end_page": _int_value(chunk, section, "end_page", 0),
        "context_header": context_header,
        "chunk_text": chunk_text,
        "embedding_text": _embedding_text(
            context_header=context_header,
            chunk_text=chunk_text,
            colloquial_mapping=colloquial_mapping,
            term_enrichments=term_enrichments,
        ),
        "colloquial_mapping": colloquial_mapping,
        "term_annotations": term_annotations_from_enrichments(term_enrichments),
        "content_type": str(_value(chunk, section, "content_type", "")),
        "table_id": str(chunk.get("table_id") or ""),
    }
    return compact_runtime_payload(row)


def build_unified_chunk(
    chunk: dict,
    section: dict,
    glossary_entries: list[dict] | None = None,
) -> dict:
    runtime = build_runtime_chunk_metadata(chunk, section, glossary_entries)
    return {
        **runtime,
        "chunk_role": str(chunk.get("chunk_role") or "prose"),
        "chunk_content_type": str(
            chunk.get("chunk_content_type") or chunk.get("content_type") or ""
        ),
        "source_block_id": str(chunk.get("source_block_id") or ""),
        "context_path": list(chunk.get("context_path") or []),
        "section_char_count": int(chunk.get("section_char_count") or 0),
        "warnings": list(chunk.get("warnings") or []),
    }


def build_qdrant_payload(runtime_chunk: dict) -> dict:
    return compact_validated_runtime_payload(
        runtime_chunk,
        label=f"Runtime chunk {runtime_chunk.get('chunk_id') or '<unknown>'}",
    )


def _write_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_unified_chunks(
    compiled_chunks: list[dict], output_path: str | Path = OUTPUT_METADATA_CHUNKS_PATH
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in compiled_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def validate_qdrant_payloads_from_metadata(
    runtime_chunks: list[dict], payload_size_warning_bytes: int = 16_384
) -> list[str]:
    warnings = []
    seen_chunk_ids = set()
    for index, chunk in enumerate(runtime_chunks, start=1):
        runtime = {
            field: chunk[field]
            for field in QDRANT_RUNTIME_PAYLOAD_FIELDS
            if field in chunk
        }
        payload = build_qdrant_payload(runtime)
        chunk_id = str(payload.get("chunk_id") or "")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate Qdrant chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        payload_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if payload_size > payload_size_warning_bytes:
            warnings.append(
                f"Qdrant payload for {chunk_id or index} is large: {payload_size} bytes"
            )
    return warnings


def validate_canonical_chunks(canonical_chunks: list[dict]) -> None:
    seen_chunk_ids = set()
    for index, row in enumerate(canonical_chunks, start=1):
        label = str(row.get("chunk_id") or index)
        runtime = {
            field: row[field] for field in QDRANT_RUNTIME_PAYLOAD_FIELDS if field in row
        }
        validate_runtime_payload(runtime, label=f"Runtime chunk {label}")
        chunk_id = str(row["chunk_id"])
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate canonical chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)


def compile_unified_chunks(
    sections_path: str | Path = SECTIONS_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
    corpus_version: str = DEFAULT_CORPUS_VERSION,
    glossary_path: str | Path = TERM_GLOSSARY_PATH,
    enable_term_enrichment: bool = True,
) -> tuple[list[dict], dict]:
    del corpus_version
    sections = _read_jsonl(sections_path)
    chunks = _read_jsonl(chunks_path)
    sections_by_id = _section_lookup(sections)
    glossary_entries = (
        load_term_glossary(glossary_path) if enable_term_enrichment else []
    )

    compiled_chunks = []
    for chunk in chunks:
        section_id = str(chunk.get("section_id") or "")
        section = sections_by_id.get(section_id, {})
        compiled_chunks.append(build_unified_chunk(chunk, section, glossary_entries))

    return compiled_chunks, {}


def run_metadata_build(
    sections_path: str | Path = SECTIONS_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
    output_path: str | Path = OUTPUT_METADATA_CHUNKS_PATH,
    corpus_version: str = DEFAULT_CORPUS_VERSION,
    glossary_path: str | Path = TERM_GLOSSARY_PATH,
    term_audit_path: str | Path | None = None,
    enable_term_enrichment: bool = True,
) -> None:
    glossary_entries = (
        load_term_glossary(glossary_path) if enable_term_enrichment else []
    )
    compiled_chunks, _metadata = compile_unified_chunks(
        sections_path,
        chunks_path,
        corpus_version=corpus_version,
        glossary_path=glossary_path,
        enable_term_enrichment=enable_term_enrichment,
    )
    validate_canonical_chunks(compiled_chunks)
    validate_qdrant_payloads_from_metadata(compiled_chunks)
    write_unified_chunks(compiled_chunks, output_path)
    audit = build_term_enrichment_audit(compiled_chunks, glossary_entries)
    if term_audit_path is not None:
        Path(term_audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(term_audit_path).write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Metadata build complete. Wrote {len(compiled_chunks)} unified chunks to {output_path}."
    )
    if term_audit_path is not None:
        print(f"Term enrichment audit written to {term_audit_path}.")
    print(f"Compiled unified metadata for {len(compiled_chunks)} chunks.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qdrant-ready RAG metadata chunks."
    )
    parser.add_argument(
        "--sections", default=SECTIONS_PATH, help="Input sections JSONL path."
    )
    parser.add_argument(
        "--chunks", default=CHUNKS_PATH, help="Input chunks JSONL path."
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_METADATA_CHUNKS_PATH,
        help="Output unified chunks JSONL path.",
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
        help="Accepted for CLI compatibility; not written to runtime payloads.",
    )
    parser.add_argument(
        "--term-glossary",
        type=Path,
        default=TERM_GLOSSARY_PATH,
        help="Curated term glossary JSON path.",
    )
    parser.add_argument(
        "--term-audit",
        type=Path,
        default=None,
        help="Optional output term enrichment audit JSON path.",
    )
    parser.add_argument(
        "--no-term-enrichment",
        action="store_true",
        help="Disable term glossary enrichment.",
    )
    args = parser.parse_args()
    run_metadata_build(
        args.sections,
        args.chunks,
        args.output,
        corpus_version=args.corpus_version,
        glossary_path=args.term_glossary,
        term_audit_path=args.term_audit,
        enable_term_enrichment=not args.no_term_enrichment,
    )


if __name__ == "__main__":
    main()
