"""Knowledge bundle ``knowledge-bundle/v1`` (spec C §5): models, file IO and validation.

Public API: pharma-lab builds a ``KnowledgeBundle``, calls ``write_bundle`` and then adds
each model's vectors with ``write_bundle_embeddings``; the backend import calls
``read_bundle`` and loads the vectors of its model with ``read_bundle_embeddings``. Both
sides run the same checks. Embedding files are streamed line by line, so memory holds at
most one model's vectors however many models a bundle ships.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import struct
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from pharma_agent.domain.shared.errors import DomainError

BUNDLE_SCHEMA_VERSION: Final = "knowledge-bundle/v1"
MANIFEST_FILE: Final = "manifest.json"
DOCUMENTS_FILE: Final = "documents.jsonl"
SECTIONS_FILE: Final = "sections.jsonl"
GLOSSARY_FILE: Final = "glossary.json"
COLLOQUIAL_MAPPINGS_FILE: Final = "colloquial_mappings.json"
EMBEDDINGS_DIR: Final = "embeddings"
REQUIRED_FILES: Final = (
    DOCUMENTS_FILE,
    SECTIONS_FILE,
    GLOSSARY_FILE,
    COLLOQUIAL_MAPPINGS_FILE,
)

_STRICT = ConfigDict(extra="forbid", strict=True)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DocumentKind(StrEnum):
    DRUG_MONOGRAPH = "drug_monograph"
    GENERAL_MONOGRAPH = "general_monograph"
    LEAFLET = "leaflet"


class BlockKind(StrEnum):
    PROSE = "prose"
    TABLE = "table"
    LIST = "list"
    INDEX_ENTRIES = "index_entries"


class RetrievalMode(StrEnum):
    DEFAULT = "default"
    INDEX_ONLY = "index_only"


class SourceInfo(BaseModel):
    model_config = _STRICT

    title: str
    url: str | None = None


class DocumentRecord(BaseModel):
    model_config = _STRICT

    key: str
    kind: DocumentKind
    title: str
    source: SourceInfo
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class BlockRecord(BaseModel):
    model_config = _STRICT

    kind: BlockKind
    markdown: str
    start_page: int | None = None
    end_page: int | None = None
    table_key: str | None = None


class SectionRecord(BaseModel):
    model_config = _STRICT

    key: str
    document_key: str
    heading: str
    context_path: list[str]
    ordinal: int
    start_page: int | None = None
    end_page: int | None = None
    retrieval: RetrievalMode = RetrievalMode.DEFAULT
    blocks: list[BlockRecord]


class GlossaryEntry(BaseModel):
    """One row of ``data/sources/term_glossary.json``."""

    model_config = _STRICT

    term: str
    case_sensitive: bool = False
    vietnamese_expansions: list[str] = Field(default_factory=list)
    english_expansions: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    category: str = ""
    confidence: str = ""
    source: str = ""


class ColloquialMappingRecord(BaseModel):
    """Colloquial names attached to sections.

    ``key`` is the curated leaflet slug from ``data/sources/colloquial_mappings.json``
    or, for a leaflet that only has title-derived product names, the leaflet slug; it
    may also be empty. It is copied into ``ColloquialMapping.key`` and never affects
    embedding text, term annotations or chunk version ids.
    """

    model_config = _STRICT

    key: str
    aliases: list[str] = Field(default_factory=list)
    visual_sign: str = ""
    product_names: list[str] = Field(default_factory=list)
    section_keys: list[str] = Field(default_factory=list)


class BundleCollection(BaseModel):
    model_config = _STRICT

    key: str
    title: str


class BundleGenerator(BaseModel):
    model_config = _STRICT

    name: str
    version: str
    build_id: str


class BundleFile(BaseModel):
    model_config = _STRICT

    sha256: str
    bytes: int


class BundleEmbeddingFile(BaseModel):
    model_config = _STRICT

    model: str
    dims: int
    file: str


class BundleManifest(BaseModel):
    model_config = _STRICT

    schema_version: Literal["knowledge-bundle/v1"]
    collection: BundleCollection
    generator: BundleGenerator
    source_digests: dict[str, str]
    document_count: int
    section_count: int
    files: dict[str, BundleFile]
    embeddings: list[BundleEmbeddingFile] = Field(default_factory=list)


class KnowledgeBundle(BaseModel):
    """The manifest and knowledge content; vectors stay in the embedding files."""

    model_config = _STRICT

    manifest: BundleManifest
    documents: list[DocumentRecord]
    sections: list[SectionRecord]
    glossary: list[GlossaryEntry]
    colloquial_mappings: list[ColloquialMappingRecord]


class BundleValidationError(DomainError):
    code = "BUNDLE_INVALID"
    problems: list[str]

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("knowledge bundle is invalid:\n" + "\n".join(self.problems))


def model_slug(model: str) -> str:
    return _SLUG_RE.sub("_", model.lower()).strip("_")


def encode_vector(values: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def decode_vector(data: str, dims: int) -> list[float]:
    return list(struct.unpack(f"<{dims}f", _vector_bytes(data, dims)))


class _EmbeddingLine(BaseModel):
    model_config = _STRICT

    embedding_text_sha256: str = Field(pattern=_SHA256_RE.pattern)
    dims: int
    vector: str


_GLOSSARY_ADAPTER = TypeAdapter(list[GlossaryEntry])
_MAPPINGS_ADAPTER = TypeAdapter(list[ColloquialMappingRecord])


def read_bundle(directory: Path) -> KnowledgeBundle:
    """Read and validate a bundle directory; every problem is reported at once.

    Embedding files are checked line by line without keeping their vectors.
    """
    manifest = _read_manifest(directory)
    problems: list[str] = []
    line_problems: list[str] = []
    texts = _read_manifest_files(directory, manifest, problems, line_problems)
    documents = _parse_jsonl(
        DOCUMENTS_FILE, texts.get(DOCUMENTS_FILE, ""), DocumentRecord, problems
    )
    sections = _parse_jsonl(
        SECTIONS_FILE, texts.get(SECTIONS_FILE, ""), SectionRecord, problems
    )
    glossary = _parse_json_list(
        GLOSSARY_FILE, texts.get(GLOSSARY_FILE), _GLOSSARY_ADAPTER, problems
    )
    mappings = _parse_json_list(
        COLLOQUIAL_MAPPINGS_FILE,
        texts.get(COLLOQUIAL_MAPPINGS_FILE),
        _MAPPINGS_ADAPTER,
        problems,
    )
    for name, field, expected in (
        (DOCUMENTS_FILE, "document_count", manifest.document_count),
        (SECTIONS_FILE, "section_count", manifest.section_count),
    ):
        if name in texts and expected != _record_count(texts[name]):
            problems.append(
                f"{MANIFEST_FILE}: {field}: {expected} does not match "
                f"{name} ({_record_count(texts[name])} records)"
            )
    problems.extend(_content_problems(documents, sections, glossary, mappings))
    problems.extend(line_problems)

    if problems:
        raise BundleValidationError(problems)
    return KnowledgeBundle(
        manifest=manifest,
        documents=[record for _, record in documents],
        sections=[record for _, record in sections],
        glossary=glossary,
        colloquial_mappings=mappings,
    )


def read_bundle_embeddings(
    directory: Path, manifest: BundleManifest, model: str, dims: int
) -> dict[str, list[float]]:
    """Vectors of `model` keyed by embedding_text_sha256, streamed from its file.

    Empty when `manifest` declares no embeddings for `model` with `dims`. The file is
    checked again, so a change after ``read_bundle`` raises BundleValidationError.
    """
    if not any(
        (entry.model, entry.dims) == (model, dims) for entry in manifest.embeddings
    ):
        return {}
    name = _embedding_file_name(model)
    expected = manifest.files.get(name)
    if expected is None:
        raise BundleValidationError(
            [f"{MANIFEST_FILE}: files.{name}: missing entry for {model!r}"]
        )
    path = directory / name
    if not path.is_file():
        raise BundleValidationError([f"{name}: file is missing"])

    vectors: dict[str, list[float]] = {}

    def keep(sha: str, raw: bytes) -> None:
        vectors[sha] = list(struct.unpack(f"<{dims}f", raw))

    problems: list[str] = []
    size, digest = _scan_embedding_file(path, name, dims, problems, keep)
    problems.extend(_digest_problems(name, size, digest, expected))
    if problems:
        raise BundleValidationError(problems)
    return vectors


def write_bundle(bundle: KnowledgeBundle, directory: Path) -> BundleManifest:
    """Validate, write the content files and return the manifest with computed digests.

    The manifest lists no embeddings; add them with ``write_bundle_embeddings``.
    """
    problems = _content_problems(
        list(enumerate(bundle.documents, start=1)),
        list(enumerate(bundle.sections, start=1)),
        bundle.glossary,
        bundle.colloquial_mappings,
    )
    if problems:
        raise BundleValidationError(problems)
    payloads: dict[str, bytes] = {
        DOCUMENTS_FILE: _jsonl_bytes(bundle.documents),
        SECTIONS_FILE: _jsonl_bytes(bundle.sections),
        GLOSSARY_FILE: _json_list_bytes(bundle.glossary),
        COLLOQUIAL_MAPPINGS_FILE: _json_list_bytes(bundle.colloquial_mappings),
    }
    manifest = BundleManifest(
        schema_version=bundle.manifest.schema_version,
        collection=bundle.manifest.collection,
        generator=bundle.manifest.generator,
        source_digests=dict(bundle.manifest.source_digests),
        document_count=len(bundle.documents),
        section_count=len(bundle.sections),
        files={
            name: BundleFile(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            for name, data in payloads.items()
        },
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    _write_manifest(directory, manifest)
    return manifest


def write_bundle_embeddings(
    directory: Path, model: str, vectors: Iterable[tuple[str, Sequence[float]]]
) -> BundleManifest:
    """Add or replace the embedding file of `model` and return the updated manifest.

    `vectors` yields (embedding_text_sha256, vector) pairs sorted by hash without
    duplicates. They are streamed to a temporary file that replaces the model's file only
    when every pair is valid; the manifest is replaced last. Other models' files are not
    read, so memory does not grow with the number or size of the models.
    """
    manifest = _read_manifest(directory)
    where = f"embeddings[{model!r}]"
    if not model_slug(model):
        raise BundleValidationError(
            [f"{where}: model name must contain letters or digits"]
        )
    name = _embedding_file_name(model)
    for entry in manifest.embeddings:
        if entry.model != model and _embedding_file_name(entry.model) == name:
            raise BundleValidationError(
                [f"{where}: model slug collides with {entry.model!r}"]
            )
    temporary = directory / f".{model_slug(model)}.jsonl.tmp"
    try:
        listed, dims = _write_embedding_lines(temporary, where, vectors)
        (directory / EMBEDDINGS_DIR).mkdir(exist_ok=True)
        os.replace(temporary, directory / name)
    finally:
        temporary.unlink(missing_ok=True)

    embeddings = sorted(
        [
            *(entry for entry in manifest.embeddings if entry.model != model),
            BundleEmbeddingFile(model=model, dims=dims, file=name),
        ],
        key=lambda entry: entry.model,
    )
    files = {
        file: entry
        for file, entry in manifest.files.items()
        if not file.startswith(f"{EMBEDDINGS_DIR}/")
    }
    for entry in embeddings:
        current = listed if entry.model == model else manifest.files.get(entry.file)
        if current is not None:
            files[entry.file] = current
    updated = manifest.model_copy(update={"files": files, "embeddings": embeddings})
    _write_manifest(directory, updated)
    return updated


def _embedding_file_name(model: str) -> str:
    return f"{EMBEDDINGS_DIR}/{model_slug(model)}.jsonl"


def _read_manifest(directory: Path) -> BundleManifest:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise BundleValidationError([f"{MANIFEST_FILE}: file is missing"])
    try:
        return BundleManifest.model_validate_json(manifest_path.read_bytes())
    except ValidationError as error:
        raise BundleValidationError(
            _validation_problems(MANIFEST_FILE, error)
        ) from error


def _write_manifest(directory: Path, manifest: BundleManifest) -> None:
    """Replace the manifest atomically so a reader never sees a partial file."""
    temporary = directory / f".{MANIFEST_FILE}.tmp"
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, directory / MANIFEST_FILE)


def _write_embedding_lines(
    path: Path, where: str, vectors: Iterable[tuple[str, Sequence[float]]]
) -> tuple[BundleFile, int]:
    problems: list[str] = []
    lengths: set[int] = set()
    previous: str | None = None
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as handle:
        for sha, vector in vectors:
            lengths.add(len(vector))
            if not _SHA256_RE.match(sha):
                problems.append(f"{where}: {sha!r} is not a sha256 hex digest")
                continue
            if previous == sha:
                problems.append(f"{where}: {sha!r} is a duplicate")
            elif previous is not None and sha < previous:
                problems.append(f"{where}: {sha!r} is not sorted after {previous!r}")
            previous = sha
            if problems:
                continue
            record = {
                "embedding_text_sha256": sha,
                "dims": len(vector),
                "vector": encode_vector(vector),
            }
            line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            digest.update(line)
            size += len(line)
    if len(lengths) != 1 or 0 in lengths:
        problems.append(
            f"{where}: vectors must share one non-zero length, got {sorted(lengths)}"
        )
    if problems:
        raise BundleValidationError(problems)
    return BundleFile(sha256=digest.hexdigest(), bytes=size), lengths.pop()


def _format_loc(loc: Sequence[int | str]) -> str:
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text or "(root)"


def _validation_problems(where: str, error: ValidationError) -> list[str]:
    return [
        f"{where}: {_format_loc(detail['loc'])}: {detail['msg']}"
        for detail in error.errors()
    ]


def _digest_problems(
    name: str, size: int, sha256: str, expected: BundleFile
) -> list[str]:
    problems: list[str] = []
    if size != expected.bytes:
        problems.append(
            f"{name}: size {size} bytes does not match manifest ({expected.bytes})"
        )
    if sha256 != expected.sha256:
        problems.append(f"{name}: sha256 does not match manifest")
    return problems


def _read_manifest_files(
    directory: Path,
    manifest: BundleManifest,
    problems: list[str],
    line_problems: list[str],
) -> dict[str, str]:
    """Check every listed file and return the text of the content files.

    Digest problems follow the manifest's file order in `problems`; embedding files are
    streamed and their line problems go to `line_problems`.
    """
    expected_embedding_files: set[str] = set()
    embedding_dims: dict[str, int] = {}
    seen_models: set[str] = set()
    for index, entry in enumerate(manifest.embeddings):
        where = f"{MANIFEST_FILE}: embeddings[{index}]"
        expected_file = _embedding_file_name(entry.model)
        if not model_slug(entry.model):
            problems.append(f"{where}.model: must contain letters or digits")
            continue
        if entry.model in seen_models:
            problems.append(f"{where}.model: duplicate model {entry.model!r}")
        seen_models.add(entry.model)
        if entry.dims < 1:
            problems.append(f"{where}.dims: must be >= 1, got {entry.dims}")
        if entry.file != expected_file:
            problems.append(
                f"{where}.file: expected {expected_file!r}, got {entry.file!r}"
            )
        elif entry.file not in manifest.files:
            problems.append(f"{where}.file: {entry.file!r} is not listed in files")
        elif entry.dims >= 1:
            embedding_dims.setdefault(entry.file, entry.dims)
        expected_embedding_files.add(expected_file)

    problems.extend(
        f"{MANIFEST_FILE}: files.{name}: missing entry"
        for name in REQUIRED_FILES
        if name not in manifest.files
    )
    texts: dict[str, str] = {}
    for name, expected in manifest.files.items():
        if name not in REQUIRED_FILES and name not in expected_embedding_files:
            problems.append(
                f"{MANIFEST_FILE}: files.{name}: not part of {BUNDLE_SCHEMA_VERSION}"
            )
            continue
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: file is missing")
            continue
        if name not in REQUIRED_FILES:
            size, digest = _scan_embedding_file(
                path, name, embedding_dims.get(name), line_problems
            )
            problems.extend(_digest_problems(name, size, digest, expected))
            continue
        data = path.read_bytes()
        problems.extend(
            _digest_problems(
                name, len(data), hashlib.sha256(data).hexdigest(), expected
            )
        )
        try:
            texts[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}: file is not valid UTF-8")
    return texts


def _scan_embedding_file(
    path: Path,
    name: str,
    dims: int | None,
    problems: list[str],
    keep: Callable[[str, bytes], None] | None = None,
) -> tuple[int, str]:
    """Stream an embedding file and return its size and sha256.

    Lines are validated only when `dims` is known; `keep` receives the hash and raw
    float32 bytes of every valid line.
    """
    digest = hashlib.sha256()
    size = 0
    first_lines: dict[str, int] = {}
    utf8 = True
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            size += len(raw)
            if dims is None or not utf8 or not raw.strip():
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                problems.append(f"{name}: file is not valid UTF-8")
                utf8 = False
                continue
            where = f"{name}:{line_number}"
            try:
                record = _EmbeddingLine.model_validate_json(line)
            except ValidationError as error:
                problems.extend(_validation_problems(where, error))
                continue
            sha = record.embedding_text_sha256
            if record.dims != dims:
                problems.append(
                    f"{where}: dims: {record.dims} does not match manifest dims {dims}"
                )
            elif sha in first_lines:
                problems.append(
                    f"{where}: embedding_text_sha256: duplicate of line {first_lines[sha]}"
                )
            else:
                first_lines[sha] = line_number
                try:
                    vector = _vector_bytes(record.vector, dims)
                except ValueError as error:
                    problems.append(f"{where}: vector: {error}")
                    continue
                if keep is not None:
                    keep(sha, vector)
    return size, digest.hexdigest()


def _vector_bytes(data: str, dims: int) -> bytes:
    try:
        raw = base64.b64decode(data, validate=True)
    except binascii.Error as error:
        raise ValueError(f"vector is not valid base64: {error}") from error
    if len(raw) != dims * 4:
        raise ValueError(f"vector has {len(raw)} bytes, expected {dims * 4} (dims * 4)")
    return raw


def _record_count(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.strip())


def _parse_jsonl[RecordT: BaseModel](
    name: str, text: str, model: type[RecordT], problems: list[str]
) -> list[tuple[int, RecordT]]:
    records: list[tuple[int, RecordT]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append((line_number, model.model_validate_json(line)))
        except ValidationError as error:
            problems.extend(_validation_problems(f"{name}:{line_number}", error))
    return records


def _parse_json_list[RecordT](
    name: str,
    text: str | None,
    adapter: TypeAdapter[list[RecordT]],
    problems: list[str],
) -> list[RecordT]:
    if text is None:
        return []
    try:
        return adapter.validate_json(text)
    except ValidationError as error:
        problems.extend(_validation_problems(name, error))
        return []


def _page_problems(
    where: str, prefix: str, start_page: int | None, end_page: int | None
) -> list[str]:
    problems = [
        f"{where}: {prefix}{field}: must be >= 1, got {page}"
        for field, page in (("start_page", start_page), ("end_page", end_page))
        if page is not None and page < 1
    ]
    if start_page is not None and end_page is not None and start_page > end_page:
        problems.append(
            f"{where}: {prefix}start_page: {start_page} is after end_page {end_page}"
        )
    return problems


def _content_problems(
    documents: Sequence[tuple[int, DocumentRecord]],
    sections: Sequence[tuple[int, SectionRecord]],
    glossary: Sequence[GlossaryEntry],
    mappings: Sequence[ColloquialMappingRecord],
) -> list[str]:
    problems: list[str] = []
    document_lines: dict[str, int] = {}
    for line, document in documents:
        where = f"{DOCUMENTS_FILE}:{line}"
        if not document.key.strip():
            problems.append(f"{where}: key: must not be empty")
        elif document.key in document_lines:
            problems.append(
                f"{where}: key: duplicate document key {document.key!r} "
                f"(first on line {document_lines[document.key]})"
            )
        else:
            document_lines[document.key] = line

    section_lines: dict[str, int] = {}
    ordinal_lines: dict[tuple[str, int], int] = {}
    for line, section in sections:
        where = f"{SECTIONS_FILE}:{line}"
        if not section.key.strip():
            problems.append(f"{where}: key: must not be empty")
        elif section.key in section_lines:
            problems.append(
                f"{where}: key: duplicate section key {section.key!r} "
                f"(first on line {section_lines[section.key]})"
            )
        else:
            section_lines[section.key] = line
        if section.document_key not in document_lines:
            problems.append(
                f"{where}: document_key: unknown document {section.document_key!r}"
            )
        ordinal_key = (section.document_key, section.ordinal)
        if ordinal_key in ordinal_lines:
            problems.append(
                f"{where}: ordinal: duplicate ordinal {section.ordinal} in document "
                f"{section.document_key!r} (first on line {ordinal_lines[ordinal_key]})"
            )
        else:
            ordinal_lines[ordinal_key] = line
        problems.extend(_page_problems(where, "", section.start_page, section.end_page))
        if not section.blocks:
            problems.append(f"{where}: blocks: must contain at least one block")
        for index, block in enumerate(section.blocks):
            prefix = f"blocks[{index}]."
            if not block.markdown.strip():
                problems.append(f"{where}: {prefix}markdown: must not be empty")
            problems.extend(
                _page_problems(where, prefix, block.start_page, block.end_page)
            )

    term_indexes: dict[str, int] = {}
    for index, entry in enumerate(glossary):
        where = f"{GLOSSARY_FILE}: [{index}].term"
        term = entry.term.strip()
        if not term:
            problems.append(f"{where}: must not be empty")
        elif term.casefold() in term_indexes:
            problems.append(
                f"{where}: duplicate term {term!r} (case-insensitive, first at "
                f"[{term_indexes[term.casefold()]}])"
            )
        else:
            term_indexes[term.casefold()] = index

    mapped_by: dict[str, int] = {}
    for index, record in enumerate(mappings):
        for key_index, section_key in enumerate(record.section_keys):
            where = f"{COLLOQUIAL_MAPPINGS_FILE}: [{index}].section_keys[{key_index}]"
            if section_key not in section_lines:
                problems.append(f"{where}: unknown section {section_key!r}")
            elif section_key in mapped_by:
                problems.append(
                    f"{where}: section {section_key!r} is already mapped by "
                    f"[{mapped_by[section_key]}]"
                )
            else:
                mapped_by[section_key] = index
    return problems


def _jsonl_bytes(records: Sequence[BaseModel]) -> bytes:
    return "".join(f"{record.model_dump_json()}\n" for record in records).encode()


def _json_list_bytes(records: Sequence[BaseModel]) -> bytes:
    return (
        json.dumps(
            [record.model_dump(mode="json") for record in records],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode()
