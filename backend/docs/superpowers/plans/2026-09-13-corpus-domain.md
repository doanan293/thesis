# Corpus Domain Implementation Plan (Plan 1 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the framework-free corpus domain in the backend: shared text helpers, the `knowledge-bundle/v1` models with file IO and full validation, deterministic section/chunk identities, glossary and colloquial enrichment, the hydrate policy and the single chunker, ported from corpus-pipeline with identical output.

**Architecture:** Everything lives in `pharma_agent.domain.corpus` (plus `pharma_agent.domain.shared.text`) and depends only on the standard library, Pydantic and existing domain models (`TermAnnotation`, `ColloquialMapping`, `HydrateStrategy`, `DomainError`). The chunker picks a splitter only from `BlockRecord.kind`; the hydrate policy uses only `SectionRecord.retrieval` and the section length. No database, Qdrant or settings code is touched; P2 builds storage and import on top of these names.

**Tech Stack:** Python 3.12, Pydantic 2.13 (strict models, `TypeAdapter`), `hashlib`, `uuid.uuid5`, `struct`/`base64`, pytest 9 with `asyncio_mode = "auto"`.

**Spec:** `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §4 (code layout, public API), §5 (bundle format and validation), §6.1 (identity functions), §7 (chunker, enrichment, hydrate policy, parity). Names pinned in `backend/docs/superpowers/plans/2026-09-13-corpus-domain.md` §2 and §3.1–§3.2.

## Global Constraints

- The environment is development only. Postgres and Qdrant may be reset; no data backfill or backward compatibility is needed.
- Python 3.12, uv, shared `ruff.toml`, strict `pyrefly.toml` with `unused-ignore = true`, pytest `filterwarnings = ["error"]`. Lint and type errors are fixed in code; never add rule ignores, `# noqa`, `# type: ignore` or new `# pyrefly: ignore`.
- Every task ends green on (run from `backend/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`. This plan touches neither Postgres nor Qdrant, so no task needs `-m integration`.
- Layering stays enforced by `tests/architecture/test_layering.py`: `pharma_agent.domain.corpus` imports no framework (`sqlalchemy`, `qdrant_client`, `fastapi`, `openai`, `httpx`, `langgraph`...) and no outer layer. Pydantic is allowed (the domain already uses it).
- Prefer established libraries over custom code (Pydantic for parsing and error locations, stdlib for hashing and encoding). No feature flag or "fake mode" in production code; fakes live under `tests/`.
- Commits: one commit per task, conventional message, ending with the session attribution trailer given in the executing session.
- Pinned names (overview §3.1–§3.2) are used exactly; P2–P4 depend on them.
- Exact values owned by this plan:
  - `BUNDLE_SCHEMA_VERSION = "knowledge-bundle/v1"`, `CHUNKER_VERSION = "chunker-v1"`, `MAX_CHUNK_CHARS = 3000`, `FULL_SECTION_MAX_CHARS = 16000`, `TERM_ANNOTATION_LIMIT = 8`.
  - `CORPUS_NAMESPACE = uuid5(NAMESPACE_URL, "pharma-agent:corpus:v1")`; the separator inside identity strings is `"\x1f"`.
  - Bundle file names: `manifest.json`, `documents.jsonl`, `sections.jsonl`, `glossary.json` (JSON array of `GlossaryEntry`), `colloquial_mappings.json` (JSON array of `ColloquialMappingRecord`), `embeddings/<model_slug>.jsonl`. JSONL files hold one compact JSON object per line and end with `"\n"`; JSON files are UTF-8, `indent=2`, non-ASCII kept.
  - Bundle models use `ConfigDict(extra="forbid", strict=True)`; bundle JSON is parsed with `model_validate_json`/`TypeAdapter.validate_json` so enum strings are accepted from files.
  - Validation messages have the form `<file>[:<line>]: <field path>: <problem>`; field paths use `blocks[0].markdown` style.
  - `ColloquialMappingRecord.key` is the curated An Khang slug, or for a leaflet that only has title-derived product names (2 417 of 2 475 leaflets today) the leaflet slug (P4); validation also accepts `""`. The key is copied into `ColloquialMapping.key` only and never affects `embedding_text`, `term_annotations` or `chunk_version_id` (tested in Tasks 3 and 8). A section key may appear in at most one record.
  - `write_bundle` always recomputes `document_count`, `section_count`, `files` (sha256 and bytes) and `embeddings` from the content it writes; callers may pass placeholders (P2 fixture generator, P4 `seed bundle embed`).
  - Glossary terms must be unique case-insensitively (makes `compose_embedding_text` from `TermAnnotation`s identical to the old enrichment-based text).
  - Pages are integers `>= 1` or `None`; the old pipeline's `0` means `None`.
  - Chunk `ordinal` starts at 1 and counts across all blocks of the section, like the old `chunk_index`.

---

## File Structure

```text
backend/
  src/pharma_agent/domain/
    shared/text.py              normalize_text, make_snippet                                   # Task 1
    corpus/__init__.py          package docstring naming the public API for seed-pipeline      # Task 2
    corpus/bundle.py            enums, bundle models, BundleValidationError, model_slug,
                                encode_vector/decode_vector                                    # Task 2
                                read_bundle, write_bundle and §5.3 validation                  # Task 3
    corpus/identity.py          CORPUS_NAMESPACE, canonical_json, sha256_hex,
                                section_revision_id, chunk_version_id                          # Task 4
    corpus/hydrate.py           FULL_SECTION_MAX_CHARS, section_char_count, hydrate_strategy_for # Task 5
    corpus/enrichment.py        build_context_header, detect_terms, mapping_for_section,
                                compose_embedding_text                                         # Task 6
    corpus/chunking.py          CHUNKER_VERSION, MAX_CHUNK_CHARS, ported splitters             # Task 7
                                ChunkDraft, chunk_section                                      # Task 8
  tests/
    domain/test_text.py                                                                        # Task 1
    domain/corpus/__init__.py, domain/corpus/test_bundle_models.py                             # Task 2
    domain/corpus/test_bundle_io.py                                                            # Task 3
    domain/corpus/test_identity.py                                                             # Task 4
    domain/corpus/test_hydrate.py                                                              # Task 5
    domain/corpus/test_enrichment.py                                                           # Task 6
    domain/corpus/test_splitters.py                                                            # Task 7
    domain/corpus/golden/chunking_v1.json   one small section per block kind + expected chunks
                                            generated by the old corpus-pipeline code           # Task 8
    domain/corpus/test_chunking.py          golden test, determinism, public API imports       # Task 8
```

No existing file is modified. `EvidenceSet.summary_view` switches to `make_snippet` in P3, not here.

All commands below run from `backend/`.

---

### Task 1: Shared text helpers

**Files:**
- Create: `backend/src/pharma_agent/domain/shared/text.py`
- Test: `backend/tests/domain/test_text.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalize_text(text: str) -> str` (NFC, `"\r\n"`/`"\r"` → `"\n"`, `rstrip` each line, drop leading/trailing blank lines, keep indentation of the first line); `make_snippet(text: str, max_chars: int) -> str` (drop markdown table rule rows and pipes, collapse whitespace, cut at a word boundary, append `"…"` when cut, result length `<= max_chars`, `ValueError` when `max_chars < 1`); `ELLIPSIS = "…"`.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/test_text.py`:

```python
import unicodedata

import pytest

from pharma_agent.domain.shared.text import make_snippet, normalize_text


def test_normalize_text_unifies_unicode_line_endings_and_blank_edges() -> None:
    decomposed = unicodedata.normalize("NFD", "Liều dùng")
    raw = f"\n  \r\n{decomposed}  \r\nTrẻ em:\t\r\r\n  - 10 mg/kg  \n\n"

    assert normalize_text(raw) == "Liều dùng\nTrẻ em:\n\n  - 10 mg/kg"


def test_normalize_text_is_idempotent_and_empties_blank_input() -> None:
    text = "PARACETAMOL\n> Liều lượng và cách dùng"

    assert normalize_text(normalize_text(text)) == text
    assert normalize_text(" \n\t\n") == ""
    assert normalize_text("") == ""


def test_make_snippet_drops_table_pipes_and_rules() -> None:
    table = (
        "| Thuốc phối hợp | Hậu quả |\n"
        "| :--- | ---: |\n"
        "| Warfarin | Tăng INR khi dùng kéo dài |"
    )

    assert make_snippet(table, 300) == (
        "Thuốc phối hợp Hậu quả Warfarin Tăng INR khi dùng kéo dài"
    )


def test_make_snippet_cuts_at_word_boundary_and_appends_ellipsis() -> None:
    text = "Người lớn:   uống 500 mg - 1 g\nmỗi 4 - 6 giờ khi cần."

    snippet = make_snippet(text, 24)

    assert snippet == "Người lớn: uống 500 mg…"
    assert len(snippet) <= 24


def test_make_snippet_keeps_word_that_ends_exactly_at_the_cut() -> None:
    assert make_snippet("Paracetamol giảm đau hạ sốt", 12) == "Paracetamol…"


def test_make_snippet_hard_cuts_a_single_long_word() -> None:
    assert make_snippet("Acetylsalicylic", 8) == "Acetyls…"


def test_make_snippet_returns_short_text_unchanged() -> None:
    assert make_snippet("  Hạ sốt  ", 7) == "Hạ sốt"


def test_make_snippet_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        make_snippet("Hạ sốt", 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/test_text.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.domain.shared.text'`.

- [ ] **Step 3: Write the implementation**

`backend/src/pharma_agent/domain/shared/text.py`:

```python
"""Text helpers shared by the corpus and retrieval domains."""

import re
import unicodedata

ELLIPSIS = "…"
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC, unify line endings, drop trailing spaces per line and outer blank lines."""
    unified = (
        unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = [line.rstrip() for line in unified.split("\n")]
    start = 0
    end = len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return "\n".join(lines[start:end])


def _is_table_rule(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "-" in stripped and not stripped.strip("|-: ")


def make_snippet(text: str, max_chars: int) -> str:
    """One-line preview: no table pipes or rules, cut at a word boundary with "…"."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    kept = [line for line in text.splitlines() if not _is_table_rule(line)]
    flat = _WHITESPACE_RE.sub(" ", " ".join(kept).replace("|", " ")).strip()
    if len(flat) <= max_chars:
        return flat
    head = flat[: max_chars - 1]
    if flat[max_chars - 1] != " ":
        boundary = head.rfind(" ")
        if boundary > 0:
            head = head[:boundary]
    return head.rstrip() + ELLIPSIS
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/test_text.py`
Expected: `8 passed`.

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green, `0 diagnostics` (pyrefly may still report the suppression count that exists before this plan).

- [ ] **Step 6: Commit**

```bash
git add src/pharma_agent/domain/shared/text.py tests/domain/test_text.py
git commit -m "feat(domain): add shared text normalization and snippet helpers"
```

The commit message ends with the session attribution trailer.

---

### Task 2: Knowledge bundle models, vector encoding and model slug

**Files:**
- Create: `backend/src/pharma_agent/domain/corpus/__init__.py`
- Create: `backend/src/pharma_agent/domain/corpus/bundle.py`
- Create: `backend/tests/domain/corpus/__init__.py` (empty)
- Test: `backend/tests/domain/corpus/test_bundle_models.py`

**Interfaces:**
- Consumes: `pharma_agent.domain.shared.errors.DomainError(message: str = "", *, code: str | None = None)`.
- Produces (overview §3.2, exact): `BUNDLE_SCHEMA_VERSION`, `DocumentKind`, `BlockKind`, `RetrievalMode`, `SourceInfo`, `DocumentRecord`, `BlockRecord`, `SectionRecord`, `GlossaryEntry`, `ColloquialMappingRecord`, `BundleCollection`, `BundleGenerator`, `BundleFile`, `BundleEmbeddingFile`, `BundleManifest`, `KnowledgeBundle`, `BundleValidationError(problems: Sequence[str])` with `code = "BUNDLE_INVALID"` and `problems: list[str]`, `model_slug(model: str) -> str`, `encode_vector(values: Sequence[float]) -> str`, `decode_vector(data: str, dims: int) -> list[float]` (raises `ValueError` on bad base64 or `len(bytes) != dims * 4`).
- List and dict defaults are written as `Field(default_factory=...)`, which is the overview's `= []` / `= {}` in the repo's style.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/corpus/__init__.py`: empty file.

`backend/tests/domain/corpus/test_bundle_models.py`:

```python
import pytest
from pydantic import ValidationError

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    BundleValidationError,
    DocumentKind,
    DocumentRecord,
    RetrievalMode,
    SectionRecord,
    decode_vector,
    encode_vector,
    model_slug,
)
from pharma_agent.domain.shared.errors import DomainError

SECTION_LINE = (
    '{"key": "drug:paracetamol:tuong-tac-thuoc", "document_key": "drug:paracetamol",'
    ' "heading": "Tương tác thuốc", "context_path": ["Tương tác thuốc"],'
    ' "ordinal": 2, "start_page": 1204, "end_page": 1205, "retrieval": "default",'
    ' "blocks": [{"kind": "table",'
    ' "markdown": "| Thuốc | Hậu quả |\\n| --- | --- |\\n| Warfarin | Tăng INR |",'
    ' "start_page": 1205, "end_page": 1205, "table_key": "tbl-0042"}]}'
)


def test_section_line_parses_enum_strings_from_json() -> None:
    section = SectionRecord.model_validate_json(SECTION_LINE)

    assert section.retrieval is RetrievalMode.DEFAULT
    assert section.blocks[0].kind is BlockKind.TABLE
    assert section.blocks[0].table_key == "tbl-0042"
    assert section.blocks[0].markdown.splitlines()[2] == "| Warfarin | Tăng INR |"


def test_document_line_uses_defaults() -> None:
    document = DocumentRecord.model_validate_json(
        '{"key": "drug:paracetamol", "kind": "drug_monograph", "title": "PARACETAMOL",'
        ' "source": {"title": "Dược thư Quốc gia Việt Nam 2022"}}'
    )

    assert document.kind is DocumentKind.DRUG_MONOGRAPH
    assert document.source.url is None
    assert document.attributes == {}


def test_bundle_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="chunk_id"):
        BlockRecord.model_validate_json(
            '{"kind": "prose", "markdown": "Hạ sốt", "chunk_id": "x"}'
        )


def test_bundle_models_reject_unknown_enum_values_and_loose_types() -> None:
    with pytest.raises(ValidationError, match="kind"):
        DocumentRecord.model_validate_json(
            '{"key": "drug:x", "kind": "brand_page", "title": "X",'
            ' "source": {"title": "Dược thư"}}'
        )
    with pytest.raises(ValidationError, match="start_page"):
        BlockRecord.model_validate_json(
            '{"kind": "prose", "markdown": "Hạ sốt", "start_page": "12"}'
        )


@pytest.mark.parametrize(
    ("model", "slug"),
    [
        ("qwen3-embedding:4b-fp16", "qwen3_embedding_4b_fp16"),
        ("Qwen/Qwen3-Embedding-4B", "qwen_qwen3_embedding_4b"),
        ("fake-embedding-4d", "fake_embedding_4d"),
        ("--", ""),
    ],
)
def test_model_slug(model: str, slug: str) -> None:
    assert model_slug(model) == slug


def test_vector_round_trip_is_float32_little_endian_base64() -> None:
    values = [0.5, -1.25, 3.0, 0.0]

    assert encode_vector([1.0]) == "AACAPw=="
    assert decode_vector(encode_vector(values), 4) == values


def test_decode_vector_rejects_wrong_length_and_bad_base64() -> None:
    with pytest.raises(ValueError, match="expected 16"):
        decode_vector(encode_vector([0.5, 0.5, 0.5]), 4)
    with pytest.raises(ValueError, match="base64"):
        decode_vector("@@@@", 1)


def test_bundle_validation_error_carries_every_problem() -> None:
    problems = [
        "sections.jsonl:3: ordinal: duplicate ordinal 2 in document 'drug:paracetamol'",
        "glossary.json: [1].term: must not be empty",
    ]

    error = BundleValidationError(problems)

    assert isinstance(error, DomainError)
    assert error.code == "BUNDLE_INVALID"
    assert error.problems == problems
    assert "sections.jsonl:3: ordinal" in str(error)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_bundle_models.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.domain.corpus'`.

- [ ] **Step 3: Write the implementation**

`backend/src/pharma_agent/domain/corpus/__init__.py`:

```python
"""Corpus domain: knowledge bundles, identities, enrichment, chunking, hydrate policy.

Public API for seed-pipeline (spec C §4): ``bundle``, ``identity``, ``chunking``,
``enrichment`` and ``hydrate``. Everything else in this package is internal.
"""
```

`backend/src/pharma_agent/domain/corpus/bundle.py`:

```python
"""Knowledge bundle ``knowledge-bundle/v1`` (spec C §5): models, file IO and validation.

Public API: seed-pipeline builds a ``KnowledgeBundle`` and calls ``write_bundle``; the
backend import calls ``read_bundle``. Both run the same content checks.
"""

import base64
import binascii
import re
import struct
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.shared.errors import DomainError

BUNDLE_SCHEMA_VERSION: Final = "knowledge-bundle/v1"

_STRICT = ConfigDict(extra="forbid", strict=True)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


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
    """One row of ``data/resources/term_glossary.json``."""

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

    ``key`` is the curated An Khang slug from ``data/resources/colloquial_mappings.json``
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
    model_config = _STRICT

    manifest: BundleManifest
    documents: list[DocumentRecord]
    sections: list[SectionRecord]
    glossary: list[GlossaryEntry]
    colloquial_mappings: list[ColloquialMappingRecord]
    embeddings: dict[str, dict[str, list[float]]] = Field(default_factory=dict)


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
    try:
        raw = base64.b64decode(data, validate=True)
    except binascii.Error as error:
        raise ValueError(f"vector is not valid base64: {error}") from error
    if len(raw) != dims * 4:
        raise ValueError(f"vector has {len(raw)} bytes, expected {dims * 4} (dims * 4)")
    return list(struct.unpack(f"<{dims}f", raw))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus/test_bundle_models.py`
Expected: `11 passed`.

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green; `tests/architecture/test_layering.py` passes (only `pydantic` and stdlib are imported).

- [ ] **Step 6: Commit**

```bash
git add src/pharma_agent/domain/corpus/__init__.py src/pharma_agent/domain/corpus/bundle.py tests/domain/corpus/__init__.py tests/domain/corpus/test_bundle_models.py
git commit -m "feat(corpus): add knowledge bundle models and vector encoding"
```

The commit message ends with the session attribution trailer.

---

### Task 3: Bundle read/write with full §5.3 validation

**Files:**
- Modify: `backend/src/pharma_agent/domain/corpus/bundle.py` (replace the import/constant block at the top of the file, lines 7–22 as written in Task 2; append IO code after `decode_vector`)
- Test: `backend/tests/domain/corpus/test_bundle_io.py`

**Interfaces:**
- Consumes: Task 2 models.
- Produces: `read_bundle(directory: Path) -> KnowledgeBundle` (raises `BundleValidationError` with every problem found); `write_bundle(bundle: KnowledgeBundle, directory: Path) -> BundleManifest` (runs the same content checks first and writes nothing when they fail; computes `document_count`, `section_count`, `files` (sha256 and bytes of the written bytes) and `embeddings` (one entry per model in `bundle.embeddings`), ignoring whatever those manifest fields held in the input, so P2's fixture generator and P4's `seed bundle embed` can pass placeholders); constants `MANIFEST_FILE`, `DOCUMENTS_FILE`, `SECTIONS_FILE`, `GLOSSARY_FILE`, `COLLOQUIAL_MAPPINGS_FILE`, `EMBEDDINGS_DIR`, `REQUIRED_FILES`.
- Checks performed (spec §5.3 plus what this plan pins): manifest parses (schema version literal); every required file is listed; every listed file exists, is part of the format, matches `bytes` and `sha256`, and is UTF-8; `document_count`/`section_count` match the non-blank line counts; each JSONL line and JSON array item validates against its model (errors carry file, line or index, and field path); document and section keys are non-empty and unique; `section.document_key` exists; `(document_key, ordinal)` is unique; sections have at least one block; block `markdown` is not blank; pages are `>= 1` and `start_page <= end_page` for sections and blocks; glossary terms are non-blank and unique ignoring case; colloquial `section_keys` exist and each section is mapped at most once; each manifest embedding entry has a sluggable model, no duplicate model, `dims >= 1` and `file == "embeddings/<model_slug>.jsonl"` listed in `files`; each embedding line has a 64-hex sha, `dims` equal to the manifest, no duplicate sha and exactly `dims * 4` bytes of base64 vector.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/corpus/test_bundle_io.py`:

```python
import hashlib
import json
from pathlib import Path

import pytest

from pharma_agent.domain.corpus.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BlockKind,
    BlockRecord,
    BundleCollection,
    BundleEmbeddingFile,
    BundleFile,
    BundleGenerator,
    BundleManifest,
    BundleValidationError,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    KnowledgeBundle,
    SectionRecord,
    SourceInfo,
    encode_vector,
    read_bundle,
    write_bundle,
)

EMBEDDINGS_FILE = "embeddings/fake_embedding_4d.jsonl"
SHA_DOSE = hashlib.sha256("PARACETAMOL\n> Liều lượng".encode()).hexdigest()
SHA_TABLE = hashlib.sha256("PARACETAMOL\n> Tương tác thuốc".encode()).hexdigest()
LEAFLET_SECTION = "brand:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440"


def make_bundle() -> KnowledgeBundle:
    paracetamol = DocumentRecord(
        key="drug:paracetamol",
        kind=DocumentKind.DRUG_MONOGRAPH,
        title="PARACETAMOL",
        source=SourceInfo(title="Dược thư Quốc gia Việt Nam 2022"),
    )
    leaflet = DocumentRecord(
        key="leaflet:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440",
        kind=DocumentKind.LEAFLET,
        title="Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)",
        source=SourceInfo(
            title="Tờ hướng dẫn sử dụng",
            url="https://www.nhathuocankhang.com/thuoc-giam-dau-ha-sot/panadol-extra-gsk-150-vien-11440",
        ),
        attributes={"category": "thuoc-giam-dau-ha-sot"},
    )
    return KnowledgeBundle(
        manifest=BundleManifest(
            schema_version=BUNDLE_SCHEMA_VERSION,
            collection=BundleCollection(key="formulary", title="Dược thư Quốc gia"),
            generator=BundleGenerator(
                name="seed-pipeline", version="0.1.0", build_id="build-test"
            ),
            source_digests={"source_pdf_sha256": "0" * 64},
            document_count=0,
            section_count=0,
            files={},
        ),
        documents=[paracetamol, leaflet],
        sections=[
            SectionRecord(
                key="drug:paracetamol:lieu-luong-va-cach-dung",
                document_key=paracetamol.key,
                heading="Liều lượng và cách dùng",
                context_path=["Liều lượng và cách dùng"],
                ordinal=1,
                start_page=1203,
                end_page=1204,
                blocks=[
                    BlockRecord(
                        kind=BlockKind.PROSE,
                        markdown="Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần.",
                    )
                ],
            ),
            SectionRecord(
                key="drug:paracetamol:tuong-tac-thuoc",
                document_key=paracetamol.key,
                heading="Tương tác thuốc",
                context_path=["Tương tác thuốc"],
                ordinal=2,
                blocks=[
                    BlockRecord(
                        kind=BlockKind.TABLE,
                        markdown="| Thuốc | Hậu quả |\n| --- | --- |\n| Warfarin | Tăng INR |",
                        start_page=1205,
                        end_page=1205,
                        table_key="tbl-0042",
                    )
                ],
            ),
            SectionRecord(
                key=LEAFLET_SECTION,
                document_key=leaflet.key,
                heading="Thông tin chi tiết",
                context_path=["Thông tin chi tiết"],
                ordinal=1,
                blocks=[
                    BlockRecord(
                        kind=BlockKind.PROSE,
                        markdown="Panadol đỏ chứa paracetamol 500 mg và cafein 65 mg.",
                    )
                ],
            ),
        ],
        glossary=[
            GlossaryEntry(
                term="INR",
                case_sensitive=True,
                vietnamese_expansions=["tỷ số chuẩn hóa quốc tế"],
                english_expansions=["International Normalized Ratio"],
                category="laboratory",
                confidence="high",
                source="curated",
            )
        ],
        colloquial_mappings=[
            ColloquialMappingRecord(
                key="panadol-extra-gsk-150-vien-11440",
                aliases=["Panadol đỏ", "Panadol hộp đỏ"],
                visual_sign="Hộp màu đỏ, vỉ thuốc màu đỏ",
                product_names=["Panadol Extra GSK"],
                section_keys=[LEAFLET_SECTION],
            )
        ],
        embeddings={
            "fake-embedding-4d": {
                SHA_DOSE: [0.5, -1.25, 3.0, 0.0],
                SHA_TABLE: [1.0, 0.0, 0.0, 0.0],
            }
        },
    )


def rewrite(directory: Path, name: str, content: str) -> None:
    """Replace a bundle file and refresh its manifest digest."""
    path = directory / name
    path.write_text(content, encoding="utf-8")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data = path.read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def edit_manifest(directory: Path, **changes: object) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(changes)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def problems_of(directory: Path) -> list[str]:
    with pytest.raises(BundleValidationError) as caught:
        read_bundle(directory)
    return caught.value.problems


def jsonl(*records: object) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def test_write_then_read_round_trips_with_computed_manifest(tmp_path: Path) -> None:
    bundle = make_bundle()

    manifest = write_bundle(bundle, tmp_path)

    assert manifest.document_count == 2
    assert manifest.section_count == 3
    assert sorted(manifest.files) == [
        "colloquial_mappings.json",
        "documents.jsonl",
        EMBEDDINGS_FILE,
        "glossary.json",
        "sections.jsonl",
    ]
    assert manifest.embeddings == [
        BundleEmbeddingFile(model="fake-embedding-4d", dims=4, file=EMBEDDINGS_FILE)
    ]
    sections_text = (tmp_path / "sections.jsonl").read_text(encoding="utf-8")
    assert len(sections_text.splitlines()) == 3
    assert "Liều lượng và cách dùng" in sections_text
    assert read_bundle(tmp_path) == bundle.model_copy(update={"manifest": manifest})


def test_write_bundle_replaces_placeholder_manifest_fields(tmp_path: Path) -> None:
    bundle = make_bundle()
    placeholder = bundle.manifest.model_copy(
        update={
            "document_count": 99,
            "section_count": 99,
            "files": {"documents.jsonl": BundleFile(sha256="0" * 64, bytes=0)},
            "embeddings": [
                BundleEmbeddingFile(
                    model="stale-model", dims=8, file="embeddings/stale_model.jsonl"
                )
            ],
        }
    )

    manifest = write_bundle(
        bundle.model_copy(update={"manifest": placeholder}), tmp_path
    )

    documents = (tmp_path / "documents.jsonl").read_bytes()
    assert manifest.document_count == 2
    assert manifest.section_count == 3
    assert manifest.files["documents.jsonl"] == BundleFile(
        sha256=hashlib.sha256(documents).hexdigest(), bytes=len(documents)
    )
    assert manifest.embeddings == [
        BundleEmbeddingFile(model="fake-embedding-4d", dims=4, file=EMBEDDINGS_FILE)
    ]
    assert read_bundle(tmp_path).manifest == manifest


def test_write_bundle_without_embeddings_lists_no_embedding_files(
    tmp_path: Path,
) -> None:
    manifest = write_bundle(
        make_bundle().model_copy(update={"embeddings": {}}), tmp_path
    )

    assert manifest.embeddings == []
    assert EMBEDDINGS_FILE not in manifest.files
    assert not (tmp_path / "embeddings").exists()
    assert read_bundle(tmp_path).embeddings == {}


@pytest.mark.parametrize("key", ["", "panadol-extra-gsk-150-vien-11440"])
def test_colloquial_mapping_key_may_be_empty_or_a_slug(
    tmp_path: Path, key: str
) -> None:
    bundle = make_bundle()
    bundle.colloquial_mappings[0].key = key

    write_bundle(bundle, tmp_path)

    assert read_bundle(tmp_path).colloquial_mappings[0].key == key


def test_write_bundle_is_deterministic(tmp_path: Path) -> None:
    first = write_bundle(make_bundle(), tmp_path / "a")
    second = write_bundle(make_bundle(), tmp_path / "b")

    assert first == second
    assert (tmp_path / "a" / EMBEDDINGS_FILE).read_bytes() == (
        tmp_path / "b" / EMBEDDINGS_FILE
    ).read_bytes()


def test_write_bundle_rejects_invalid_content_before_writing(tmp_path: Path) -> None:
    bundle = make_bundle()
    bundle.documents.append(bundle.documents[0])
    bundle.embeddings["fake-embedding-4d"][SHA_TABLE] = [1.0, 0.0, 0.0]

    with pytest.raises(BundleValidationError) as caught:
        write_bundle(bundle, tmp_path)

    assert caught.value.problems == [
        "documents.jsonl:3: key: duplicate document key 'drug:paracetamol' "
        "(first on line 1)",
        "embeddings['fake-embedding-4d']: vectors must share one non-zero length, "
        "got [3, 4]",
    ]
    assert not (tmp_path / "manifest.json").exists()


def test_missing_manifest(tmp_path: Path) -> None:
    assert problems_of(tmp_path) == ["manifest.json: file is missing"]


def test_schema_version_must_match(tmp_path: Path) -> None:
    write_bundle(make_bundle(), tmp_path)
    edit_manifest(tmp_path, schema_version="knowledge-bundle/v2")

    assert problems_of(tmp_path) == [
        "manifest.json: schema_version: Input should be 'knowledge-bundle/v1'"
    ]


def test_file_digests_sizes_and_listing_must_match(tmp_path: Path) -> None:
    manifest_written = write_bundle(make_bundle(), tmp_path)
    written_bytes = manifest_written.files["documents.jsonl"].bytes
    with (tmp_path / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    (tmp_path / "glossary.json").unlink()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    edit_manifest(
        tmp_path,
        files={**manifest["files"], "notes.txt": {"sha256": "0" * 64, "bytes": 0}},
        document_count=5,
    )

    assert problems_of(tmp_path) == [
        f"documents.jsonl: size {written_bytes + 1} bytes does not match manifest "
        f"({written_bytes})",
        "documents.jsonl: sha256 does not match manifest",
        "glossary.json: file is missing",
        "manifest.json: files.notes.txt: not part of knowledge-bundle/v1",
        "manifest.json: document_count: 5 does not match documents.jsonl (2 records)",
    ]


def test_record_fields_are_reported_with_line_and_path(tmp_path: Path) -> None:
    bundle = make_bundle()
    write_bundle(bundle, tmp_path)
    rewrite(
        tmp_path,
        "documents.jsonl",
        jsonl(
            bundle.documents[0].model_dump(mode="json"),
            {
                "key": "drug:ibuprofen",
                "kind": "brand_page",
                "title": "IBUPROFEN",
                "source": {"title": "Dược thư Quốc gia Việt Nam 2022"},
            },
        ),
    )

    assert problems_of(tmp_path) == [
        "documents.jsonl:2: kind: Input should be 'drug_monograph', "
        "'general_monograph' or 'leaflet'",
        "sections.jsonl:3: document_key: unknown document "
        "'leaflet:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440'",
    ]


def test_section_keys_ordinals_blocks_and_pages_are_validated(tmp_path: Path) -> None:
    bundle = make_bundle()
    write_bundle(bundle, tmp_path)
    first = bundle.sections[0].model_dump(mode="json")
    duplicate = {
        **first,
        "start_page": 1206,
        "end_page": 1205,
        "blocks": [{"kind": "prose", "markdown": "   ", "start_page": 0}],
    }
    orphan = {
        **first,
        "key": "drug:ibuprofen:chi-dinh",
        "document_key": "drug:ibuprofen",
        "blocks": [],
    }
    rewrite(tmp_path, "sections.jsonl", jsonl(first, duplicate, orphan))

    assert problems_of(tmp_path) == [
        "sections.jsonl:2: key: duplicate section key "
        "'drug:paracetamol:lieu-luong-va-cach-dung' (first on line 1)",
        "sections.jsonl:2: ordinal: duplicate ordinal 1 in document "
        "'drug:paracetamol' (first on line 1)",
        "sections.jsonl:2: start_page: 1206 is after end_page 1205",
        "sections.jsonl:2: blocks[0].markdown: must not be empty",
        "sections.jsonl:2: blocks[0].start_page: must be >= 1, got 0",
        "sections.jsonl:3: document_key: unknown document 'drug:ibuprofen'",
        "sections.jsonl:3: blocks: must contain at least one block",
        f"colloquial_mappings.json: [0].section_keys[0]: unknown section "
        f"'{LEAFLET_SECTION}'",
    ]


def test_glossary_terms_must_be_present_and_unique_ignoring_case(
    tmp_path: Path,
) -> None:
    write_bundle(make_bundle(), tmp_path)
    rewrite(
        tmp_path,
        "glossary.json",
        json.dumps(
            [
                {"term": "INR", "vietnamese_expansions": ["tỷ số chuẩn hóa quốc tế"]},
                {
                    "term": "inr",
                    "english_expansions": ["International Normalized Ratio"],
                },
                {"term": " ", "aliases": ["x"]},
                {"case_sensitive": True},
            ],
            ensure_ascii=False,
        ),
    )

    assert problems_of(tmp_path) == ["glossary.json: [3].term: Field required"]

    rewrite(
        tmp_path,
        "glossary.json",
        json.dumps(
            [
                {"term": "INR", "vietnamese_expansions": ["tỷ số chuẩn hóa quốc tế"]},
                {
                    "term": "inr",
                    "english_expansions": ["International Normalized Ratio"],
                },
                {"term": " ", "aliases": ["x"]},
            ],
            ensure_ascii=False,
        ),
    )

    assert problems_of(tmp_path) == [
        "glossary.json: [1].term: duplicate term 'inr' (case-insensitive, first at [0])",
        "glossary.json: [2].term: must not be empty",
    ]


def test_a_section_belongs_to_at_most_one_colloquial_mapping(tmp_path: Path) -> None:
    write_bundle(make_bundle(), tmp_path)
    rewrite(
        tmp_path,
        "colloquial_mappings.json",
        json.dumps(
            [
                {
                    "key": "panadol-extra-gsk-150-vien-11440",
                    "section_keys": [LEAFLET_SECTION],
                },
                {
                    "key": "",
                    "product_names": ["Panadol Extra GSK"],
                    "section_keys": [LEAFLET_SECTION],
                },
            ],
            ensure_ascii=False,
        ),
    )

    assert problems_of(tmp_path) == [
        f"colloquial_mappings.json: [1].section_keys[0]: section '{LEAFLET_SECTION}' "
        "is already mapped by [0]"
    ]


def test_embedding_lines_are_validated(tmp_path: Path) -> None:
    write_bundle(make_bundle(), tmp_path)
    rewrite(
        tmp_path,
        EMBEDDINGS_FILE,
        jsonl(
            {
                "embedding_text_sha256": SHA_DOSE,
                "dims": 3,
                "vector": encode_vector([1.0] * 3),
            },
            {
                "embedding_text_sha256": SHA_TABLE,
                "dims": 4,
                "vector": encode_vector([1.0] * 3),
            },
            {
                "embedding_text_sha256": SHA_TABLE,
                "dims": 4,
                "vector": encode_vector([1.0] * 4),
            },
            {
                "embedding_text_sha256": "abc",
                "dims": 4,
                "vector": encode_vector([1.0] * 4),
            },
        ),
    )

    assert problems_of(tmp_path) == [
        f"{EMBEDDINGS_FILE}:1: dims: 3 does not match manifest dims 4",
        f"{EMBEDDINGS_FILE}:2: vector: vector has 12 bytes, expected 16 (dims * 4)",
        f"{EMBEDDINGS_FILE}:3: embedding_text_sha256: duplicate of line 2",
        f"{EMBEDDINGS_FILE}:4: embedding_text_sha256: String should match pattern "
        "'^[0-9a-f]{64}$'",
    ]


def test_manifest_embedding_entries_must_name_their_file(tmp_path: Path) -> None:
    write_bundle(make_bundle(), tmp_path)
    edit_manifest(
        tmp_path,
        embeddings=[
            {"model": "fake-embedding-4d", "dims": 0, "file": "embeddings/other.jsonl"}
        ],
    )

    assert problems_of(tmp_path) == [
        "manifest.json: embeddings[0].dims: must be >= 1, got 0",
        "manifest.json: embeddings[0].file: expected "
        f"'{EMBEDDINGS_FILE}', got 'embeddings/other.jsonl'",
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_bundle_io.py`
Expected: collection error `ImportError: cannot import name 'read_bundle' from 'pharma_agent.domain.corpus.bundle'`.

- [ ] **Step 3: Replace the import and constant block**

In `backend/src/pharma_agent/domain/corpus/bundle.py`, replace everything from `import base64` down to and including `_SLUG_RE = re.compile(r"[^a-z0-9]+")` with:

```python
import base64
import binascii
import hashlib
import json
import re
import struct
from collections.abc import Sequence
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
```

- [ ] **Step 4: Append the IO and validation code**

Append to the end of `backend/src/pharma_agent/domain/corpus/bundle.py` (after `decode_vector`):

```python
class _EmbeddingLine(BaseModel):
    model_config = _STRICT

    embedding_text_sha256: str = Field(pattern=_SHA256_RE.pattern)
    dims: int
    vector: str


_GLOSSARY_ADAPTER = TypeAdapter(list[GlossaryEntry])
_MAPPINGS_ADAPTER = TypeAdapter(list[ColloquialMappingRecord])


def read_bundle(directory: Path) -> KnowledgeBundle:
    """Read and validate a bundle directory; every problem is reported at once."""
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise BundleValidationError([f"{MANIFEST_FILE}: file is missing"])
    try:
        manifest = BundleManifest.model_validate_json(manifest_path.read_bytes())
    except ValidationError as error:
        raise BundleValidationError(
            _validation_problems(MANIFEST_FILE, error)
        ) from error

    problems: list[str] = []
    texts = _read_manifest_files(directory, manifest, problems)
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

    embeddings: dict[str, dict[str, list[float]]] = {}
    for entry in manifest.embeddings:
        text = texts.get(entry.file)
        if text is not None and entry.dims >= 1:
            embeddings[entry.model] = _parse_embeddings(
                entry.file, text, entry.dims, problems
            )

    if problems:
        raise BundleValidationError(problems)
    return KnowledgeBundle(
        manifest=manifest,
        documents=[record for _, record in documents],
        sections=[record for _, record in sections],
        glossary=glossary,
        colloquial_mappings=mappings,
        embeddings=embeddings,
    )


def write_bundle(bundle: KnowledgeBundle, directory: Path) -> BundleManifest:
    """Validate, write every file and return the manifest with computed digests."""
    problems = _content_problems(
        list(enumerate(bundle.documents, start=1)),
        list(enumerate(bundle.sections, start=1)),
        bundle.glossary,
        bundle.colloquial_mappings,
    )
    payloads: dict[str, bytes] = {
        DOCUMENTS_FILE: _jsonl_bytes(bundle.documents),
        SECTIONS_FILE: _jsonl_bytes(bundle.sections),
        GLOSSARY_FILE: _json_list_bytes(bundle.glossary),
        COLLOQUIAL_MAPPINGS_FILE: _json_list_bytes(bundle.colloquial_mappings),
    }
    embedding_files: list[BundleEmbeddingFile] = []
    for model, vectors in sorted(bundle.embeddings.items()):
        where = f"embeddings[{model!r}]"
        file_name = f"{EMBEDDINGS_DIR}/{model_slug(model)}.jsonl"
        lengths = sorted({len(vector) for vector in vectors.values()})
        if not model_slug(model):
            problems.append(f"{where}: model name must contain letters or digits")
        elif file_name in payloads:
            problems.append(f"{where}: model slug collides with another model")
        elif len(lengths) != 1 or lengths[0] < 1:
            problems.append(
                f"{where}: vectors must share one non-zero length, got {lengths}"
            )
        else:
            problems.extend(
                f"{where}: {sha!r} is not a sha256 hex digest"
                for sha in vectors
                if not _SHA256_RE.match(sha)
            )
            lines = [
                json.dumps(
                    {
                        "embedding_text_sha256": sha,
                        "dims": lengths[0],
                        "vector": encode_vector(vector),
                    },
                    separators=(",", ":"),
                )
                for sha, vector in sorted(vectors.items())
            ]
            payloads[file_name] = "".join(f"{line}\n" for line in lines).encode()
            embedding_files.append(
                BundleEmbeddingFile(model=model, dims=lengths[0], file=file_name)
            )
    if problems:
        raise BundleValidationError(problems)

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
        embeddings=embedding_files,
    )
    directory.mkdir(parents=True, exist_ok=True)
    if embedding_files:
        (directory / EMBEDDINGS_DIR).mkdir(exist_ok=True)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    (directory / MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


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


def _read_manifest_files(
    directory: Path, manifest: BundleManifest, problems: list[str]
) -> dict[str, str]:
    expected_embedding_files: set[str] = set()
    seen_models: set[str] = set()
    for index, entry in enumerate(manifest.embeddings):
        where = f"{MANIFEST_FILE}: embeddings[{index}]"
        expected_file = f"{EMBEDDINGS_DIR}/{model_slug(entry.model)}.jsonl"
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
        data = path.read_bytes()
        if len(data) != expected.bytes:
            problems.append(
                f"{name}: size {len(data)} bytes does not match manifest "
                f"({expected.bytes})"
            )
        if hashlib.sha256(data).hexdigest() != expected.sha256:
            problems.append(f"{name}: sha256 does not match manifest")
        try:
            texts[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}: file is not valid UTF-8")
    return texts


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


def _parse_embeddings(
    name: str, text: str, dims: int, problems: list[str]
) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    first_lines: dict[str, int] = {}
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
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
                vectors[sha] = decode_vector(record.vector, dims)
            except ValueError as error:
                problems.append(f"{where}: vector: {error}")
    return vectors


def _jsonl_bytes(records: Sequence[BaseModel]) -> bytes:
    return "".join(f"{record.model_dump_json()}\n" for record in records).encode()


def _json_list_bytes(records: Sequence[BaseModel]) -> bytes:
    payload = [record.model_dump(mode="json") for record in records]
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
```

Notes for the implementer: JSONL is split on `"\n"` only (not `str.splitlines()`), because `model_dump_json` keeps U+2028 and similar characters raw inside strings. `write_bundle` writes records in the given order so the export is deterministic; embedding lines are sorted by sha.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus/test_bundle_io.py tests/domain/corpus/test_bundle_models.py`
Expected: `27 passed`.

- [ ] **Step 6: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/pharma_agent/domain/corpus/bundle.py tests/domain/corpus/test_bundle_io.py
git commit -m "feat(corpus): read and write knowledge bundles with full validation"
```

The commit message ends with the session attribution trailer.

---

### Task 4: Section revision and chunk version identities

**Files:**
- Create: `backend/src/pharma_agent/domain/corpus/identity.py`
- Test: `backend/tests/domain/corpus/test_identity.py`

**Interfaces:**
- Consumes: `BlockRecord` (Task 2), `normalize_text` (Task 1).
- Produces (overview §3.2, exact): `CORPUS_NAMESPACE: uuid.UUID`; `canonical_json(value: object) -> str` (sorted keys, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`, every string passed through `normalize_text`); `sha256_hex(text: str) -> str` (UTF-8, lowercase hex); `section_revision_id(section_key: str, blocks: Sequence[BlockRecord]) -> uuid.UUID` = `uuid5(NS, "section-revision\x1f" + section_key + "\x1f" + sha256_hex(canonical_json([block.model_dump(mode="json") ...])))`; `chunk_version_id(section_key: str, chunk_text: str, embedding_text: str, chunker_version: str) -> uuid.UUID` = `uuid5(NS, "chunk\x1f" + sha256_hex(section_key + "\x1f" + normalize_text(chunk_text) + "\x1f" + normalize_text(embedding_text) + "\x1f" + chunker_version))`.
- `embedding_text_sha256` is `sha256_hex(embedding_text)` on the exact stored text (the cache key must match what the embedder receives); only the id normalizes, so whitespace-only differences never create a new chunk version.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/corpus/test_identity.py`:

```python
import hashlib
import unicodedata
import uuid

from pharma_agent.domain.corpus.bundle import BlockKind, BlockRecord
from pharma_agent.domain.corpus.identity import (
    CORPUS_NAMESPACE,
    canonical_json,
    chunk_version_id,
    section_revision_id,
    sha256_hex,
)

SECTION_KEY = "drug:paracetamol:lieu-luong-va-cach-dung"
BLOCKS = [
    BlockRecord(
        kind=BlockKind.PROSE,
        markdown="Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ, tối đa 4 g/ngày.",
        start_page=1203,
        end_page=1203,
    ),
    BlockRecord(
        kind=BlockKind.TABLE,
        markdown="| Tuổi | Liều |\n| --- | --- |\n| Trẻ em | 10 - 15 mg/kg |",
        start_page=1204,
        end_page=1204,
        table_key="tbl-0007",
    ),
]
CHUNK_TEXT = "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ, tối đa 4 g/ngày."
EMBEDDING_TEXT = f"PARACETAMOL\n> Liều lượng và cách dùng\n\n{CHUNK_TEXT}"


def test_namespace_is_pinned() -> None:
    assert uuid.uuid5(uuid.NAMESPACE_URL, "pharma-agent:corpus:v1") == CORPUS_NAMESPACE


def test_canonical_json_sorts_keys_compacts_and_normalizes_strings() -> None:
    value = {
        "b": [unicodedata.normalize("NFD", "Liều") + "  \r\n", 2.5],
        "a": {"z": 1, "y": None},
    }

    assert canonical_json(value) == '{"a":{"y":null,"z":1},"b":["Liều",2.5]}'


def test_sha256_hex_hashes_utf8_text() -> None:
    assert sha256_hex("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_hex("Liều") == hashlib.sha256("Liều".encode()).hexdigest()


def test_section_revision_id_follows_the_spec_formula() -> None:
    blocks_json = canonical_json([block.model_dump(mode="json") for block in BLOCKS])
    expected = uuid.uuid5(
        CORPUS_NAMESPACE,
        "section-revision\x1f" + SECTION_KEY + "\x1f" + sha256_hex(blocks_json),
    )

    assert section_revision_id(SECTION_KEY, BLOCKS) == expected


def test_section_revision_id_is_deterministic_and_content_sensitive() -> None:
    base = section_revision_id(SECTION_KEY, BLOCKS)
    with_crlf = [
        BLOCKS[0].model_copy(update={"markdown": BLOCKS[0].markdown + "  \r\n"}),
        BLOCKS[1],
    ]
    with_other_page = [BLOCKS[0].model_copy(update={"start_page": 1202}), BLOCKS[1]]

    assert section_revision_id(SECTION_KEY, [b.model_copy() for b in BLOCKS]) == base
    assert section_revision_id(SECTION_KEY, with_crlf) == base
    assert section_revision_id(SECTION_KEY, with_other_page) != base
    assert section_revision_id(SECTION_KEY, list(reversed(BLOCKS))) != base
    assert section_revision_id("drug:paracetamol:chong-chi-dinh", BLOCKS) != base


def test_chunk_version_id_follows_the_spec_formula() -> None:
    digest = sha256_hex(
        "\x1f".join((SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1"))
    )

    assert chunk_version_id(
        SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1"
    ) == uuid.uuid5(CORPUS_NAMESPACE, "chunk\x1f" + digest)


def test_chunk_version_id_changes_with_every_input_but_not_with_whitespace() -> None:
    base = chunk_version_id(SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1")
    enriched = (
        f"{EMBEDDING_TEXT}\n\nThuật ngữ: G6PD = glucose-6-phosphate dehydrogenase"
    )

    assert (
        chunk_version_id(SECTION_KEY, CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v2") != base
    )
    assert chunk_version_id(SECTION_KEY, CHUNK_TEXT, enriched, "chunker-v1") != base
    assert (
        chunk_version_id(
            SECTION_KEY, f"{CHUNK_TEXT} Trẻ em", EMBEDDING_TEXT, "chunker-v1"
        )
        != base
    )
    assert (
        chunk_version_id(
            "drug:ibuprofen:lieu-dung", CHUNK_TEXT, EMBEDDING_TEXT, "chunker-v1"
        )
        != base
    )
    assert (
        chunk_version_id(
            SECTION_KEY,
            f"{CHUNK_TEXT}  \n",
            EMBEDDING_TEXT.replace("\n", "\r\n"),
            "chunker-v1",
        )
        == base
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_identity.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.domain.corpus.identity'`.

- [ ] **Step 3: Write the implementation**

`backend/src/pharma_agent/domain/corpus/identity.py`:

```python
"""Deterministic identities for section revisions and chunk versions (spec C §6.1).

Public API: seed-pipeline uses ``sha256_hex`` to key precomputed embeddings.
"""

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence

from pharma_agent.domain.corpus.bundle import BlockRecord
from pharma_agent.domain.shared.text import normalize_text

CORPUS_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "pharma-agent:corpus:v1")
_SEPARATOR = "\x1f"


def _normalized(value: object) -> object:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalized(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Sorted keys, no insignificant whitespace, every string normalized."""
    return json.dumps(
        _normalized(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def section_revision_id(section_key: str, blocks: Sequence[BlockRecord]) -> uuid.UUID:
    blocks_digest = sha256_hex(
        canonical_json([block.model_dump(mode="json") for block in blocks])
    )
    return uuid.uuid5(
        CORPUS_NAMESPACE,
        _SEPARATOR.join(("section-revision", section_key, blocks_digest)),
    )


def chunk_version_id(
    section_key: str, chunk_text: str, embedding_text: str, chunker_version: str
) -> uuid.UUID:
    digest = sha256_hex(
        _SEPARATOR.join(
            (
                section_key,
                normalize_text(chunk_text),
                normalize_text(embedding_text),
                chunker_version,
            )
        )
    )
    return uuid.uuid5(CORPUS_NAMESPACE, "chunk" + _SEPARATOR + digest)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus/test_identity.py`
Expected: `7 passed`.

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/pharma_agent/domain/corpus/identity.py tests/domain/corpus/test_identity.py
git commit -m "feat(corpus): add deterministic section revision and chunk version ids"
```

The commit message ends with the session attribution trailer.

---

### Task 5: Hydrate policy

**Files:**
- Create: `backend/src/pharma_agent/domain/corpus/hydrate.py`
- Test: `backend/tests/domain/corpus/test_hydrate.py`

**Interfaces:**
- Consumes: `SectionRecord`, `RetrievalMode` (Task 2); `pharma_agent.domain.retrieval.models.HydrateStrategy` (`FULL_SECTION`, `CHUNK_WINDOW`, `SEARCH_ONLY`).
- Produces (overview §3.2, exact; public API for seed-pipeline per spec C §4, P4 imports `hydrate_strategy_for`): `FULL_SECTION_MAX_CHARS = 16000`; `section_char_count(section: SectionRecord) -> int` = `len("\n\n".join(block.markdown for block in section.blocks))`; `hydrate_strategy_for(section: SectionRecord) -> HydrateStrategy`.
- Port of `hydrate_strategy_for_section` (corpus-pipeline `build_canonical_rag.py`): the hard-coded `BRAND_INDEX_SECTION_ID` check becomes `retrieval is RetrievalMode.INDEX_ONLY`; the old `len(section.text)` is the same join of non-blank block texts, and bundle validation (Task 3) guarantees every block is non-blank. P2 stores `section_char_count` in `section_revisions.char_count` and the strategy in `release_chunks.hydrate_strategy`.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/corpus/test_hydrate.py`:

```python
from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    RetrievalMode,
    SectionRecord,
)
from pharma_agent.domain.corpus.hydrate import (
    FULL_SECTION_MAX_CHARS,
    hydrate_strategy_for,
    section_char_count,
)
from pharma_agent.domain.retrieval.models import HydrateStrategy

SENTENCE = "Paracetamol 500 mg. "


def make_section(
    *markdowns: str, retrieval: RetrievalMode = RetrievalMode.DEFAULT
) -> SectionRecord:
    return SectionRecord(
        key="drug:paracetamol:thong-tin-chung",
        document_key="drug:paracetamol",
        heading="Thông tin chung",
        context_path=["Thông tin chung"],
        ordinal=1,
        retrieval=retrieval,
        blocks=[
            BlockRecord(kind=BlockKind.PROSE, markdown=markdown)
            for markdown in markdowns
        ],
    )


def test_section_char_count_joins_blocks_with_a_blank_line() -> None:
    section = make_section("Hạ sốt", "Giảm đau")

    assert section_char_count(section) == len("Hạ sốt\n\nGiảm đau")


def test_full_section_up_to_the_limit() -> None:
    half = (SENTENCE * 400)[:7999]
    section = make_section(half, half)

    assert FULL_SECTION_MAX_CHARS == 16000
    assert section_char_count(section) == 16000
    assert hydrate_strategy_for(section) is HydrateStrategy.FULL_SECTION


def test_chunk_window_above_the_limit() -> None:
    section = make_section(SENTENCE * 400, (SENTENCE * 400)[:7999])

    assert section_char_count(section) == 16001
    assert hydrate_strategy_for(section) is HydrateStrategy.CHUNK_WINDOW


def test_index_only_sections_are_search_only_whatever_their_length() -> None:
    short = make_section(
        "- **Efferalgan**: Paracetamol", retrieval=RetrievalMode.INDEX_ONLY
    )
    long = make_section(SENTENCE * 1000, retrieval=RetrievalMode.INDEX_ONLY)

    assert hydrate_strategy_for(short) is HydrateStrategy.SEARCH_ONLY
    assert hydrate_strategy_for(long) is HydrateStrategy.SEARCH_ONLY
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_hydrate.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.domain.corpus.hydrate'`.

- [ ] **Step 3: Write the implementation**

`backend/src/pharma_agent/domain/corpus/hydrate.py`:

```python
"""Hydrate policy for a section revision (spec C §7.2).

Public API: seed-pipeline imports ``hydrate_strategy_for``.
"""

from pharma_agent.domain.corpus.bundle import RetrievalMode, SectionRecord
from pharma_agent.domain.retrieval.models import HydrateStrategy

FULL_SECTION_MAX_CHARS = 16000


def section_char_count(section: SectionRecord) -> int:
    return len("\n\n".join(block.markdown for block in section.blocks))


def hydrate_strategy_for(section: SectionRecord) -> HydrateStrategy:
    if section.retrieval is RetrievalMode.INDEX_ONLY:
        return HydrateStrategy.SEARCH_ONLY
    if section_char_count(section) > FULL_SECTION_MAX_CHARS:
        return HydrateStrategy.CHUNK_WINDOW
    return HydrateStrategy.FULL_SECTION
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus/test_hydrate.py`
Expected: `4 passed`.

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/pharma_agent/domain/corpus/hydrate.py tests/domain/corpus/test_hydrate.py
git commit -m "feat(corpus): add hydrate policy for section revisions"
```

The commit message ends with the session attribution trailer.

---

### Task 6: Glossary terms, colloquial mapping and embedding text

**Files:**
- Create: `backend/src/pharma_agent/domain/corpus/enrichment.py`
- Test: `backend/tests/domain/corpus/test_enrichment.py`

**Interfaces:**
- Consumes: `GlossaryEntry`, `ColloquialMappingRecord` (Task 2); `TermAnnotation(term: str, vi: list[str], en: list[str])` and `ColloquialMapping(key: str = "", aliases: list[str], visual_sign: str = "", product_names: list[str])` from `pharma_agent.domain.retrieval.models` (unchanged).
- Produces (overview §3.2, exact): `build_context_header(title: str, context_path: Sequence[str]) -> str`; `detect_terms(text: str, glossary: Sequence[GlossaryEntry]) -> list[TermAnnotation]`; `mapping_for_section(section_key: str, mappings: Sequence[ColloquialMappingRecord]) -> ColloquialMapping | None`; `compose_embedding_text(*, context_header: str, chunk_text: str, colloquial: ColloquialMapping | None, terms: Sequence[TermAnnotation]) -> str`; constants `TERM_ANNOTATION_LIMIT = 8`, `COLLOQUIAL_ALIAS_LABEL = "Tên gọi khác"`, `VISUAL_SIGN_LABEL = "Dấu hiệu nhận biết"`, `TERM_SEARCH_PREFIX = "Thuật ngữ: "`.
- Port map (corpus-pipeline → this module), behaviour identical:
  - `preprocess_rag_corpus.build_context_header` + `payload_layers.clean_context_header` → `build_context_header` (join `title` and non-empty path items with `"\n> "`, strip each line, drop blank lines and lines starting with the colloquial labels).
  - `term_enrichment.detect_term_enrichments` + `load_term_glossary` list normalization + `payload_layers.term_annotations_from_enrichments` → `detect_terms` (first match per entry over `[term, *aliases]`, word boundary `[A-Za-zÀ-ỹ0-9]`, sort by position then term, keep the first 8 matches, then drop entries without expansions and case-insensitive duplicates).
  - `payload_layers.compact_colloquial_mapping` → `mapping_for_section` (the record whose `section_keys` contain the section; `None` when nothing is left after compaction).
  - `build_rag_metadata._embedding_text` + `payload_layers.format_colloquial_mapping` + `term_enrichment.format_term_search_text` → `compose_embedding_text`. The old function took enrichments; taking `TermAnnotation`s gives the same text because annotations keep every enrichment that has an expansion and Task 3 rejects case-insensitive duplicate glossary terms.
- `GlossaryEntry.case_sensitive` defaults to `False` (overview) where the old loader defaulted to `True`; all 71 entries in `data/resources/term_glossary.json` set it explicitly, so output is unchanged.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/corpus/test_enrichment.py`:

```python
from pharma_agent.domain.corpus.bundle import ColloquialMappingRecord, GlossaryEntry
from pharma_agent.domain.corpus.enrichment import (
    build_context_header,
    compose_embedding_text,
    detect_terms,
    mapping_for_section,
)
from pharma_agent.domain.retrieval.models import ColloquialMapping, TermAnnotation

ADR = GlossaryEntry(
    term="ADR",
    case_sensitive=True,
    vietnamese_expansions=["tác dụng không mong muốn", "phản ứng có hại của thuốc"],
    english_expansions=["Adverse Drug Reactions"],
    aliases=["adverse drug reaction"],
    category="safety",
    confidence="high",
    source="curated",
)
NSAID = GlossaryEntry(
    term="NSAID",
    case_sensitive=True,
    vietnamese_expansions=[
        "thuốc chống viêm không steroid",
        "thuốc kháng viêm không steroid",
    ],
    english_expansions=["Nonsteroidal Anti-inflammatory Drug"],
    aliases=["NSAIDs"],
    category="drug_class",
    confidence="high",
    source="curated",
)
G6PD = GlossaryEntry(
    term="G6PD",
    case_sensitive=True,
    vietnamese_expansions=["glucose-6-phosphate dehydrogenase"],
    english_expansions=["glucose-6-phosphate dehydrogenase"],
    aliases=["thiếu men G6PD", "thiếu enzym G6PD"],
    category="laboratory",
    confidence="high",
    source="curated",
)
GLOSSARY = [ADR, NSAID, G6PD]
ADR_ANNOTATION = TermAnnotation(
    term="ADR",
    vi=["tác dụng không mong muốn", "phản ứng có hại của thuốc"],
    en=["Adverse Drug Reactions"],
)
NSAID_ANNOTATION = TermAnnotation(
    term="NSAID",
    vi=["thuốc chống viêm không steroid", "thuốc kháng viêm không steroid"],
    en=["Nonsteroidal Anti-inflammatory Drug"],
)
G6PD_ANNOTATION = TermAnnotation(
    term="G6PD",
    vi=["glucose-6-phosphate dehydrogenase"],
    en=["glucose-6-phosphate dehydrogenase"],
)
LEAFLET_SECTION = "brand:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440"
LEAFLET_HEADER = (
    "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n> Thông tin chi tiết"
)
PANADOL = ColloquialMappingRecord(
    key="panadol-extra-gsk-150-vien-11440",
    aliases=["Panadol đỏ", " Panadol  extra đỏ ", "panadol đỏ", "Panadol hộp đỏ"],
    visual_sign=" Hộp màu đỏ, vỉ thuốc màu đỏ ",
    product_names=["Panadol Extra GSK giảm đau, hạ sốt", "Panadol Extra GSK"],
    section_keys=[LEAFLET_SECTION],
)


def test_build_context_header_joins_title_and_non_empty_path_items() -> None:
    assert (
        build_context_header("PARACETAMOL", ["Liều lượng và cách dùng"])
        == "PARACETAMOL\n> Liều lượng và cách dùng"
    )
    assert build_context_header(
        "PHỤ LỤC 2. PHA THÊM THUỐC TIÊM VÀO DỊCH TRUYỀN TĨNH MẠCH",
        ["7. Các thuốc đưa vào bằng đường truyền tĩnh mạch", ""],
    ) == (
        "PHỤ LỤC 2. PHA THÊM THUỐC TIÊM VÀO DỊCH TRUYỀN TĨNH MẠCH\n"
        "> 7. Các thuốc đưa vào bằng đường truyền tĩnh mạch"
    )
    assert build_context_header("", []) == ""


def test_build_context_header_drops_colloquial_label_lines_and_blank_lines() -> None:
    title = "Panadol Extra GSK\nTên gọi khác: Panadol đỏ\n\ndấu hiệu nhận biết: Hộp đỏ"

    assert (
        build_context_header(title, ["Thông tin chi tiết"])
        == "Panadol Extra GSK\n> Thông tin chi tiết"
    )


def test_detect_terms_orders_by_first_match_and_matches_aliases() -> None:
    text = (
        "PARACETAMOL\n> Chống chỉ định\n\n"
        "Người thiếu men G6PD, người đang dùng NSAIDs; theo dõi ADR."
    )

    assert detect_terms(text, GLOSSARY) == [
        G6PD_ANNOTATION,
        NSAID_ANNOTATION,
        ADR_ANNOTATION,
    ]


def test_detect_terms_respects_case_and_vietnamese_word_boundaries() -> None:
    cmax = GlossaryEntry(
        term="CMAX", case_sensitive=False, english_expansions=["peak concentration"]
    )
    text = "adr, ADRs, NSAIDđ và nsaids không khớp; Cmax đạt sau 1 giờ."

    assert detect_terms(text, [*GLOSSARY, cmax]) == [
        TermAnnotation(term="CMAX", vi=[], en=["peak concentration"])
    ]


def test_detect_terms_limit_counts_matches_without_expansions() -> None:
    glossary = [
        GlossaryEntry(
            term=f"XT{index}",
            case_sensitive=True,
            english_expansions=[] if index < 2 else [f"expansion {index}"],
            aliases=[f"xt-{index}"],
        )
        for index in range(10)
    ]
    text = " ".join(f"XT{index}" for index in range(10))

    assert [annotation.term for annotation in detect_terms(text, glossary)] == [
        "XT2",
        "XT3",
        "XT4",
        "XT5",
        "XT6",
        "XT7",
    ]


def test_detect_terms_normalizes_expansions() -> None:
    inr = GlossaryEntry(
        term=" INR ",
        case_sensitive=True,
        vietnamese_expansions=[
            "tỷ số  chuẩn hóa\nquốc tế",
            "Tỷ số chuẩn hóa quốc tế",
            " ",
        ],
        english_expansions=["International Normalized Ratio"],
    )

    assert detect_terms("Theo dõi INR hằng tuần.", [inr]) == [
        TermAnnotation(
            term="INR",
            vi=["tỷ số chuẩn hóa quốc tế"],
            en=["International Normalized Ratio"],
        )
    ]


def test_mapping_for_section_compacts_the_record() -> None:
    assert mapping_for_section(LEAFLET_SECTION, [PANADOL]) == ColloquialMapping(
        key="panadol-extra-gsk-150-vien-11440",
        aliases=["Panadol đỏ", "Panadol extra đỏ", "Panadol hộp đỏ"],
        visual_sign="Hộp màu đỏ, vỉ thuốc màu đỏ",
        product_names=["Panadol Extra GSK giảm đau, hạ sốt", "Panadol Extra GSK"],
    )
    assert (
        mapping_for_section("drug:paracetamol:lieu-luong-va-cach-dung", [PANADOL])
        is None
    )


def test_mapping_for_section_keeps_product_names_without_a_curated_key() -> None:
    hapacol = "brand:ankhang:thuoc-giam-dau-ha-sot:hapacol-250-dhg"
    efferalgan = "brand:ankhang:thuoc-giam-dau-ha-sot:efferalgan-500mg"
    records = [
        ColloquialMappingRecord(
            key="",
            product_names=["Hapacol 250 DHG", "hapacol 250 dhg"],
            section_keys=[hapacol],
        ),
        ColloquialMappingRecord(key="", section_keys=[efferalgan]),
    ]

    assert mapping_for_section(hapacol, records) == ColloquialMapping(
        product_names=["Hapacol 250 DHG"]
    )
    assert mapping_for_section(efferalgan, records) is None


def test_compose_embedding_text_orders_header_colloquial_chunk_and_terms() -> None:
    chunk = "Panadol đỏ chứa paracetamol 500 mg. Thận trọng khi dùng cùng NSAID."
    mapping = mapping_for_section(LEAFLET_SECTION, [PANADOL])
    terms = detect_terms(f"{LEAFLET_HEADER}\n\n{chunk}", GLOSSARY)

    assert compose_embedding_text(
        context_header=LEAFLET_HEADER, chunk_text=chunk, colloquial=mapping, terms=terms
    ) == (
        f"{LEAFLET_HEADER}\n\n"
        "Tên gọi khác: Panadol extra đỏ, Panadol hộp đỏ\n"
        "Dấu hiệu nhận biết: Hộp màu đỏ, vỉ thuốc màu đỏ\n\n"
        f"{chunk}\n\n"
        "Thuật ngữ: NSAID = thuốc chống viêm không steroid; "
        "thuốc kháng viêm không steroid; Nonsteroidal Anti-inflammatory Drug"
    )


def test_compose_embedding_text_without_header_colloquial_or_terms() -> None:
    assert (
        compose_embedding_text(
            context_header="", chunk_text="Hạ sốt.", colloquial=None, terms=[]
        )
        == "Hạ sốt."
    )


def test_compose_embedding_text_skips_visible_names_and_existing_term_block() -> None:
    mapping = ColloquialMapping(aliases=["Panadol đỏ"], visual_sign="hộp màu đỏ")
    chunk = (
        "Panadol đỏ có HỘP MÀU ĐỎ.\n\n"
        "Thuật ngữ: G6PD = glucose-6-phosphate dehydrogenase"
    )

    assert (
        compose_embedding_text(
            context_header="PHỤ LỤC 1. THUẬT NGỮ",
            chunk_text=chunk,
            colloquial=mapping,
            terms=[G6PD_ANNOTATION],
        )
        == f"PHỤ LỤC 1. THUẬT NGỮ\n\n{chunk}"
    )


def test_compose_embedding_text_dedupes_expansions_and_repeated_terms() -> None:
    terms = [
        G6PD_ANNOTATION,
        ADR_ANNOTATION,
        TermAnnotation(term="ADR", vi=["phản ứng phụ"]),
    ]

    text = compose_embedding_text(
        context_header="PARACETAMOL\n> Chống chỉ định",
        chunk_text="Thiếu men G6PD; ADR trên gan.",
        colloquial=None,
        terms=terms,
    )

    assert text.endswith(
        "\n\nThuật ngữ: G6PD = glucose-6-phosphate dehydrogenase | "
        "ADR = tác dụng không mong muốn; phản ứng có hại của thuốc; "
        "Adverse Drug Reactions"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_enrichment.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.domain.corpus.enrichment'`.

- [ ] **Step 3: Write the implementation**

`backend/src/pharma_agent/domain/corpus/enrichment.py`:

```python
"""Glossary terms, colloquial mappings and embedding text (spec C §7.1).

Public API. Ported from corpus-pipeline ``term_enrichment.py``, ``payload_layers.py``
and ``build_rag_metadata.py``; output must stay identical to the old chunks.jsonl.
"""

import re
from collections.abc import Iterable, Sequence
from functools import cache

from pharma_agent.domain.corpus.bundle import ColloquialMappingRecord, GlossaryEntry
from pharma_agent.domain.retrieval.models import ColloquialMapping, TermAnnotation

TERM_ANNOTATION_LIMIT = 8
COLLOQUIAL_ALIAS_LABEL = "Tên gọi khác"
VISUAL_SIGN_LABEL = "Dấu hiệu nhận biết"
TERM_SEARCH_PREFIX = "Thuật ngữ: "
_WORD_CHARS = "A-Za-zÀ-ỹ0-9"
_WHITESPACE_RE = re.compile(r"\s+")


def _normalized_list(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _WHITESPACE_RE.sub(" ", raw).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        output.append(text)
    return output


def build_context_header(title: str, context_path: Sequence[str]) -> str:
    """``title\\n> path...`` without blank lines or colloquial label lines."""
    raw = "\n> ".join(line for line in (title, *context_path) if line)
    labels = (
        f"{COLLOQUIAL_ALIAS_LABEL}:".casefold(),
        f"{VISUAL_SIGN_LABEL}:".casefold(),
    )
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().casefold().startswith(labels)
    ]
    return "\n".join(lines).strip()


@cache
def _term_pattern(match_text: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(
        rf"(?<![{_WORD_CHARS}]){re.escape(match_text)}(?![{_WORD_CHARS}])", flags
    )


def detect_terms(text: str, glossary: Sequence[GlossaryEntry]) -> list[TermAnnotation]:
    """Glossary terms found in ``text``, by first occurrence, at most 8 matches."""
    matches: list[tuple[int, str, GlossaryEntry]] = []
    seen_terms: set[str] = set()
    for entry in glossary:
        term = entry.term.strip()
        best: re.Match[str] | None = None
        for match_text in _normalized_list([term, *entry.aliases]):
            match = _term_pattern(match_text, entry.case_sensitive).search(text)
            if match is not None and (best is None or match.start() < best.start()):
                best = match
        if best is None or term in seen_terms:
            continue
        seen_terms.add(term)
        matches.append((best.start(), term, entry))
    matches.sort(key=lambda item: (item[0], item[1]))

    annotations: list[TermAnnotation] = []
    seen_annotations: set[str] = set()
    for _, term, entry in matches[:TERM_ANNOTATION_LIMIT]:
        if not term or term.casefold() in seen_annotations:
            continue
        seen_annotations.add(term.casefold())
        vi = _normalized_list(entry.vietnamese_expansions)
        en = _normalized_list(entry.english_expansions)
        if vi or en:
            annotations.append(TermAnnotation(term=term, vi=vi, en=en))
    return annotations


def mapping_for_section(
    section_key: str, mappings: Sequence[ColloquialMappingRecord]
) -> ColloquialMapping | None:
    for record in mappings:
        if section_key not in record.section_keys:
            continue
        mapping = ColloquialMapping(
            key=record.key.strip(),
            aliases=_normalized_list(record.aliases),
            visual_sign=record.visual_sign.strip(),
            product_names=_normalized_list(record.product_names),
        )
        has_content = (
            mapping.key
            or mapping.aliases
            or mapping.visual_sign
            or mapping.product_names
        )
        return mapping if has_content else None
    return None


def _colloquial_text(mapping: ColloquialMapping, visible_text: str) -> str:
    visible = visible_text.casefold()
    lines: list[str] = []
    aliases = [
        alias
        for alias in _normalized_list(mapping.aliases)
        if alias.casefold() not in visible
    ]
    if aliases:
        lines.append(f"{COLLOQUIAL_ALIAS_LABEL}: {', '.join(aliases)}")
    visual_sign = mapping.visual_sign.strip()
    if visual_sign and visual_sign.casefold() not in visible:
        lines.append(f"{VISUAL_SIGN_LABEL}: {visual_sign}")
    return "\n".join(lines)


def _term_search_text(terms: Sequence[TermAnnotation]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for annotation in terms:
        term = annotation.term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        expansion = "; ".join(_normalized_list([*annotation.vi, *annotation.en]))
        if expansion:
            parts.append(f"{term} = {expansion}")
        if len(seen) >= TERM_ANNOTATION_LIMIT:
            break
    return TERM_SEARCH_PREFIX + " | ".join(parts) if parts else ""


def compose_embedding_text(
    *,
    context_header: str,
    chunk_text: str,
    colloquial: ColloquialMapping | None,
    terms: Sequence[TermAnnotation],
) -> str:
    """Header, unseen colloquial names, chunk, then term expansions."""
    visible_text = f"{context_header}\n\n{chunk_text}" if context_header else chunk_text
    parts: list[str] = []
    if context_header:
        parts.append(context_header)
    if colloquial is not None:
        colloquial_text = _colloquial_text(colloquial, visible_text)
        if colloquial_text:
            parts.append(colloquial_text)
    parts.append(chunk_text)
    term_text = _term_search_text(terms)
    if term_text and "thuật ngữ:" not in "\n\n".join(parts).casefold():
        parts.append(term_text)
    return "\n\n".join(parts).strip()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus/test_enrichment.py`
Expected: `12 passed`.

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/pharma_agent/domain/corpus/enrichment.py tests/domain/corpus/test_enrichment.py
git commit -m "feat(corpus): port glossary, colloquial and embedding text enrichment"
```

The commit message ends with the session attribution trailer.

---

### Task 7: Ported splitters

**Files:**
- Create: `backend/src/pharma_agent/domain/corpus/chunking.py` (constants and splitters; Task 8 adds `ChunkDraft` and `chunk_section`)
- Test: `backend/tests/domain/corpus/test_splitters.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CHUNKER_VERSION = "chunker-v1"`, `MAX_CHUNK_CHARS = 3000` (overview §3.2); `split_long_text(text: str, max_chars: int) -> list[str]`, `split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]`, `split_table_markdown(markdown: str, max_chars: int) -> list[str]`, `split_lines_without_breaking_entries(text: str, max_chars: int) -> list[str]`.
- Faithful ports, same names as the originals so they can be diffed: `split_long_text` and `split_oversized_paragraph` from `corpus-pipeline/.../processing/preprocess_rag_corpus.py`; `split_table_markdown`, `is_markdown_separator_line` (private `_is_separator_row` here) and `split_lines_without_breaking_entries` from `.../canonical/build_canonical_rag.py`. `split_brand_index_text` was only `split_lines_without_breaking_entries` with an explicit empty return, so it is not ported separately. The expected values in the test were produced by running the old functions on the same inputs, and a 20 000-input random comparison of old and new functions (79 524 calls) found no difference while this plan was written.

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/corpus/test_splitters.py`:

```python
import pytest

from pharma_agent.domain.corpus.chunking import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    split_lines_without_breaking_entries,
    split_long_text,
    split_oversized_paragraph,
    split_table_markdown,
)

PHARMACOKINETICS = (
    "Paracetamol được hấp thu nhanh qua đường tiêu hóa. Nồng độ đỉnh trong huyết "
    "tương đạt sau 30 - 60 phút; thời gian bán thải khoảng 2 giờ và kéo dài khi "
    "suy gan."
)
PHARMACOKINETICS_PARTS = [
    "Paracetamol được hấp thu nhanh qua đường tiêu hóa.",
    "Nồng độ đỉnh trong huyết tương đạt sau 30 - 60 phút;",
    "thời gian bán thải khoảng 2 giờ và kéo dài khi suy gan.",
]
TABLE_HEADER = "| Thuốc phối hợp | Hậu quả |\n| --- | --- |"


def test_chunker_constants() -> None:
    assert CHUNKER_VERSION == "chunker-v1"
    assert MAX_CHUNK_CHARS == 3000


def test_split_long_text_returns_fitting_text_unchanged() -> None:
    text = "Người lớn: uống 500 mg.\n \nTrẻ em: 10 mg/kg."

    assert split_long_text(text, 100) == [text]


def test_split_long_text_packs_paragraphs_and_normalizes_breaks() -> None:
    text = (
        "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần.\n\n"
        "Trẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ.\n\n\n"
        "Không dùng quá 5 lần trong 24 giờ."
    )

    assert split_long_text(text, 90) == [
        "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần.\n\n"
        "Trẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ.",
        "Không dùng quá 5 lần trong 24 giờ.",
    ]


def test_split_long_text_isolates_an_oversized_paragraph() -> None:
    text = f"Hạ sốt.\n\n{PHARMACOKINETICS}\n\nGiảm đau."

    assert split_long_text(text, 80) == [
        "Hạ sốt.",
        *PHARMACOKINETICS_PARTS,
        "Giảm đau.",
    ]


@pytest.mark.parametrize(
    ("paragraph", "max_chars", "parts"),
    [
        (PHARMACOKINETICS, 80, PHARMACOKINETICS_PARTS),
        (
            "Liều dùng cho người lớn\nuống mỗi lần một viên sau ăn\n"
            "không quá tám viên mỗi ngày",
            40,
            [
                "Liều dùng cho người lớn",
                "uống mỗi lần một viên sau ăn",
                "không quá tám viên mỗi ngày",
            ],
        ),
        (
            "Acetylcystein hòa tan trong nước uống sau bữa ăn tối",
            20,
            ["Acetylcystein hòa", "tan trong nước uống", "sau bữa ăn tối"],
        ),
        (
            "Natriphenylbutyratglycerolphenylbutyrat",
            10,
            ["Natripheny", "lbutyratgl", "ycerolphen", "ylbutyrat"],
        ),
    ],
    ids=["sentence-end", "newline", "whitespace", "hard-cut"],
)
def test_split_oversized_paragraph_prefers_sentence_then_newline_then_space(
    paragraph: str, max_chars: int, parts: list[str]
) -> None:
    assert split_oversized_paragraph(paragraph, max_chars) == parts


def test_split_table_markdown_strips_a_fitting_table() -> None:
    table = "\n| Thuốc | Hậu quả |\n| --- | --- |\n| Warfarin | Tăng INR |\n"

    assert split_table_markdown(table, 100) == [table.strip()]


def test_split_table_markdown_repeats_header_and_separator_in_every_part() -> None:
    table = (
        f"{TABLE_HEADER}\n"
        "| Warfarin | Tăng INR |\n"
        "|  Rượu | Tăng độc tính trên gan |\n\n"
        "| Isoniazid | Tăng nguy cơ độc gan |\n"
        "| Cholestyramin | Giảm hấp thu |"
    )

    assert split_table_markdown(table, 70) == [
        f"{TABLE_HEADER}\n| Warfarin | Tăng INR |",
        f"{TABLE_HEADER}\n|  Rượu | Tăng độc tính trên gan |",
        f"{TABLE_HEADER}\n| Isoniazid | Tăng nguy cơ độc gan |",
        f"{TABLE_HEADER}\n| Cholestyramin | Giảm hấp thu |",
    ]


def test_split_table_markdown_without_separator_falls_back_to_prose() -> None:
    table = (
        "| Thuốc | Hậu quả |\n| Warfarin | Tăng INR khi dùng kéo dài |\n\n"
        "| Rượu | Tăng độc tính trên gan |"
    )

    assert split_table_markdown(table, 40) == [
        "| Thuốc | Hậu quả |",
        "| Warfarin | Tăng INR khi dùng kéo dài |",
        "| Rượu | Tăng độc tính trên gan |",
    ]


def test_split_table_markdown_keeps_a_wide_row_whole() -> None:
    table = (
        "| Thuốc | Hậu quả |\n|---|---|\n"
        "| Warfarin | Tăng INR khi dùng kéo dài, cần theo dõi INR hằng tuần |\n"
        "| Rượu | Độc gan |"
    )

    assert split_table_markdown(table, 40) == [
        "| Thuốc | Hậu quả |\n|---|---|\n"
        "| Warfarin | Tăng INR khi dùng kéo dài, cần theo dõi INR hằng tuần |",
        "| Thuốc | Hậu quả |\n|---|---|\n| Rượu | Độc gan |",
    ]


def test_split_lines_keeps_lowercase_continuations_with_their_entry() -> None:
    text = (
        "N02BE01 Paracetamol\n"
        "M01AB05 Diclofenac, dùng đường uống\n"
        "hoặc đặt trực tràng\n"
        "M01AC01 Piroxicam"
    )

    assert split_lines_without_breaking_entries(text, 60) == [
        "N02BE01 Paracetamol",
        "M01AB05 Diclofenac, dùng đường uống\nhoặc đặt trực tràng",
        "M01AC01 Piroxicam",
    ]


def test_split_lines_drops_blank_lines_and_keeps_long_entries_whole() -> None:
    text = (
        "  - **Efferalgan**: Paracetamol   \n\n"
        "- **Panadol Extra**: Paracetamol, cafein, dùng giảm đau hạ sốt\n"
        "- **Voltaren**: Diclofenac"
    )

    assert split_lines_without_breaking_entries(text, 30) == [
        "- **Efferalgan**: Paracetamol",
        "- **Panadol Extra**: Paracetamol, cafein, dùng giảm đau hạ sốt",
        "- **Voltaren**: Diclofenac",
    ]


def test_split_lines_never_splits_a_run_of_continuation_lines() -> None:
    assert split_lines_without_breaking_entries(
        "dạng uống\nhoặc tiêm\nhoặc đặt trực tràng", 12
    ) == ["dạng uống\nhoặc tiêm\nhoặc đặt trực tràng"]
    assert split_lines_without_breaking_entries(" \n\n", 12) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_splitters.py`
Expected: collection error `ModuleNotFoundError: No module named 'pharma_agent.domain.corpus.chunking'`.

- [ ] **Step 3: Write the implementation**

`backend/src/pharma_agent/domain/corpus/chunking.py`:

```python
"""The single corpus chunker (spec C §7.1).

Public API. Splitting is ported from corpus-pipeline ``build_final_chunk_records``,
``split_table_markdown``, ``split_lines_without_breaking_entries`` and
``split_long_text``; the splitter is chosen only by ``BlockRecord.kind``.
"""

import re

CHUNKER_VERSION = "chunker-v1"
MAX_CHUNK_CHARS = 3000

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?;:…]\s+")
_LAST_WORD_RE = re.compile(r"\s+\S*$")


def split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Cut at the last sentence end, else newline, else whitespace, else hard."""
    parts: list[str] = []
    remaining = paragraph.strip()
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        split_at = -1
        sentence_ends = list(_SENTENCE_END_RE.finditer(window))
        if sentence_ends:
            split_at = sentence_ends[-1].end()
        if split_at <= 0:
            newline_at = window.rfind("\n", 0, max_chars + 1)
            if newline_at > 0:
                split_at = newline_at + 1
        if split_at <= 0:
            last_word = _LAST_WORD_RE.search(window)
            if last_word and last_word.start() > 0:
                split_at = last_word.start()
        if split_at <= 0:
            split_at = max_chars
        chunk = remaining[:split_at].strip()
        if chunk:
            parts.append(chunk)
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Pack blank-line separated paragraphs; text that fits is returned as is."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_BREAK_RE.split(text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        chunks.extend(split_oversized_paragraph(paragraph, max_chars))
    if current:
        chunks.append(current)
    return chunks


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or "-" not in stripped:
        return False
    return stripped.replace("|", "").replace("-", "").replace(":", "").strip() == ""


def split_table_markdown(markdown: str, max_chars: int) -> list[str]:
    """Split rows into parts that each repeat the header and separator rows."""
    text = markdown.strip()
    if len(text) <= max_chars:
        return [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or not _is_separator_row(lines[1]):
        return split_long_text(text, max_chars)
    prefix = [lines[0], lines[1]]
    chunks: list[str] = []
    current_rows: list[str] = []
    for row in lines[2:]:
        candidate_rows = [*current_rows, row]
        if current_rows and len("\n".join(prefix + candidate_rows)) > max_chars:
            chunks.append("\n".join(prefix + current_rows))
            current_rows = [row]
            continue
        current_rows = candidate_rows
    if current_rows:
        chunks.append("\n".join(prefix + current_rows))
    return chunks or [text]


def split_lines_without_breaking_entries(text: str, max_chars: int) -> list[str]:
    """Pack lines; a line starting lowercase continues the entry above it."""
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join([*current, line]) if current else line
        if current and len(candidate) > max_chars:
            next_lines = [line]
            while current and next_lines[0][:1].islower():
                next_lines.insert(0, current.pop())
            if current:
                chunks.append("\n".join(current))
            current = next_lines
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus/test_splitters.py`
Expected: `15 passed`.

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/pharma_agent/domain/corpus/chunking.py tests/domain/corpus/test_splitters.py
git commit -m "feat(corpus): port prose, table and entry-list splitters"
```

The commit message ends with the session attribution trailer.

---

### Task 8: `chunk_section`, golden parity fixture and public API test

**Files:**
- Modify: `backend/src/pharma_agent/domain/corpus/chunking.py` (replace the header block, lines 1–15 as written in Task 7; append `_split_block`, `_context_header`, `chunk_section`)
- Create: `backend/tests/domain/corpus/golden/chunking_v1.json`
- Test: `backend/tests/domain/corpus/test_chunking.py`

**Interfaces:**
- Consumes: Task 2 models, Task 4 `chunk_version_id`/`sha256_hex`, Task 5 `hydrate_strategy_for` (test only), Task 6 enrichment functions, Task 7 splitters.
- Produces (overview §3.2, exact): `ChunkDraft` (frozen Pydantic model with `chunk_version_id: uuid.UUID`, `section_key: str`, `ordinal: int`, `kind: BlockKind`, `chunk_text: str`, `context_header: str`, `embedding_text: str`, `embedding_text_sha256: str`, `start_page: int | None`, `end_page: int | None`, `table_key: str | None`, `term_annotations: list[TermAnnotation]`, `colloquial: ColloquialMapping | None`); `chunk_section(document: DocumentRecord, section: SectionRecord, glossary: Sequence[GlossaryEntry], mappings: Sequence[ColloquialMappingRecord], *, max_chars: int = MAX_CHUNK_CHARS) -> list[ChunkDraft]`.
- Port of `build_final_chunk_records` (corpus-pipeline `build_canonical_rag.py`) plus `build_runtime_chunk_metadata` (`build_rag_metadata.py`):
  - Splitter by `block.kind` only: `TABLE` → `split_table_markdown` (old role `table`); `INDEX_ENTRIES` → `split_lines_without_breaking_entries` (old role `index_entry`, formerly `BRAND_INDEX_SECTION_ID`); `LIST` → the same splitter (old role `appendix_list`, formerly `APPENDIX_LIST_SECTION_IDS`); `PROSE` → `split_long_text`. `section.retrieval` does not affect splitting; it only drives the hydrate policy (Task 5).
  - Blocks whose stripped markdown is empty are skipped; `ordinal` starts at 1 and counts across all blocks (old `chunk_index`).
  - `chunk_text` is the stripped part (the old metadata builder stripped `text`).
  - Pages come from the block, else the section; unknown stays `None` (old `0`).
  - `table_key` is kept only for `TABLE` blocks (old `table_id` was only set for table blocks).
  - `context_header = build_context_header(document.title, section.context_path)`; if that is empty, `"<title> > <heading>"` when both exist, else whichever exists (old `_context_header` fallback).
  - `term_annotations = detect_terms(context_header + "\n\n" + chunk_text, glossary)` (just `chunk_text` when the header is empty); `colloquial = mapping_for_section(section.key, mappings)`; `embedding_text = compose_embedding_text(...)`; `embedding_text_sha256 = sha256_hex(embedding_text)`; `chunk_version_id = chunk_version_id(section.key, chunk_text, embedding_text, CHUNKER_VERSION)`.
  - Old fallback: when no block produced a chunk, `build_final_chunk_records` re-chunked `section.text`. That text is the `"\n\n"` join of the non-blank blocks, so it was always empty and produced a single empty chunk that `build_runtime_chunk_metadata` then rejected ("missing required chunk text"). The reachable behaviour is therefore "no chunks": `chunk_section` returns `[]` for a section whose blocks are all blank, and `read_bundle` rejects such sections anyway (Task 3).
- Golden file provenance: the `expected` values were produced while writing this plan by running the old corpus-pipeline code on the same inputs: `build_final_chunk_records` + `build_runtime_chunk_metadata` with `load_term_glossary`. For the table, list and index cases the section keys equal the old hard-coded ids (`ATC_SECTION_ID`, `BRAND_INDEX_SECTION_ID`), so the old roles applied. The leaflet case used `integrate_ankhang.integrate_drug_file`'s path (`chunk_text`, `build_context_header(title)`, `compact_colloquial_mapping` with `product_names_from_title`). Old `0` pages were mapped to `null`, and `colloquial` is the old compact dict. Spec C §7.3 says this golden test replaces the full parity check once P4 has run it.
- Contract for P4 (`seed bundle export`), needed for 100% parity: `document.title` equals the old section `title`; `context_path` is copied verbatim (the old `context_header` was always `build_context_header(title, context_path)`); leaflet sections have `context_path = ["Thông tin chi tiết"]`; each leaflet gets one `ColloquialMappingRecord` carrying `product_names_from_title(title)`, keyed by the curated slug or, for uncurated leaflets, the leaflet slug. The key only reaches `ColloquialMapping.key` (`test_colloquial_key_changes_nothing_but_the_colloquial_key`), so embedding text, terms and ids match the old build; the old compact mapping of uncurated leaflets had no `key`, so P4's parity check compares `colloquial` without `key` for those sections.

- [ ] **Step 1: Create the golden fixture**

`backend/tests/domain/corpus/golden/chunking_v1.json`:

```json
{
  "glossary": [
    {
      "term": "ADR",
      "case_sensitive": true,
      "vietnamese_expansions": [
        "tác dụng không mong muốn",
        "phản ứng có hại của thuốc"
      ],
      "english_expansions": [
        "Adverse Drug Reactions"
      ],
      "aliases": [
        "adverse drug reaction"
      ],
      "category": "safety",
      "confidence": "high",
      "source": "curated"
    },
    {
      "term": "NSAID",
      "case_sensitive": true,
      "vietnamese_expansions": [
        "thuốc chống viêm không steroid",
        "thuốc kháng viêm không steroid"
      ],
      "english_expansions": [
        "Nonsteroidal Anti-inflammatory Drug"
      ],
      "aliases": [
        "NSAIDs"
      ],
      "category": "drug_class",
      "confidence": "high",
      "source": "curated"
    },
    {
      "term": "INR",
      "case_sensitive": true,
      "vietnamese_expansions": [
        "tỷ số chuẩn hóa quốc tế"
      ],
      "english_expansions": [
        "International Normalized Ratio"
      ],
      "aliases": [],
      "category": "laboratory",
      "confidence": "high",
      "source": "curated"
    },
    {
      "term": "G6PD",
      "case_sensitive": true,
      "vietnamese_expansions": [
        "glucose-6-phosphate dehydrogenase"
      ],
      "english_expansions": [
        "glucose-6-phosphate dehydrogenase"
      ],
      "aliases": [
        "thiếu men G6PD",
        "thiếu enzym G6PD"
      ],
      "category": "laboratory",
      "confidence": "high",
      "source": "curated"
    }
  ],
  "colloquial_mappings": [
    {
      "key": "panadol-extra-gsk-150-vien-11440",
      "aliases": [
        "Panadol đỏ",
        "Panadol extra đỏ",
        "Panadol vỉ đỏ",
        "Panadol hộp đỏ"
      ],
      "visual_sign": "Hộp màu đỏ, vỉ thuốc màu đỏ",
      "product_names": [
        "Panadol Extra GSK giảm đau, hạ sốt",
        "Panadol Extra GSK"
      ],
      "section_keys": [
        "brand:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440"
      ]
    }
  ],
  "cases": [
    {
      "name": "prose",
      "max_chars": 160,
      "document": {
        "key": "drug:paracetamol",
        "kind": "drug_monograph",
        "title": "PARACETAMOL",
        "source": {
          "title": "Dược thư Quốc gia Việt Nam 2022",
          "url": null
        },
        "attributes": {}
      },
      "section": {
        "key": "drug:paracetamol:lieu-luong-va-cach-dung",
        "document_key": "drug:paracetamol",
        "heading": "Liều lượng và cách dùng",
        "context_path": [
          "Liều lượng và cách dùng"
        ],
        "ordinal": 1,
        "start_page": 1203,
        "end_page": 1204,
        "retrieval": "default",
        "blocks": [
          {
            "kind": "prose",
            "markdown": "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần, tối đa 4 g/ngày.\n\nTrẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ, không quá 5 lần trong 24 giờ.\n\nNgười bệnh thiếu men G6PD hoặc đang dùng NSAID cần theo dõi ADR trên gan và thận. Giảm liều khi suy gan, suy thận nặng; tránh dùng kéo dài ở người nghiện rượu mạn tính vì nguy cơ hoại tử tế bào gan.",
            "start_page": null,
            "end_page": null,
            "table_key": null
          }
        ]
      },
      "expected": {
        "hydrate_strategy": "full_section",
        "chunks": [
          {
            "ordinal": 1,
            "kind": "prose",
            "chunk_text": "Người lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần, tối đa 4 g/ngày.\n\nTrẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ, không quá 5 lần trong 24 giờ.",
            "start_page": 1203,
            "end_page": 1204,
            "table_key": null,
            "context_header": "PARACETAMOL\n> Liều lượng và cách dùng",
            "embedding_text": "PARACETAMOL\n> Liều lượng và cách dùng\n\nNgười lớn: uống 500 mg - 1 g mỗi 4 - 6 giờ khi cần, tối đa 4 g/ngày.\n\nTrẻ em: 10 - 15 mg/kg mỗi 4 - 6 giờ, không quá 5 lần trong 24 giờ.",
            "term_annotations": [],
            "colloquial": null
          },
          {
            "ordinal": 2,
            "kind": "prose",
            "chunk_text": "Người bệnh thiếu men G6PD hoặc đang dùng NSAID cần theo dõi ADR trên gan và thận. Giảm liều khi suy gan, suy thận nặng;",
            "start_page": 1203,
            "end_page": 1204,
            "table_key": null,
            "context_header": "PARACETAMOL\n> Liều lượng và cách dùng",
            "embedding_text": "PARACETAMOL\n> Liều lượng và cách dùng\n\nNgười bệnh thiếu men G6PD hoặc đang dùng NSAID cần theo dõi ADR trên gan và thận. Giảm liều khi suy gan, suy thận nặng;\n\nThuật ngữ: G6PD = glucose-6-phosphate dehydrogenase | NSAID = thuốc chống viêm không steroid; thuốc kháng viêm không steroid; Nonsteroidal Anti-inflammatory Drug | ADR = tác dụng không mong muốn; phản ứng có hại của thuốc; Adverse Drug Reactions",
            "term_annotations": [
              {
                "term": "G6PD",
                "vi": [
                  "glucose-6-phosphate dehydrogenase"
                ],
                "en": [
                  "glucose-6-phosphate dehydrogenase"
                ]
              },
              {
                "term": "NSAID",
                "vi": [
                  "thuốc chống viêm không steroid",
                  "thuốc kháng viêm không steroid"
                ],
                "en": [
                  "Nonsteroidal Anti-inflammatory Drug"
                ]
              },
              {
                "term": "ADR",
                "vi": [
                  "tác dụng không mong muốn",
                  "phản ứng có hại của thuốc"
                ],
                "en": [
                  "Adverse Drug Reactions"
                ]
              }
            ],
            "colloquial": null
          },
          {
            "ordinal": 3,
            "kind": "prose",
            "chunk_text": "tránh dùng kéo dài ở người nghiện rượu mạn tính vì nguy cơ hoại tử tế bào gan.",
            "start_page": 1203,
            "end_page": 1204,
            "table_key": null,
            "context_header": "PARACETAMOL\n> Liều lượng và cách dùng",
            "embedding_text": "PARACETAMOL\n> Liều lượng và cách dùng\n\ntránh dùng kéo dài ở người nghiện rượu mạn tính vì nguy cơ hoại tử tế bào gan.",
            "term_annotations": [],
            "colloquial": null
          }
        ]
      }
    },
    {
      "name": "table",
      "max_chars": 160,
      "document": {
        "key": "drug:paracetamol",
        "kind": "drug_monograph",
        "title": "PARACETAMOL",
        "source": {
          "title": "Dược thư Quốc gia Việt Nam 2022",
          "url": null
        },
        "attributes": {}
      },
      "section": {
        "key": "drug:paracetamol:tuong-tac-thuoc",
        "document_key": "drug:paracetamol",
        "heading": "Tương tác thuốc",
        "context_path": [
          "Tương tác thuốc"
        ],
        "ordinal": 2,
        "start_page": 1204,
        "end_page": 1205,
        "retrieval": "default",
        "blocks": [
          {
            "kind": "prose",
            "markdown": "Dùng đồng thời với thuốc cảm ứng enzym gan làm tăng độc tính trên gan của paracetamol.",
            "start_page": 1204,
            "end_page": 1204,
            "table_key": null
          },
          {
            "kind": "table",
            "markdown": "| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Warfarin | Tăng INR khi dùng kéo dài | Theo dõi INR |\n| Rượu | Tăng độc tính trên gan | Tránh uống rượu |\n| Isoniazid | Tăng nguy cơ độc gan | Giám sát men gan |\n| Cholestyramin | Giảm hấp thu paracetamol | Uống cách 1 giờ |",
            "start_page": 1205,
            "end_page": 1205,
            "table_key": "tbl-0042"
          }
        ]
      },
      "expected": {
        "hydrate_strategy": "full_section",
        "chunks": [
          {
            "ordinal": 1,
            "kind": "prose",
            "chunk_text": "Dùng đồng thời với thuốc cảm ứng enzym gan làm tăng độc tính trên gan của paracetamol.",
            "start_page": 1204,
            "end_page": 1204,
            "table_key": null,
            "context_header": "PARACETAMOL\n> Tương tác thuốc",
            "embedding_text": "PARACETAMOL\n> Tương tác thuốc\n\nDùng đồng thời với thuốc cảm ứng enzym gan làm tăng độc tính trên gan của paracetamol.",
            "term_annotations": [],
            "colloquial": null
          },
          {
            "ordinal": 2,
            "kind": "table",
            "chunk_text": "| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Warfarin | Tăng INR khi dùng kéo dài | Theo dõi INR |",
            "start_page": 1205,
            "end_page": 1205,
            "table_key": "tbl-0042",
            "context_header": "PARACETAMOL\n> Tương tác thuốc",
            "embedding_text": "PARACETAMOL\n> Tương tác thuốc\n\n| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Warfarin | Tăng INR khi dùng kéo dài | Theo dõi INR |\n\nThuật ngữ: INR = tỷ số chuẩn hóa quốc tế; International Normalized Ratio",
            "term_annotations": [
              {
                "term": "INR",
                "vi": [
                  "tỷ số chuẩn hóa quốc tế"
                ],
                "en": [
                  "International Normalized Ratio"
                ]
              }
            ],
            "colloquial": null
          },
          {
            "ordinal": 3,
            "kind": "table",
            "chunk_text": "| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Rượu | Tăng độc tính trên gan | Tránh uống rượu |",
            "start_page": 1205,
            "end_page": 1205,
            "table_key": "tbl-0042",
            "context_header": "PARACETAMOL\n> Tương tác thuốc",
            "embedding_text": "PARACETAMOL\n> Tương tác thuốc\n\n| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Rượu | Tăng độc tính trên gan | Tránh uống rượu |",
            "term_annotations": [],
            "colloquial": null
          },
          {
            "ordinal": 4,
            "kind": "table",
            "chunk_text": "| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Isoniazid | Tăng nguy cơ độc gan | Giám sát men gan |",
            "start_page": 1205,
            "end_page": 1205,
            "table_key": "tbl-0042",
            "context_header": "PARACETAMOL\n> Tương tác thuốc",
            "embedding_text": "PARACETAMOL\n> Tương tác thuốc\n\n| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Isoniazid | Tăng nguy cơ độc gan | Giám sát men gan |",
            "term_annotations": [],
            "colloquial": null
          },
          {
            "ordinal": 5,
            "kind": "table",
            "chunk_text": "| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Cholestyramin | Giảm hấp thu paracetamol | Uống cách 1 giờ |",
            "start_page": 1205,
            "end_page": 1205,
            "table_key": "tbl-0042",
            "context_header": "PARACETAMOL\n> Tương tác thuốc",
            "embedding_text": "PARACETAMOL\n> Tương tác thuốc\n\n| Thuốc phối hợp | Hậu quả | Xử trí |\n| --- | --- | --- |\n| Cholestyramin | Giảm hấp thu paracetamol | Uống cách 1 giờ |",
            "term_annotations": [],
            "colloquial": null
          }
        ]
      }
    },
    {
      "name": "list",
      "max_chars": 110,
      "document": {
        "key": "general:phu-luc-3-danh-muc-thuoc-phan-loai-theo-ma-atc",
        "kind": "general_monograph",
        "title": "PHỤ LỤC 3. DANH MỤC THUỐC PHÂN LOẠI THEO MÃ ATC",
        "source": {
          "title": "Dược thư Quốc gia Việt Nam 2022",
          "url": null
        },
        "attributes": {}
      },
      "section": {
        "key": "general:phu-luc-3-danh-muc-thuoc-phan-loai-theo-ma-atc:bang-phan-loai-atc",
        "document_key": "general:phu-luc-3-danh-muc-thuoc-phan-loai-theo-ma-atc",
        "heading": "Bảng phân loại ATC",
        "context_path": [
          "Bảng phân loại ATC"
        ],
        "ordinal": 1,
        "start_page": 1601,
        "end_page": 1602,
        "retrieval": "default",
        "blocks": [
          {
            "kind": "list",
            "markdown": "N02BE01 Paracetamol\nN02BA01 Acid acetylsalicylic\nM01AE01 Ibuprofen\nM01AB05 Diclofenac, dùng đường uống\nhoặc đặt trực tràng\nM01AC01 Piroxicam\nM01AH01 Celecoxib\nN02AX02 Tramadol",
            "start_page": null,
            "end_page": null,
            "table_key": null
          }
        ]
      },
      "expected": {
        "hydrate_strategy": "full_section",
        "chunks": [
          {
            "ordinal": 1,
            "kind": "list",
            "chunk_text": "N02BE01 Paracetamol\nN02BA01 Acid acetylsalicylic\nM01AE01 Ibuprofen",
            "start_page": 1601,
            "end_page": 1602,
            "table_key": null,
            "context_header": "PHỤ LỤC 3. DANH MỤC THUỐC PHÂN LOẠI THEO MÃ ATC\n> Bảng phân loại ATC",
            "embedding_text": "PHỤ LỤC 3. DANH MỤC THUỐC PHÂN LOẠI THEO MÃ ATC\n> Bảng phân loại ATC\n\nN02BE01 Paracetamol\nN02BA01 Acid acetylsalicylic\nM01AE01 Ibuprofen",
            "term_annotations": [],
            "colloquial": null
          },
          {
            "ordinal": 2,
            "kind": "list",
            "chunk_text": "M01AB05 Diclofenac, dùng đường uống\nhoặc đặt trực tràng\nM01AC01 Piroxicam\nM01AH01 Celecoxib\nN02AX02 Tramadol",
            "start_page": 1601,
            "end_page": 1602,
            "table_key": null,
            "context_header": "PHỤ LỤC 3. DANH MỤC THUỐC PHÂN LOẠI THEO MÃ ATC\n> Bảng phân loại ATC",
            "embedding_text": "PHỤ LỤC 3. DANH MỤC THUỐC PHÂN LOẠI THEO MÃ ATC\n> Bảng phân loại ATC\n\nM01AB05 Diclofenac, dùng đường uống\nhoặc đặt trực tràng\nM01AC01 Piroxicam\nM01AH01 Celecoxib\nN02AX02 Tramadol",
            "term_annotations": [],
            "colloquial": null
          }
        ]
      }
    },
    {
      "name": "index_entries",
      "max_chars": 110,
      "document": {
        "key": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat",
        "kind": "general_monograph",
        "title": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT",
        "source": {
          "title": "Dược thư Quốc gia Việt Nam 2022",
          "url": null
        },
        "attributes": {}
      },
      "section": {
        "key": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc",
        "document_key": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat",
        "heading": "Bảng tra cứu biệt dược",
        "context_path": [
          "Bảng tra cứu biệt dược"
        ],
        "ordinal": 1,
        "start_page": 1650,
        "end_page": 1652,
        "retrieval": "index_only",
        "blocks": [
          {
            "kind": "index_entries",
            "markdown": "- **Efferalgan**: Paracetamol\n- **Hapacol**: Paracetamol\n- **Panadol Extra**: Paracetamol, cafein\n- **Voltaren**: Diclofenac\n- **Celebrex**: Celecoxib",
            "start_page": null,
            "end_page": null,
            "table_key": null
          }
        ]
      },
      "expected": {
        "hydrate_strategy": "search_only",
        "chunks": [
          {
            "ordinal": 1,
            "kind": "index_entries",
            "chunk_text": "- **Efferalgan**: Paracetamol\n- **Hapacol**: Paracetamol\n- **Panadol Extra**: Paracetamol, cafein",
            "start_page": 1650,
            "end_page": 1652,
            "table_key": null,
            "context_header": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược",
            "embedding_text": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược\n\n- **Efferalgan**: Paracetamol\n- **Hapacol**: Paracetamol\n- **Panadol Extra**: Paracetamol, cafein",
            "term_annotations": [],
            "colloquial": null
          },
          {
            "ordinal": 2,
            "kind": "index_entries",
            "chunk_text": "- **Voltaren**: Diclofenac\n- **Celebrex**: Celecoxib",
            "start_page": 1650,
            "end_page": 1652,
            "table_key": null,
            "context_header": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược",
            "embedding_text": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược\n\n- **Voltaren**: Diclofenac\n- **Celebrex**: Celecoxib",
            "term_annotations": [],
            "colloquial": null
          }
        ]
      }
    },
    {
      "name": "leaflet",
      "max_chars": 160,
      "document": {
        "key": "leaflet:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440",
        "kind": "leaflet",
        "title": "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)",
        "source": {
          "title": "Tờ hướng dẫn sử dụng",
          "url": "https://www.nhathuocankhang.com/thuoc-giam-dau-ha-sot/panadol-extra-gsk-150-vien-11440"
        },
        "attributes": {}
      },
      "section": {
        "key": "brand:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440",
        "document_key": "leaflet:ankhang:thuoc-giam-dau-ha-sot:panadol-extra-gsk-150-vien-11440",
        "heading": "Thông tin chi tiết",
        "context_path": [
          "Thông tin chi tiết"
        ],
        "ordinal": 1,
        "start_page": null,
        "end_page": null,
        "retrieval": "default",
        "blocks": [
          {
            "kind": "prose",
            "markdown": "# Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n\nPanadol đỏ chứa paracetamol 500 mg và cafein 65 mg, giúp giảm đau đầu, đau răng, đau cơ và hạ sốt.\n\nNgười lớn uống 1 - 2 viên mỗi 4 - 6 giờ, tối đa 8 viên mỗi ngày. Không dùng chung với thuốc khác chứa paracetamol.",
            "start_page": null,
            "end_page": null,
            "table_key": null
          }
        ]
      },
      "expected": {
        "hydrate_strategy": "full_section",
        "chunks": [
          {
            "ordinal": 1,
            "kind": "prose",
            "chunk_text": "# Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n\nPanadol đỏ chứa paracetamol 500 mg và cafein 65 mg, giúp giảm đau đầu, đau răng, đau cơ và hạ sốt.",
            "start_page": null,
            "end_page": null,
            "table_key": null,
            "context_header": "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n> Thông tin chi tiết",
            "embedding_text": "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n> Thông tin chi tiết\n\nTên gọi khác: Panadol extra đỏ, Panadol vỉ đỏ, Panadol hộp đỏ\nDấu hiệu nhận biết: Hộp màu đỏ, vỉ thuốc màu đỏ\n\n# Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n\nPanadol đỏ chứa paracetamol 500 mg và cafein 65 mg, giúp giảm đau đầu, đau răng, đau cơ và hạ sốt.",
            "term_annotations": [],
            "colloquial": {
              "key": "panadol-extra-gsk-150-vien-11440",
              "aliases": [
                "Panadol đỏ",
                "Panadol extra đỏ",
                "Panadol vỉ đỏ",
                "Panadol hộp đỏ"
              ],
              "visual_sign": "Hộp màu đỏ, vỉ thuốc màu đỏ",
              "product_names": [
                "Panadol Extra GSK giảm đau, hạ sốt",
                "Panadol Extra GSK"
              ]
            }
          },
          {
            "ordinal": 2,
            "kind": "prose",
            "chunk_text": "Người lớn uống 1 - 2 viên mỗi 4 - 6 giờ, tối đa 8 viên mỗi ngày. Không dùng chung với thuốc khác chứa paracetamol.",
            "start_page": null,
            "end_page": null,
            "table_key": null,
            "context_header": "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n> Thông tin chi tiết",
            "embedding_text": "Panadol Extra GSK giảm đau, hạ sốt (15 vỉ x 12 viên)\n> Thông tin chi tiết\n\nTên gọi khác: Panadol đỏ, Panadol extra đỏ, Panadol vỉ đỏ, Panadol hộp đỏ\nDấu hiệu nhận biết: Hộp màu đỏ, vỉ thuốc màu đỏ\n\nNgười lớn uống 1 - 2 viên mỗi 4 - 6 giờ, tối đa 8 viên mỗi ngày. Không dùng chung với thuốc khác chứa paracetamol.",
            "term_annotations": [],
            "colloquial": {
              "key": "panadol-extra-gsk-150-vien-11440",
              "aliases": [
                "Panadol đỏ",
                "Panadol extra đỏ",
                "Panadol vỉ đỏ",
                "Panadol hộp đỏ"
              ],
              "visual_sign": "Hộp màu đỏ, vỉ thuốc màu đỏ",
              "product_names": [
                "Panadol Extra GSK giảm đau, hạ sốt",
                "Panadol Extra GSK"
              ]
            }
          }
        ]
      }
    }
  ]
}
```

Keep the file byte-for-byte as above (UTF-8, two-space indent, trailing newline). Never regenerate it from the new chunker: a change in expected output means a behaviour change and needs a new `CHUNKER_VERSION`.

- [ ] **Step 2: Write the failing test**

`backend/tests/domain/corpus/test_chunking.py`:

```python
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import TypeAdapter

from pharma_agent.domain.corpus import bundle, chunking, enrichment, hydrate, identity
from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    ColloquialMappingRecord,
    DocumentRecord,
    GlossaryEntry,
    SectionRecord,
)
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION, chunk_section
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for
from pharma_agent.domain.corpus.identity import chunk_version_id, sha256_hex

GOLDEN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "golden" / "chunking_v1.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = GOLDEN["cases"]
GLOSSARY = TypeAdapter(list[GlossaryEntry]).validate_json(
    json.dumps(GOLDEN["glossary"])
)
MAPPINGS = TypeAdapter(list[ColloquialMappingRecord]).validate_json(
    json.dumps(GOLDEN["colloquial_mappings"])
)
PUBLIC_API: dict[ModuleType, list[str]] = {
    bundle: [
        "BUNDLE_SCHEMA_VERSION",
        "KnowledgeBundle",
        "BundleValidationError",
        "read_bundle",
        "write_bundle",
        "model_slug",
        "encode_vector",
        "decode_vector",
    ],
    identity: [
        "CORPUS_NAMESPACE",
        "canonical_json",
        "sha256_hex",
        "section_revision_id",
        "chunk_version_id",
    ],
    chunking: ["CHUNKER_VERSION", "MAX_CHUNK_CHARS", "ChunkDraft", "chunk_section"],
    enrichment: [
        "build_context_header",
        "detect_terms",
        "mapping_for_section",
        "compose_embedding_text",
    ],
    hydrate: ["FULL_SECTION_MAX_CHARS", "section_char_count", "hydrate_strategy_for"],
}


def load_case(name: str) -> tuple[DocumentRecord, SectionRecord, int]:
    case = next(case for case in CASES if case["name"] == name)
    return (
        DocumentRecord.model_validate_json(json.dumps(case["document"])),
        SectionRecord.model_validate_json(json.dumps(case["section"])),
        case["max_chars"],
    )


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_chunk_section_matches_the_old_pipeline(case: dict[str, Any]) -> None:
    document, section, max_chars = load_case(case["name"])

    drafts = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)

    assert [
        {
            "ordinal": draft.ordinal,
            "kind": draft.kind.value,
            "chunk_text": draft.chunk_text,
            "start_page": draft.start_page,
            "end_page": draft.end_page,
            "table_key": draft.table_key,
            "context_header": draft.context_header,
            "embedding_text": draft.embedding_text,
            "term_annotations": [a.model_dump() for a in draft.term_annotations],
            "colloquial": draft.colloquial.model_dump(exclude_defaults=True)
            if draft.colloquial is not None
            else None,
        }
        for draft in drafts
    ] == case["expected"]["chunks"]
    assert hydrate_strategy_for(section).value == case["expected"]["hydrate_strategy"]


def test_chunk_drafts_carry_their_identities() -> None:
    for case in CASES:
        document, section, max_chars = load_case(case["name"])
        for draft in chunk_section(
            document, section, GLOSSARY, MAPPINGS, max_chars=max_chars
        ):
            assert draft.section_key == section.key
            assert draft.embedding_text_sha256 == sha256_hex(draft.embedding_text)
            assert draft.chunk_version_id == chunk_version_id(
                section.key, draft.chunk_text, draft.embedding_text, CHUNKER_VERSION
            )


def test_chunk_section_is_deterministic() -> None:
    document, section, _ = load_case("table")

    first = chunk_section(document, section, GLOSSARY, MAPPINGS)
    second = chunk_section(
        document.model_copy(deep=True),
        section.model_copy(deep=True),
        list(GLOSSARY),
        list(MAPPINGS),
    )

    assert first == second
    assert len({draft.chunk_version_id for draft in first}) == len(first) == 2


def test_glossary_change_creates_new_versions_only_for_affected_chunks() -> None:
    document, section, max_chars = load_case("prose")
    without_adr = [entry for entry in GLOSSARY if entry.term != "ADR"]

    before = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    after = chunk_section(document, section, without_adr, MAPPINGS, max_chars=max_chars)

    assert [draft.chunk_text for draft in after] == [d.chunk_text for d in before]
    assert [
        old.ordinal
        for old, new in zip(before, after, strict=True)
        if old.chunk_version_id != new.chunk_version_id
    ] == [2]
    assert "ADR = " not in after[1].embedding_text


def test_colloquial_mapping_change_creates_new_versions() -> None:
    document, section, max_chars = load_case("leaflet")
    edited = [MAPPINGS[0].model_copy(update={"visual_sign": "Vỉ thuốc màu đỏ"})]

    before = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    after = chunk_section(document, section, GLOSSARY, edited, max_chars=max_chars)

    assert all(
        old.chunk_version_id != new.chunk_version_id
        for old, new in zip(before, after, strict=True)
    )


def test_colloquial_key_changes_nothing_but_the_colloquial_key() -> None:
    document, section, max_chars = load_case("leaflet")
    unkeyed = [MAPPINGS[0].model_copy(update={"key": ""})]

    keyed = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    without_key = chunk_section(
        document, section, GLOSSARY, unkeyed, max_chars=max_chars
    )

    assert [d.colloquial.key if d.colloquial else None for d in keyed] == [
        "panadol-extra-gsk-150-vien-11440",
        "panadol-extra-gsk-150-vien-11440",
    ]
    assert [d.colloquial.key if d.colloquial else None for d in without_key] == [
        "",
        "",
    ]
    assert [d.model_dump(exclude={"colloquial": {"key"}}) for d in without_key] == [
        d.model_dump(exclude={"colloquial": {"key"}}) for d in keyed
    ]


def test_chunker_version_change_creates_new_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, section, max_chars = load_case("list")

    before = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)
    monkeypatch.setattr(chunking, "CHUNKER_VERSION", "chunker-v2")
    after = chunk_section(document, section, GLOSSARY, MAPPINGS, max_chars=max_chars)

    assert [draft.chunk_text for draft in after] == [d.chunk_text for d in before]
    assert not {d.chunk_version_id for d in before} & {
        d.chunk_version_id for d in after
    }


def test_a_section_whose_blocks_are_all_blank_has_no_chunks() -> None:
    document, section, _ = load_case("prose")
    blank = section.model_copy(
        update={
            "blocks": [
                BlockRecord(kind=BlockKind.PROSE, markdown=" \n "),
                BlockRecord(kind=BlockKind.TABLE, markdown=""),
            ]
        }
    )

    assert chunk_section(document, blank, GLOSSARY, MAPPINGS) == []


def test_context_header_falls_back_like_the_old_metadata_builder() -> None:
    document, section, _ = load_case("prose")
    no_path = section.model_copy(update={"context_path": []})
    label_title = document.model_copy(update={"title": "Tên gọi khác: Panadol đỏ"})
    untitled = document.model_copy(update={"title": ""})

    assert chunk_section(document, no_path, [], [])[0].context_header == "PARACETAMOL"
    assert (
        chunk_section(label_title, no_path, [], [])[0].context_header
        == "Tên gọi khác: Panadol đỏ > Liều lượng và cách dùng"
    )
    assert (
        chunk_section(untitled, no_path, [], [])[0].context_header
        == "Liều lượng và cách dùng"
    )


def test_table_key_is_kept_only_for_table_blocks() -> None:
    document, section, _ = load_case("table")
    prose, table = section.blocks
    tagged = section.model_copy(
        update={"blocks": [prose.model_copy(update={"table_key": "tbl-9999"}), table]}
    )

    drafts = chunk_section(document, tagged, GLOSSARY, MAPPINGS)

    assert [draft.table_key for draft in drafts] == [None, "tbl-0042"]


@pytest.mark.parametrize(
    "module", list(PUBLIC_API), ids=[module.__name__ for module in PUBLIC_API]
)
def test_public_api_for_seed_pipeline_is_documented(module: ModuleType) -> None:
    assert "Public API" in (module.__doc__ or "")
    for name in PUBLIC_API[module]:
        assert hasattr(module, name), f"{module.__name__}.{name}"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest -q tests/domain/corpus/test_chunking.py`
Expected: collection error `ImportError: cannot import name 'chunk_section' from 'pharma_agent.domain.corpus.chunking'`.

- [ ] **Step 4: Replace the header block of `chunking.py`**

In `backend/src/pharma_agent/domain/corpus/chunking.py`, replace everything from the module docstring down to and including `_LAST_WORD_RE = re.compile(r"\s+\S*$")` with:

```python
"""The single corpus chunker (spec C §7.1).

Public API. Splitting is ported from corpus-pipeline ``build_final_chunk_records``,
``split_table_markdown``, ``split_lines_without_breaking_entries`` and
``split_long_text``; the splitter is chosen only by ``BlockRecord.kind``.
"""

import re
import uuid
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    ColloquialMappingRecord,
    DocumentRecord,
    GlossaryEntry,
    SectionRecord,
)
from pharma_agent.domain.corpus.enrichment import (
    build_context_header,
    compose_embedding_text,
    detect_terms,
    mapping_for_section,
)
from pharma_agent.domain.corpus.identity import chunk_version_id, sha256_hex
from pharma_agent.domain.retrieval.models import ColloquialMapping, TermAnnotation

CHUNKER_VERSION = "chunker-v1"
MAX_CHUNK_CHARS = 3000

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?;:…]\s+")
_LAST_WORD_RE = re.compile(r"\s+\S*$")


class ChunkDraft(BaseModel):
    """One chunk of a section, ready to be stored as an immutable chunk version."""

    model_config = ConfigDict(frozen=True)

    chunk_version_id: uuid.UUID
    section_key: str
    ordinal: int
    kind: BlockKind
    chunk_text: str
    context_header: str
    embedding_text: str
    embedding_text_sha256: str
    start_page: int | None
    end_page: int | None
    table_key: str | None
    term_annotations: list[TermAnnotation]
    colloquial: ColloquialMapping | None
```

- [ ] **Step 5: Append the chunker**

Append to the end of `backend/src/pharma_agent/domain/corpus/chunking.py` (after `split_lines_without_breaking_entries`):

```python
def _split_block(kind: BlockKind, text: str, max_chars: int) -> list[str]:
    if kind is BlockKind.TABLE:
        return split_table_markdown(text, max_chars)
    if kind is BlockKind.LIST or kind is BlockKind.INDEX_ENTRIES:
        return split_lines_without_breaking_entries(text, max_chars)
    return split_long_text(text, max_chars)


def _context_header(title: str, section: SectionRecord) -> str:
    header = build_context_header(title, section.context_path)
    if header:
        return header
    title_text = title.strip()
    heading = section.heading.strip()
    if title_text and heading:
        return f"{title_text} > {heading}"
    return title_text or heading


def chunk_section(
    document: DocumentRecord,
    section: SectionRecord,
    glossary: Sequence[GlossaryEntry],
    mappings: Sequence[ColloquialMappingRecord],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[ChunkDraft]:
    """Chunk one section; a section whose blocks are all blank yields no chunks."""
    context_header = _context_header(document.title, section)
    colloquial = mapping_for_section(section.key, mappings)
    drafts: list[ChunkDraft] = []
    for block in section.blocks:
        raw_text = block.markdown.strip()
        if not raw_text:
            continue
        for part in _split_block(block.kind, raw_text, max_chars):
            chunk_text = part.strip()
            searchable = (
                f"{context_header}\n\n{chunk_text}" if context_header else chunk_text
            )
            terms = detect_terms(searchable, glossary)
            embedding_text = compose_embedding_text(
                context_header=context_header,
                chunk_text=chunk_text,
                colloquial=colloquial,
                terms=terms,
            )
            drafts.append(
                ChunkDraft(
                    chunk_version_id=chunk_version_id(
                        section.key, chunk_text, embedding_text, CHUNKER_VERSION
                    ),
                    section_key=section.key,
                    ordinal=len(drafts) + 1,
                    kind=block.kind,
                    chunk_text=chunk_text,
                    context_header=context_header,
                    embedding_text=embedding_text,
                    embedding_text_sha256=sha256_hex(embedding_text),
                    start_page=block.start_page
                    if block.start_page is not None
                    else section.start_page,
                    end_page=block.end_page
                    if block.end_page is not None
                    else section.end_page,
                    table_key=block.table_key
                    if block.kind is BlockKind.TABLE
                    else None,
                    term_annotations=terms,
                    colloquial=colloquial,
                )
            )
    return drafts
```

`chunk_section` reads the module-level `CHUNKER_VERSION` at call time (the version test monkeypatches it), so do not bind it as a default argument.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest -q tests/domain/corpus`
Expected: `84 passed` (11 models + 16 IO + 7 identity + 4 hydrate + 12 enrichment + 15 splitters + 19 chunking).

- [ ] **Step 7: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green (while writing this plan the whole unit suite ran with these files in place: `266 passed, 20 deselected`, ruff and format clean, pyrefly `0 diagnostics`).

- [ ] **Step 8: Commit**

```bash
git add src/pharma_agent/domain/corpus/chunking.py tests/domain/corpus/golden/chunking_v1.json tests/domain/corpus/test_chunking.py
git commit -m "feat(corpus): add chunk_section with golden parity fixture"
```

The commit message ends with the session attribution trailer.

---

## Self-Review

**Spec coverage**

| Spec requirement | Task |
| --- | --- |
| §4 code layout `domain/corpus/{bundle,identity,chunking,enrichment,hydrate}.py` | 2–8 (`models.py`, `ports.py` belong to P2) |
| §4 public API (`bundle`, `identity`, `chunking`, `enrichment`, `hydrate`) named in module docstrings, with an import test | 2, 4, 5, 6, 7 (docstrings); 8 (`test_public_api_for_seed_pipeline_is_documented`, five modules) |
| §5.1 bundle files, manifest fields, JSONL records, base64 float32 vectors | 2 (models, `encode_vector`/`decode_vector`, `model_slug`), 3 (`write_bundle`/`read_bundle`) |
| §5.1 glossary keeps the `term_glossary.json` schema; colloquial mappings add `section_keys` | 2 (`GlossaryEntry`, `ColloquialMappingRecord`), 3 |
| §5.2 keys and `kind`/`retrieval` replacing hard-coded ids | 2 (enums), 5 (`INDEX_ONLY` → `search_only`), 8 (splitter by `kind`) |
| §5.3 schema version, per-file sha256 and size | 3 |
| §5.3 unique keys, `document_key` exists, unique ordinal per document | 3 |
| §5.3 non-empty markdown, valid pages | 3 |
| §5.3 vector dims match manifest, bytes = dims × 4 | 3 |
| §5.3 every problem listed with file, line and field; nothing written on failure | 3 (`BundleValidationError.problems`; `write_bundle` validates before writing) |
| Overview §3.2 `write_bundle` computes sha256/bytes/counts and embedding entries (placeholders allowed) | 3 (`test_write_bundle_replaces_placeholder_manifest_fields`, `test_write_bundle_without_embeddings_lists_no_embedding_files`) |
| Colloquial key (curated slug, leaflet slug or empty) accepted and without effect on chunk output | 3 (`test_colloquial_mapping_key_may_be_empty_or_a_slug`), 8 (`test_colloquial_key_changes_nothing_but_the_colloquial_key`) |
| §6.1 normalization, canonical JSON, namespace, `section_revision_id`, `chunk_version_id`, `embedding_text_sha256` | 1 (`normalize_text`), 4, 8 (`embedding_text_sha256`) |
| §6.1 glossary/colloquial change → new chunk version | 8 (`test_glossary_change_...`, `test_colloquial_mapping_change_...`) |
| §7.1 `CHUNKER_VERSION`, `MAX_CHUNK_CHARS`, `FULL_SECTION_MAX_CHARS` | 7, 5 |
| §7.1 table keeps header, entries/lists not broken, prose via `split_long_text` | 7 |
| §7.1 empty-section fallback | 8 (reachable behaviour: no chunks; see Task 8 notes) |
| §7.1 pages from block, else section | 8 |
| §7.1 `context_header`, `embedding_text` order, term annotations on header + chunk | 6, 8 |
| §7.1 changing `CHUNKER_VERSION` changes ids | 4, 8 |
| §7.2 hydrate policy | 5 |
| §7.3 parity with the old `chunks.jsonl` | 7 (outputs taken from the old functions + fuzz comparison), 8 (golden fixture from the old code for all four kinds and a leaflet); the full-corpus check stays in P4 |
| §12 unit domain tests (each kind, fallback, identity determinism, hydrate policy, each validation error) | 3–8 |
| Overview §3.1 `make_snippet` | 1 (P3 switches `summary_view`) |

**Names checked against overview §3.1–§3.2:**
- `normalize_text`, `make_snippet`
- `BUNDLE_SCHEMA_VERSION`, `DocumentKind`, `BlockKind`, `RetrievalMode` with their exact values
- `SourceInfo`, `DocumentRecord`, `BlockRecord`, `SectionRecord`, `GlossaryEntry`, `ColloquialMappingRecord`
- `BundleCollection`, `BundleGenerator`, `BundleFile`, `BundleEmbeddingFile`, `BundleManifest`, `KnowledgeBundle`
- `BundleValidationError` (`code = "BUNDLE_INVALID"`, `problems`)
- `model_slug`, `encode_vector`, `decode_vector`, `read_bundle`, `write_bundle`
- `CORPUS_NAMESPACE`, `canonical_json`, `sha256_hex`, `section_revision_id`, `chunk_version_id`
- `build_context_header`, `detect_terms`, `mapping_for_section`, `compose_embedding_text`
- `FULL_SECTION_MAX_CHARS`, `section_char_count`, `hydrate_strategy_for`
- `CHUNKER_VERSION`, `MAX_CHUNK_CHARS`, `ChunkDraft` (field names and types exact), `chunk_section` (signature exact)

`TermAnnotation`, `ColloquialMapping` and `HydrateStrategy` stay in `pharma_agent.domain.retrieval.models`, and embedding file lines are `{"embedding_text_sha256", "dims", "vector"}`.

**Decisions that refine the spec or overview (P2 and P4 must follow them):**
1. `colloquial_mappings.json` is a JSON array of `ColloquialMappingRecord` objects, not the resource file's slug-keyed object: one record per leaflet carries its title-derived `product_names`. P4 keys the 2 417 uncurated leaflets by their leaflet slug, while the old compact mapping had no `key` for them. The key never changes `embedding_text`, `term_annotations` or ids, so parity holds except `colloquial.key`, and P4 compares without it for those sections.
2. Glossary terms must be unique ignoring case. The old loader only rejected exact duplicates; the current glossary has none either way.
3. `GlossaryEntry.case_sensitive` defaults to `False` as pinned. The old loader defaulted to `True`, but every current entry sets the field.
4. Old leaflet chunks came from `integrate_ankhang.chunk_text`, a second algorithm. It re-joins paragraphs even when the text fits, splits oversized pipe-table paragraphs with `split_table` (which breaks at `>= max_chars`), and strips bold markers. The single chunker ports the formulary algorithms (spec §7.1). P4's export must shape leaflet blocks as follows, and its parity run is the check for the remaining edge cases:
   - content already cleaned
   - paragraphs joined with `"\n\n"`
   - any paragraph that is a pipe table longer than 3 000 characters becomes a `table` block
