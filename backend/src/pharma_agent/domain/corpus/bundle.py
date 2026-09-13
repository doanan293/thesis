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
