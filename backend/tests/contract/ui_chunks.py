"""Strict Pydantic mirror of AI SDK 7 `uiMessageChunkSchema` (ai 7.0.99, ui-message-chunks.ts).

Field names are the wire names. Optional zod fields default to a value instead of None because
zod `.optional()` rejects an explicit null. The frontend contract test validates the same
fixtures with the real zod schema.
"""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    StringConstraints,
    Tag,
    TypeAdapter,
)

ProviderMetadata = dict[str, dict[str, JsonValue]]


class StrictChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StartChunk(StrictChunk):
    type: Literal["start"]
    messageId: str = ""
    messageMetadata: JsonValue = None


class TextStartChunk(StrictChunk):
    type: Literal["text-start"]
    id: str
    providerMetadata: ProviderMetadata = Field(default_factory=dict)


class TextDeltaChunk(StrictChunk):
    type: Literal["text-delta"]
    id: str
    delta: str
    providerMetadata: ProviderMetadata = Field(default_factory=dict)


class TextEndChunk(StrictChunk):
    type: Literal["text-end"]
    id: str
    providerMetadata: ProviderMetadata = Field(default_factory=dict)


class SourceDocumentChunk(StrictChunk):
    type: Literal["source-document"]
    sourceId: str
    mediaType: str
    title: str
    filename: str = ""
    providerMetadata: ProviderMetadata = Field(default_factory=dict)


class DataChunk(StrictChunk):
    type: Annotated[str, StringConstraints(pattern=r"^data-.+")]
    id: str = ""
    data: JsonValue
    transient: bool = False


class MessageMetadataChunk(StrictChunk):
    type: Literal["message-metadata"]
    messageMetadata: JsonValue


class FinishChunk(StrictChunk):
    type: Literal["finish"]
    finishReason: Literal[
        "stop", "length", "content-filter", "tool-calls", "error", "other"
    ] = "other"
    messageMetadata: JsonValue = None


class ErrorChunk(StrictChunk):
    type: Literal["error"]
    errorText: str


class AbortChunk(StrictChunk):
    type: Literal["abort"]
    reason: str = ""


def _chunk_tag(value: object) -> str:
    raw = value.get("type") if isinstance(value, dict) else getattr(value, "type", "")
    text = raw if isinstance(raw, str) else ""
    return "data" if text.startswith("data-") else text


UIChunk = Annotated[
    Annotated[StartChunk, Tag("start")]
    | Annotated[TextStartChunk, Tag("text-start")]
    | Annotated[TextDeltaChunk, Tag("text-delta")]
    | Annotated[TextEndChunk, Tag("text-end")]
    | Annotated[SourceDocumentChunk, Tag("source-document")]
    | Annotated[DataChunk, Tag("data")]
    | Annotated[MessageMetadataChunk, Tag("message-metadata")]
    | Annotated[FinishChunk, Tag("finish")]
    | Annotated[ErrorChunk, Tag("error")]
    | Annotated[AbortChunk, Tag("abort")],
    Discriminator(_chunk_tag),
]

_CHUNK: TypeAdapter[StrictChunk] = TypeAdapter(UIChunk)


def validate_chunk(chunk: object) -> StrictChunk:
    return _CHUNK.validate_python(chunk)


# Payloads of our own data parts and metadata (the frontend's dataPartSchemas and
# messageMetadataSchema come from the OpenAPI models of Task 2).
class PhaseData(StrictChunk):
    phase: Literal[
        "guarding",
        "understanding",
        "selecting_skills",
        "searching",
        "reading",
        "answering",
    ]
    round: int = 0


class SkillRef(StrictChunk):
    name: str
    title: str


class SkillsData(StrictChunk):
    skills: list[SkillRef]


class EvidenceEntry(StrictChunk):
    index: int
    source: str
    title: str
    section: str
    startPage: int | None
    endPage: int | None
    snippet: str


class EvidenceData(StrictChunk):
    items: list[EvidenceEntry]


class ConversationData(StrictChunk):
    id: str
    title: str


DATA_PAYLOADS: dict[str, type[StrictChunk]] = {
    "data-phase": PhaseData,
    "data-skills": SkillsData,
    "data-evidence": EvidenceData,
    "data-conversation": ConversationData,
}


class PharmaSource(EvidenceEntry):
    isCurrent: bool


class UsageMetadata(StrictChunk):
    llmCalls: int
    promptTokens: int
    completionTokens: int
    searchRounds: int


class FinishMetadata(StrictChunk):
    status: Literal[
        "completed", "partial", "abstained", "blocked", "redirected", "error", "timeout"
    ]
    errorCode: str | None
    usage: UsageMetadata
    runId: str
    persisted: bool
    createdAt: str
