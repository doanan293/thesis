# Implementation plans overview (2026-09-13)

This file lists every implementation plan for the three specs approved on 2026-09-13 and pins
the names that more than one plan uses. A plan must use these names exactly. Anything not
pinned here is decided inside the plan that owns it.

Specs:

- C: `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md`
- A: `backend/docs/superpowers/specs/2026-09-13-web-client-contract-design.md`
- B: `frontend/docs/superpowers/specs/2026-09-13-frontend-v1-design.md`

## 1. Plans and order

| # | File | Spec | Depends on | Delivers |
| --- | --- | --- | --- | --- |
| P1 | `backend/docs/superpowers/plans/2026-09-13-corpus-domain.md` | C §4–§7 | none | `pharma_agent.domain.corpus` (bundle models and IO, identity, enrichment, chunking, hydrate policy) and `pharma_agent.domain.shared.text` |
| P2 | `backend/docs/superpowers/plans/2026-09-13-corpus-store-import.md` | C §6, §8 | P1 | Schema `corpus` (migration 0005), repositories, embedding cache, Qdrant index, `ImportKnowledgeBundle`, release commands, fixture bundle |
| P3 | `backend/docs/superpowers/plans/2026-09-13-corpus-retrieval.md` | C §9 | P1, P2 | Retrieval reads Postgres, filters by release, `PostgresHydrator`, audit with release ids (migration 0006), health check, `build_retrieval_service` |
| P4 | `backend/docs/superpowers/plans/2026-09-13-seed-pipeline.md` | C §10, §11 | P1, P2, P3 | Rename to `seed-pipeline`, `seed bundle export`, `seed bundle embed`, evaluation through the backend, removals, docs, chunk parity check, migration runbook |
| P5 | `backend/docs/superpowers/plans/2026-09-13-web-api-foundation.md` | A §4.1, §7, §8 | P3 | RFC 9457 errors, operation ids and `export-openapi`, keyset cursors (migration 0007), `POST /conversations` |
| P6 | `backend/docs/superpowers/plans/2026-09-13-web-chat-stream.md` | A §3, §4.2, §5 | P5 | UI Message Stream encoder, `UIMessage` history, `message_citations` (migration 0008), citation detail endpoint, contract fixtures |
| P7 | `backend/docs/superpowers/plans/2026-09-13-web-auth-e2e.md` | A §6, §9, §11 | P6 | Cookie sessions (migration 0009), CSRF, OAuth on cookie, `cleanup-sessions`, E2E server, README |
| P8 | `frontend/docs/superpowers/plans/2026-09-13-frontend-foundation.md` | B §2–§9, §12, §14 | Tasks marked "needs P5/P7" wait for those plans | Scaffold, tooling, i18n, theme, orval client, auth pages and guard, app shell, landing, Docker and nginx |
| P9 | `frontend/docs/superpowers/plans/2026-09-13-frontend-chat.md` | B §10 | P6, P8 | Chat module |
| P10 | `frontend/docs/superpowers/plans/2026-09-13-frontend-skills-settings-e2e.md` | B §11, §13 | P7, P9 | Skills, settings, Playwright E2E |

Execution order: P1 → P2 → P3 → P4 → P5 → P6 → P7. P8 can start any time; its tasks that call the
backend wait for P5 (errors, OpenAPI) and P7 (cookie auth). P9 after P6. P10 after P7 and P9.

## 2. Global constraints (every plan)

- The environment is development only. Postgres and Qdrant may be reset; no data backfill or
  backward compatibility is needed.
- Python projects: Python 3.12, uv, shared `ruff.toml`, strict `pyrefly.toml` with
  `unused-ignore = true`, pytest `filterwarnings = ["error"]`. Lint and type errors are fixed in
  code; never add rule ignores, `# noqa`, `# type: ignore` or new `# pyrefly: ignore`.
- Every backend task ends green on: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`. Tasks touching Postgres or Qdrant also run `uv run pytest -q -m integration`.
- Backend layering stays enforced by `tests/architecture/test_layering.py`: the domain imports no
  framework (`sqlalchemy`, `qdrant_client`, `fastapi`, `openai`, `httpx`, `langgraph`...) and no
  outer layer; `api/` never imports `pharma_agent.domain`.
- Prefer established libraries over custom code. Do not add a feature flag or "fake mode" to
  production code for tests; fakes live under `tests/`.
- Frontend: TypeScript 7 strict, Oxlint type-aware with `--deny-warnings`, Prettier, no
  `oxlint-disable`, `@ts-ignore` or `@ts-expect-error`. npm (bundled with Node 24).
- Commits: one commit per task, conventional message, ending with the session attribution
  trailer given in the executing session.
- Product copy is Vietnamese by default (i18n key based in the frontend); the agent never adds
  medical disclaimers.

## 3. Pinned backend names

### 3.1 Shared text helpers — `pharma_agent.domain.shared.text` (P1)

```python
def normalize_text(text: str) -> str: ...        # NFC, "\r\n"/"\r" -> "\n", rstrip each line, strip outer blank lines
def make_snippet(text: str, max_chars: int) -> str: ...  # collapse whitespace, drop markdown table pipes/rules, cut at word boundary, append "…" when cut
```

`EvidenceSet.summary_view` switches to `make_snippet(..., 300)` in P3.

### 3.2 Corpus domain — `pharma_agent.domain.corpus` (P1)

`bundle.py`

```python
BUNDLE_SCHEMA_VERSION = "knowledge-bundle/v1"

class DocumentKind(StrEnum): DRUG_MONOGRAPH = "drug_monograph"; GENERAL_MONOGRAPH = "general_monograph"; LEAFLET = "leaflet"
class BlockKind(StrEnum): PROSE = "prose"; TABLE = "table"; LIST = "list"; INDEX_ENTRIES = "index_entries"
class RetrievalMode(StrEnum): DEFAULT = "default"; INDEX_ONLY = "index_only"

class SourceInfo(BaseModel): title: str; url: str | None = None
class DocumentRecord(BaseModel): key: str; kind: DocumentKind; title: str; source: SourceInfo; attributes: dict[str, str | int | float | bool | None] = {}
class BlockRecord(BaseModel): kind: BlockKind; markdown: str; start_page: int | None = None; end_page: int | None = None; table_key: str | None = None
class SectionRecord(BaseModel): key: str; document_key: str; heading: str; context_path: list[str]; ordinal: int; start_page: int | None = None; end_page: int | None = None; retrieval: RetrievalMode = RetrievalMode.DEFAULT; blocks: list[BlockRecord]
class GlossaryEntry(BaseModel):   # mirrors data/resources/term_glossary.json
    term: str; case_sensitive: bool = False; vietnamese_expansions: list[str] = []; english_expansions: list[str] = []
    aliases: list[str] = []; category: str = ""; confidence: str = ""; source: str = ""
class ColloquialMappingRecord(BaseModel):   # data/resources/colloquial_mappings.json entry, keyed by An Khang slug
    key: str; aliases: list[str] = []; visual_sign: str = ""; product_names: list[str] = []; section_keys: list[str] = []
class BundleCollection(BaseModel): key: str; title: str
class BundleGenerator(BaseModel): name: str; version: str; build_id: str
class BundleFile(BaseModel): sha256: str; bytes: int
class BundleEmbeddingFile(BaseModel): model: str; dims: int; file: str
class BundleManifest(BaseModel):
    schema_version: Literal["knowledge-bundle/v1"]; collection: BundleCollection; generator: BundleGenerator
    source_digests: dict[str, str]; document_count: int; section_count: int
    files: dict[str, BundleFile]; embeddings: list[BundleEmbeddingFile] = []
class KnowledgeBundle(BaseModel):
    manifest: BundleManifest; documents: list[DocumentRecord]; sections: list[SectionRecord]
    glossary: list[GlossaryEntry]; colloquial_mappings: list[ColloquialMappingRecord]
    embeddings: dict[str, dict[str, list[float]]] = {}   # model -> embedding_text_sha256 -> vector

class BundleValidationError(DomainError): code = "BUNDLE_INVALID"; problems: list[str]

def model_slug(model: str) -> str: ...                 # lowercase, non [a-z0-9] runs -> "_", strip "_"  ("qwen3-embedding:4b-fp16" -> "qwen3_embedding_4b_fp16")
def encode_vector(values: Sequence[float]) -> str: ... # float32 little-endian, base64
def decode_vector(data: str, dims: int) -> list[float]: ...
def read_bundle(directory: Path) -> KnowledgeBundle: ...                     # validates §5.3, raises BundleValidationError
def write_bundle(bundle: KnowledgeBundle, directory: Path) -> BundleManifest: ...  # writes files, computes sha256/bytes/counts, returns the manifest written
```

Files inside a bundle: `manifest.json`, `documents.jsonl`, `sections.jsonl`, `glossary.json`,
`colloquial_mappings.json`, `embeddings/<model_slug>.jsonl` with lines
`{"embedding_text_sha256": str, "dims": int, "vector": str}`.

`identity.py`

```python
CORPUS_NAMESPACE: uuid.UUID            # uuid5(NAMESPACE_URL, "pharma-agent:corpus:v1")
def canonical_json(value: object) -> str: ...
def sha256_hex(text: str) -> str: ...
def section_revision_id(section_key: str, blocks: Sequence[BlockRecord]) -> uuid.UUID: ...
def chunk_version_id(section_key: str, chunk_text: str, embedding_text: str, chunker_version: str) -> uuid.UUID: ...
```

`enrichment.py`

```python
def build_context_header(title: str, context_path: Sequence[str]) -> str: ...
def detect_terms(text: str, glossary: Sequence[GlossaryEntry]) -> list[TermAnnotation]: ...
def mapping_for_section(section_key: str, mappings: Sequence[ColloquialMappingRecord]) -> ColloquialMapping | None: ...
def compose_embedding_text(*, context_header: str, chunk_text: str, colloquial: ColloquialMapping | None, terms: Sequence[TermAnnotation]) -> str: ...
```

`TermAnnotation` and `ColloquialMapping` stay in `pharma_agent.domain.retrieval.models`.

`hydrate.py`

```python
FULL_SECTION_MAX_CHARS = 16000
def section_char_count(section: SectionRecord) -> int: ...   # len("\n\n".join(block.markdown for block in blocks))
def hydrate_strategy_for(section: SectionRecord) -> HydrateStrategy: ...
```

`chunking.py`

```python
CHUNKER_VERSION = "chunker-v1"
MAX_CHUNK_CHARS = 3000

class ChunkDraft(BaseModel):   # frozen
    chunk_version_id: uuid.UUID; section_key: str; ordinal: int   # ordinal starts at 1 inside the section
    kind: BlockKind; chunk_text: str; context_header: str; embedding_text: str; embedding_text_sha256: str
    start_page: int | None; end_page: int | None; table_key: str | None
    term_annotations: list[TermAnnotation]; colloquial: ColloquialMapping | None

def chunk_section(document: DocumentRecord, section: SectionRecord, glossary: Sequence[GlossaryEntry],
                  mappings: Sequence[ColloquialMappingRecord], *, max_chars: int = MAX_CHUNK_CHARS) -> list[ChunkDraft]: ...
```

`models.py`, `ports.py`: owned by P2 (P2 may add types here). P3 adds `CorpusReader` to
`pharma_agent.domain.retrieval.ports`.

### 3.3 Corpus storage (P2)

- Alembic revision ids: `0005` (P2, schema `corpus`), `0006` (P3, audit), `0007` (P5, keyset
  indexes), `0008` (P6, `message_citations`), `0009` (P7, `access_tokens`).
- SQLAlchemy tables live in `pharma_agent/infrastructure/persistence/postgres/corpus_tables.py`,
  on the same `Base.metadata`, `schema="corpus"`, table and column names exactly as spec C §6.2:
  `collections`, `documents`, `sections`, `section_revisions`, `chunk_versions`, `releases`,
  `release_chunks`, `glossary_entries`, `colloquial_mappings`, `embedding_cache`. Class names:
  `CollectionTable`, `DocumentTable`, `SectionTable`, `SectionRevisionTable`,
  `ChunkVersionTable`, `ReleaseTable`, `ReleaseChunkTable`, `GlossaryEntryTable`,
  `ColloquialMappingTable`, `EmbeddingCacheTable`. `chunk_versions` also has `context_header text not null`
  (from `ChunkDraft.context_header`), so readers never rebuild it.
- Primary keys are `uuid` columns; corpus ids leave the database as `uuid.UUID` in Python and as
  the canonical hyphenated string in JSON, Qdrant payloads and API responses.
- Application service `pharma_agent.application.corpus.import_bundle.ImportKnowledgeBundle` with
  `async def __call__(self, bundle: KnowledgeBundle, *, publish: bool) -> ImportReport`.
- Release commands service `pharma_agent.application.corpus.releases.ReleaseService` with
  `list_releases(collection_key)`, `publish(release_id)`, `rollback(collection_key)`,
  `gc(collection_key, keep)`, `reindex(collection_key)`.
- CLI group `pharma-agent corpus` with commands `import`, `releases`, `publish`, `rollback`,
  `gc`, `reindex` (signatures in spec C §8.1).
- Settings: `CorpusSettings(embed_batch_size: int = 32, embed_max_concurrent: int = 1, gc_keep: int = 2)` as `Settings.corpus`.
- Qdrant: physical collection `chunks_<model_slug(embedding model)>`, alias `chunks_current`;
  vector names `dense_vector` and `bm25_sparse_vector` (constants already in
  `infrastructure/retrieval/qdrant_adapter.py`); payload keys `collection_id`, `release_ids`,
  `document_id`, `section_id`, `section_revision_id`, `kind`; point id = `str(chunk_version_id)`.
  Qdrant adapter for writes: `pharma_agent.infrastructure.retrieval.qdrant_index.QdrantVectorIndex`.
- Constructors (as written in P2): `PostgresCorpusRepository(sessions)`, `PostgresEmbeddingCache(sessions)`,
  `QdrantVectorIndex(client, *, model: str, dimension: int, alias: str = "chunks_current")` (its
  `ensure_collection()` creates the physical collection and points `alias` at it; P3 makes
  `open_corpus_services` pass `settings.retrieval.qdrant_collection` so writes and reads use the same alias), `ImportKnowledgeBundle(repository, cache, index, embedder, clock, *,
  embed_batch_size: int, embed_max_concurrent: int)`, `ReleaseService(` same arguments `)`. Factory
  `pharma_agent.infrastructure.corpus_factory.open_corpus_services(settings, *, embedder: Embedder | None = None)`
  yields `CorpusServices(importer, releases, index, embedder)`; tests and the E2E server inject `FakeEmbedder` here.
- Fixture bundle for tests: `backend/tests/fixtures/knowledge_bundle_small/` (2 documents, a
  prose section, a table section, an `index_only` section with `index_entries`, a leaflet with a
  colloquial mapping, glossary with 2 entries, embeddings for a 4-dim fake model
  `fake-embedding-4d`). Loader helper `tests/corpus_fixtures.py::small_bundle() -> KnowledgeBundle`.
  Fake embedder `tests/fakes.py::FakeEmbedder` (deterministic 4-dim vectors from the sha256).

### 3.4 Retrieval (P3)

- `Hit` fields after P3: `chunk_version_id: UUID`, `release_id: UUID`, `collection_id: UUID`,
  `document_key: str`, `section_key: str`, `section_revision_id: UUID`, `ordinal: int`,
  `hydrate_strategy`, `source`, `title`, `section`, `start_page: int | None`,
  `end_page: int | None`, `context_header`, `chunk_text`, `embedding_text`, `kind: str`, `table_key: str | None`,
  `colloquial_mapping`, `term_annotations`, `fusion_score`, `rerank_score`, `matched_queries`.
  `chunk_id`, `section_id`, `chunk_index`, `content_type`, `table_id` are removed; `embedding_text`
  stays because the reranker scores it, as it does today. `page_label` handles `None`.
- `Chunk` fields after P3: `chunk_version_id: UUID`, `section_revision_id: UUID`, `ordinal: int`,
  `text: str`, `kind: str`, `table_key: str | None`, `start_page: int | None`, `end_page: int | None`.
  `is_table` is `kind == "table"`.
- `Citation` (domain/conversation/models.py) after P3: `index: int`, `chunk_version_id: UUID`,
  `release_id: UUID`, `strategy: HydrateStrategy`, `block_chunk_version_ids: list[UUID]`,
  `source: str`, `title: str`, `section: str`, `start_page: int | None`, `end_page: int | None`,
  `snippet: str`. P3 keeps storing it in `messages.citations` JSONB; P6 moves it to
  `message_citations`.
- Evidence event items after P3: `{"index", "source", "title", "section", "start_page", "end_page", "snippet"}` (snake_case in `ProgressEvent.data`; P6 converts to camelCase on the wire).
- `pharma_agent.domain.retrieval.ports.CorpusReader` implemented by
  `pharma_agent.infrastructure.retrieval.postgres_corpus.PostgresCorpusReader`;
  `PostgresHydrator` in the same module.
- Settings: `RetrievalSettings.qdrant_collection: str = "chunks_current"`,
  `RetrievalSettings.collections: list[str] = ["formulary"]`,
  `RetrievalSettings.mode: Literal["hybrid", "dense", "bm25"] = "hybrid"` (`bm25` = sparse only, no query
  embedding, used by the evaluation baseline); `collection_alias` removed.
- `pharma_agent.infrastructure.composition.build_retrieval_service(settings: Settings, *, database: Database | None = None, embedder: Embedder | None = None) -> RetrievalStack`
  (both keyword arguments are added in P3; P4 injects a cached query embedder, P7 reuses it)
  where `RetrievalStack` is a dataclass with `service: RetrievalService` and
  `async def aclose(self) -> None`. The seed-pipeline evaluation uses this.
- Health check name `corpus`; failure reason code `CORPUS_NOT_READY`.

### 3.5 Web API (P5–P7)

- Route function names become operation ids (`generate_unique_id_function=lambda route: route.name`):
  `health`, `chat`, `chat_stream`, `create_conversation`, `list_conversations`,
  `get_conversation`, `list_messages`, `rename_conversation`, `delete_conversation`,
  `get_message_citation`, `submit_feedback`, `list_skills`, `upload_skill`,
  `set_skill_enabled`, `delete_skill`; fastapi-users keeps its own names
  (`auth:cookie.login`, `auth:cookie.logout`, `auth:jwt.login`, `auth:jwt.logout`,
  `register:register`, `users:current_user`, `users:patch_current_user`, `users:user`,
  `users:patch_user`, `users:delete_user`, `oauth:google.cookie.authorize`,
  `oauth:google.cookie.callback`).
- Response model names: `ConversationView`, `ConversationPage`, `UIMessage`, `MessagePage`,
  `MessageMetadata`, `CitationDetail`, `CitationChunk`, `FeedbackView`, `SkillView`,
  `HealthResponse`, `Problem`, `ProblemItem`, `EvidenceItem`, `PharmaSourceMetadata`.
- Error `type` is `urn:pharma-agent:problem:<code in lower kebab case>`.
- Cursor helpers: `pharma_agent.application.pagination` with `encode_cursor(timestamp: datetime, id: str) -> str` and `decode_cursor(cursor: str) -> tuple[datetime, str]` raising `InvalidCursor` (code `INVALID_CURSOR`).
- Export command: `pharma-agent export-openapi --output PATH`.
- Stream encoder module: `pharma_agent.api.ui_stream`.
- Contract fixtures: `backend/tests/contract/fixtures/ui-stream/<scenario>.sse` with scenarios
  `completed-with-citations`, `blocked`, `timeout`, `persist-failed`, `no-evidence`.
- E2E server: `backend/tests/e2e/server.py`, run `uv run python -m tests.e2e.server --port 8001`;
  scenario markers in the question: `[e2e:blocked]`, `[e2e:timeout]`, default answers with `[1]`.

## 4. Pinned frontend names (P8–P10)

- `frontend/openapi.json` generated by `uv run --directory ../backend pharma-agent export-openapi --output ../frontend/openapi.json`; npm script `api:openapi` runs it, `api:generate` runs orval.
- orval outputs (orval 8.32 file layout, as written in P8): `app/api/gen/endpoints.ts` (react-query hooks and query options), `app/api/gen/schemas/` with `index.ts` (types, imported as `~/api/gen/schemas`), `app/api/gen/zod.ts`, `app/api/gen/endpoints.msw.ts` (MSW handlers, imported as `~/api/gen/endpoints.msw`). orval runs with `override.fetch.includeHttpResponseReturnType: false`, `query.version: 5`, `useInfinite: true`, `useInfiniteQueryParam: "cursor"`. Mutator `app/api/fetcher.ts` exports `fetcher<T>(url: string, init?: RequestInit): Promise<T>`.
- `app/api/problem.ts` exports `class ApiError extends Error { status: number; code: string; detail: string | undefined; errors: ProblemItem[] }` and `isApiError(value: unknown): value is ApiError`.
- `app/lib/csrf.ts` exports `readCsrfToken(): string | undefined` and `CSRF_HEADER = "x-csrftoken"`.
- `app/api/query-client.ts` exports `queryClient`.
- i18n namespaces: `common`, `auth`, `chat`, `citations`, `skills`, `settings`, `landing`, `errors` (error messages keyed by problem `code`).
- Chat feature entry: `app/features/chat/ChatThread.tsx`, `app/features/conversations/ConversationSidebar.tsx`, `app/features/citations/CitationSheet.tsx`, `app/features/chat/lib/cite-markers.ts` exporting `toCiteRefMarkup(text: string): string`.
- Registry components folder: `app/components/elements/`.
- Test support (P8): MSW `server` in `tests/msw/node.ts`, `worker` in `tests/msw/browser.ts`, Vitest setup files in `tests/setup/` (`fail-on-console.ts`, `msw-node.ts`, `msw-browser.ts`), `tests/utils/providers.tsx` (`TestProviders`, `createTestQueryClient`), `tests/utils/i18n.ts` (`createTestI18n`). Browser tests are `app/**/*.browser.test.tsx`; unit tests `app/**/*.test.{ts,tsx}`.
- Toast: `~/components/ui/toast` exports `Toaster` and the shared manager `toast` (`toast.add({ title, description, type })`).
- Actions: `POST /actions/locale` (`routes/actions/locale.ts`) and `POST /actions/theme` (`routes/actions/theme.ts`, remix-themes). Landing paths `/` (vi) and `/en/` (en).
- Passwords entered in the UI (register, change password) require at least 8 characters.
- Commit messages in plans say they end with the session attribution trailer of the executing session; plans never hard-code a session URL.
- Playwright config `frontend/playwright.config.ts`; E2E specs in `frontend/tests/e2e/`.
