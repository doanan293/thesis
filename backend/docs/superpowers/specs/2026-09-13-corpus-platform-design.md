# Thiết kế corpus platform (backend) và seed-pipeline

Ngày: 2026-09-13. Trạng thái: đã duyệt qua brainstorming, chờ implementation plan.

Spec liên quan, cùng đợt:

- `backend/docs/superpowers/specs/2026-09-13-web-client-contract-design.md` (contract cho web client, gọi tắt là spec A).

Spec này ghi lại kiến trúc corpus hiện hành và thay thế mô tả lưu payload cũ bằng Postgres là nguồn chính.

## 1. Mục tiêu

1. Mỗi đoạn corpus có **danh tính bền vững và nội dung bất biến**, để citation luôn trỏ đúng đoạn văn mà câu trả lời đã dựa vào, kể cả sau khi build lại corpus.
2. **Postgres là nguồn chính của corpus** (schema `corpus`), Qdrant chỉ là index dẫn xuất, dựng lại được từ Postgres.
3. **Mọi năng lực corpus lúc chạy nằm trong backend**: schema, import, chia chunk, embedding, index, release, retrieval. `corpus-pipeline` đổi tên thành `seed-pipeline` và chỉ còn là công cụ offline tạo dữ liệu gốc từ nguồn nội bộ.
4. Schema **sẵn cho roadmap** (corpus cá nhân của người dùng, dữ liệu public có trang SEO) nhưng đợt này chưa làm các tính năng đó.

### 1.1 Vấn đề của hệ thống hiện tại

| Vấn đề | Chi tiết |
| --- | --- |
| ID theo vị trí | `chunk_id = "{section_id}:chunk-{index:03d}"`, `chunk_index` đếm theo block trong section. Đổi `max_chars`, sửa đoạn phía trước hoặc curate lại bảng là ID trỏ sang nội dung khác mà không báo lỗi |
| `section_id` không hoàn toàn ổn định | Slug của tiêu đề, trùng thì thêm `:part-002` theo vị trí |
| Không có tầng document | Section chỉ được nhóm ngầm theo `title` |
| Backend chỉ biết alias | `retrieval_runs.corpus_version` đang lưu tên alias nên không đổi qua các lần build |
| Citation không có nội dung | `Citation` chỉ có `index, chunk_id, section_id, title, section, pages, table_id`, không xem lại được đoạn văn |
| Quy tắc cũ | `corpus-pipeline/docs/guides/downstream.md` mục 10 và spec backend §8.1 cấm corpus vào Postgres, không ghi lý do kỹ thuật; thực chất chỉ để tách trạng thái app khỏi corpus. Spec này giữ tinh thần tách biệt bằng schema `corpus` riêng |

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
| --- | --- |
| Phạm vi đợt này | Nền tảng corpus cho dữ liệu hệ thống (Dược thư Quốc gia + tờ hướng dẫn sử dụng An Khang). Schema có sẵn `owner_user_id` và `visibility`. Chưa làm upload corpus cá nhân và trang public |
| Ranh giới | Backend sở hữu corpus lúc chạy. `seed-pipeline` là công cụ offline, bàn giao qua gói import, **không ghi thẳng vào DB backend** |
| Dạng bàn giao | Tri thức có cấu trúc: document → section → block (văn bản/bảng), kèm dữ liệu curate (glossary, colloquial mapping) |
| Chia chunk | Một bộ chunker duy nhất trong backend. `seed-pipeline` import nó như thư viện |
| Embedding dữ liệu gốc | Tính trước trên Kaggle bằng chunker của backend. Gói import có thể kèm vector theo `(model, sha256(embedding_text))`. Backend không biết gì về Kaggle, chỉ embed qua endpoint đã cấu hình |
| Evaluation retrieval | Ở lại `seed-pipeline`, bước retrieve gọi `RetrievalService` của backend như thư viện |
| Tên project offline | `corpus-pipeline` → `seed-pipeline`, package `seed_pipeline`, CLI `seed` |
| Tổ chức Qdrant | Một collection cho mỗi embedding model, phân vùng bằng payload theo khuyến nghị multitenancy của Qdrant |
| Citation | Lưu tham chiếu tới chunk version bất biến (FK `RESTRICT`), không lưu bản sao text |
| Môi trường | Đang phát triển: được reset Postgres và Qdrant, không cần backfill hay tương thích dữ liệu cũ |

## 3. Bố cục repo sau thay đổi

```text
thesis/
  backend/            pharma-agent: API, agent, corpus platform
  seed-pipeline/      công cụ offline: nguồn nội bộ → knowledge bundle, embed Kaggle, evaluation
  frontend/           web client (spec B)
  docker/             llama.cpp, nginx
  docker-compose.yml
  ai-models/
  ruff.toml
  .pre-commit-config.yaml
```

- `seed-pipeline/pyproject.toml` khai báo `pharma-agent = { path = "../backend", editable = true }`. Backend **không** import `seed_pipeline`.
- Thống nhất `qdrant-client>=1.19,<2` ở cả hai project (hiện `corpus-pipeline` ghim `1.18.0`).
- Cả hai project giữ chuẩn chung của repo: ruff dùng chung `ruff.toml`, pyrefly strict, pytest `filterwarnings = ["error"]`, không ignore rule.

## 4. Bố cục code corpus trong backend

Theo quy tắc phân tầng hiện có (`api → application → domain`, `infrastructure → domain`; domain không import sqlalchemy, qdrant_client, fastapi; có test AST cưỡng chế).

```text
backend/src/pharma_agent/
  domain/corpus/
    bundle.py          model Pydantic của knowledge-bundle/v1 (strict, extra="forbid")
    identity.py        chuẩn hoá text, canonical JSON, section_revision_id, chunk_version_id
    chunking.py        chunk_section() thuần, CHUNKER_VERSION, hằng số kích thước
    enrichment.py      phát hiện thuật ngữ theo glossary, colloquial mapping, embedding_text
    hydrate.py         hydrate policy
    models.py          Collection, Document, Section, SectionRevision, ChunkVersion, Release
    ports.py           CorpusRepository, EmbeddingCache, VectorIndex, CorpusReader
  application/corpus/
    import_bundle.py   ImportKnowledgeBundle
    releases.py        publish, rollback, gc, reindex
  infrastructure/
    persistence/postgres/corpus_tables.py, corpus_repository.py, embedding_cache.py
    retrieval/qdrant_index.py        tạo collection, upsert, cập nhật payload, xoá point
    retrieval/qdrant_adapter.py      QdrantHybridRetriever (sửa)
    retrieval/postgres_corpus.py     PostgresCorpusReader, PostgresHydrator
  cli.py                              nhóm lệnh `pharma-agent corpus ...`
```

API công khai mà `seed-pipeline` được phép import (ghi rõ trong docstring module, có test import):

- `pharma_agent.domain.corpus.bundle`
- `pharma_agent.domain.corpus.identity`
- `pharma_agent.domain.corpus.chunking`
- `pharma_agent.domain.corpus.enrichment`
- `pharma_agent.domain.corpus.hydrate`
- `pharma_agent.infrastructure.composition.build_retrieval_service(settings, *, embedder=None)` (cho evaluation; evaluation truyền embedder đọc cache query embedding đã tính sẵn)

## 5. Knowledge bundle `knowledge-bundle/v1`

### 5.1 Cấu trúc

```text
bundle/
  manifest.json
  documents.jsonl
  sections.jsonl
  glossary.json
  colloquial_mappings.json
  embeddings/<model_slug>.jsonl      tuỳ chọn, mỗi model một file
```

| File | Nội dung |
| --- | --- |
| `manifest.json` | `schema_version: "knowledge-bundle/v1"`, `collection: {key, title}`, `generator: {name, version, build_id}`, digest nguồn (`source_pdf_sha256`, `snapshot_sha256`, `curated_input_digests`), số lượng document/section, `files: {tên: {sha256, bytes}}`, `embeddings: [{model, dims, file}]` |
| `documents.jsonl` | `{key, kind, title, source: {title, url}, attributes}`; `kind ∈ {drug_monograph, general_monograph, leaflet}`; `url` là `null` nếu không có |
| `sections.jsonl` | `{key, document_key, heading, context_path[], ordinal, start_page, end_page, retrieval, blocks[]}`; `retrieval ∈ {default, index_only}` |
| `blocks[]` | `{kind, markdown, start_page, end_page, table_key}`; `kind ∈ {prose, table, list, index_entries}`; trang là số nguyên từ 1 hoặc `null` |
| `glossary.json` | Giữ nguyên schema của `data/resources/term_glossary.json` hiện tại |
| `colloquial_mappings.json` | Mảng JSON các record `{key, aliases[], visual_sign, product_names[], section_keys[]}`: field lấy từ `data/resources/colloquial_mappings.json` (tệp nguồn là object theo slug), `key` là slug An Khang hoặc rỗng, mỗi section thuộc tối đa một mapping |
| `embeddings/<model_slug>.jsonl` | `{embedding_text_sha256, dims, vector}`; `vector` là float32 little-endian mã hoá base64 |

### 5.2 Key

- Document: `drug:<slug>`, `general:<slug>`, `leaflet:ankhang:<category>:<slug>`. Với An Khang, `source.url` dựng lại từ key (`https://www.nhathuocankhang.com/<category>/<slug>`).
- Section: **giữ nguyên định dạng `section_id` hiện tại** (ví dụ `drug:paracetamol:lieu-luong-va-cach-dung`), để `expected_section_ids` trong bộ gold evaluation vẫn dùng được.
- Block `kind` và section `retrieval` thay cho các ID viết cứng trong code cũ (`BRAND_INDEX_SECTION_ID` → `index_entries` + `index_only`; `APPENDIX_LIST_SECTION_IDS` → `list`).

### 5.3 Validate khi import

- `schema_version` khớp, sha256 và kích thước từng file khớp manifest.
- Key duy nhất; `section.document_key` tồn tại; `ordinal` không trùng trong một document.
- Block có `markdown` không rỗng; trang hợp lệ (`start_page <= end_page` khi cả hai có giá trị).
- Vector trong `embeddings/`: `dims` khớp manifest, độ dài byte bằng `dims * 4`.
- Lỗi validate liệt kê đủ vị trí (file, dòng, field), import dừng trước khi ghi DB.

## 6. Danh tính và schema Postgres `corpus`

### 6.1 Hàm danh tính (`domain/corpus/identity.py`)

- Chuẩn hoá text: Unicode NFC, `\r\n` → `\n`, bỏ khoảng trắng cuối dòng.
- Canonical JSON: key sắp xếp, không khoảng trắng thừa, text đã chuẩn hoá.
- `NS = uuid5(NAMESPACE_URL, "pharma-agent:corpus:v1")`.
- `section_revision_id = uuid5(NS, "section-revision\x1f" + section_key + "\x1f" + sha256(canonical_json(blocks)))`.
- `chunk_version_id = uuid5(NS, "chunk\x1f" + sha256(section_key + "\x1f" + chunk_text + "\x1f" + embedding_text + "\x1f" + CHUNKER_VERSION))`.
- `embedding_text_sha256 = sha256(embedding_text)` (hex).

`embedding_text` nằm trong ID nên đổi glossary hoặc colloquial mapping sẽ sinh chunk version mới, vector không bao giờ lệch với text.

### 6.2 Bảng

Migration Alembic của backend tạo schema `corpus`; `env.py` bật `include_schemas=True` và `include_name` nhận schema này.

| Bảng | Cột chính | Ràng buộc | Bất biến |
| --- | --- | --- | --- |
| `collections` | `id uuid, key, title, owner_user_id uuid null, visibility, current_release_id uuid null, created_at, updated_at` | `key` unique; `visibility ∈ {private, public}`; `owner_user_id` FK `user` `ON DELETE CASCADE`; `current_release_id` FK `releases` deferrable | Không |
| `documents` | `id, collection_id, key, kind, title, source_title, source_url, attributes jsonb` | unique `(collection_id, key)` | Không, chỉ metadata |
| `sections` | `id, document_id, key, heading, context_path text[], ordinal, retrieval_mode` | unique `(document_id, key)` | Không; danh tính lâu dài cho URL và SEO |
| `section_revisions` | `id, section_id, blocks jsonb, start_page, end_page, char_count` | `id` từ §6.1 | Có |
| `chunk_versions` | `id, section_revision_id, ordinal, kind, context_header, chunk_text, embedding_text, embedding_text_sha256, start_page, end_page, table_key, term_annotations jsonb, colloquial jsonb, chunker_version, created_at` | FK `section_revisions` `RESTRICT`; index `embedding_text_sha256` | Có |
| `releases` | `id, collection_id, number, status, bundle_digest, chunker_version, embedding_model, stats jsonb, created_at, ready_at, published_at, retired_at` | unique `(collection_id, number)`; `status ∈ {building, ready, retired}` | Chỉ trạng thái |
| `release_chunks` | `release_id, chunk_version_id, section_id, section_revision_id, ordinal, hydrate_strategy` | PK `(release_id, chunk_version_id)`; index `(release_id, section_revision_id, ordinal)`; FK chunk `RESTRICT` | Có |
| `glossary_entries` | `release_id, term, data jsonb` | FK release `CASCADE` | Có |
| `colloquial_mappings` | `release_id, key, data jsonb` | FK release `CASCADE` | Có |
| `embedding_cache` | `model, embedding_text_sha256, dims, vector bytea, created_at` | PK `(model, embedding_text_sha256)` | Có |

- Release "đang dùng" của một collection là `collections.current_release_id`. `published_at` ghi lần đầu release được trỏ tới, dùng cho rollback.
- `hydrate_strategy` nằm ở `release_chunks` vì nó phụ thuộc cả section revision (độ dài) chứ không chỉ text của chunk.
- Bảng `message_citations` (schema `public`, FK tới `corpus.chunk_versions`) được định nghĩa ở spec A.

## 7. Chunker và enrichment

### 7.1 `chunk_section()`

```python
CHUNKER_VERSION = "chunker-v1"
MAX_CHUNK_CHARS = 3000
FULL_SECTION_MAX_CHARS = 16000

def chunk_section(
    section: SectionInput,          # key, heading, context_path, retrieval, blocks
    glossary: Sequence[GlossaryEntry],
    mappings: Sequence[ColloquialMapping],
    *, max_chars: int = MAX_CHUNK_CHARS,
) -> list[ChunkDraft]: ...
```

- Duyệt block theo thứ tự, chia theo `kind`:
  - `table`: chia bảng markdown, mỗi phần giữ dòng header (port từ `split_table_markdown`).
  - `index_entries`, `list`: cắt theo dòng, không làm vỡ một mục (port từ `split_lines_without_breaking_entries` và `split_brand_index_text`).
  - `prose`: port từ `split_long_text`.
- Section không có block nào có text: fallback chia toàn bộ text section như `chunk_sections` hiện tại.
- Trang của chunk lấy từ block, thiếu thì lấy từ section.
- `context_header` dựng từ heading và `context_path` như `build_context_header` hiện tại.
- `embedding_text` ghép như `_embedding_text` hiện tại: context header → colloquial text → chunk text → term search text (nếu chưa có "thuật ngữ:").
- `term_annotations` từ `detect_term_enrichments` trên `context_header + chunk_text`.
- `CHUNKER_VERSION` tăng khi thuật toán đổi; đổi version bắt buộc tạo release mới.

### 7.2 Hydrate policy (`domain/corpus/hydrate.py`)

| Điều kiện | Strategy |
| --- | --- |
| `retrieval = index_only` | `search_only` |
| `section_revisions.char_count > 16000` | `chunk_window` (bán kính lấy từ setting `retrieval.hydrate_window`, mặc định 1) |
| Còn lại | `full_section` |

### 7.3 Yêu cầu khớp với hệ thống cũ

Test so khớp chạy trên bundle xuất từ build hiện tại: với mọi section, output của `chunk_section` phải trùng 100% với `chunks.jsonl` cũ về `chunk_text`, trang, `hydrate_strategy`, `embedding_text`, `term_annotations` và `colloquial_mapping`; chỉ khác ID. Test này dùng trong giai đoạn chuyển đổi (§11) rồi được thay bằng test golden trên fixture nhỏ.

## 8. Import, embedding, index và release

### 8.1 Lệnh

| Lệnh | Việc |
| --- | --- |
| `pharma-agent corpus import <bundle_dir> --collection <key> [--publish]` | Import bundle thành một release mới |
| `pharma-agent corpus releases [--collection <key>]` | Liệt kê release, trạng thái, số chunk, release hiện hành |
| `pharma-agent corpus publish <release_id>` | Trỏ `current_release_id` tới release `ready` |
| `pharma-agent corpus rollback --collection <key>` | Trỏ về release có `published_at` liền trước |
| `pharma-agent corpus gc --collection <key> --keep 2` | Retire release cũ và dọn dữ liệu (§8.5) |
| `pharma-agent corpus reindex --collection <key>` | Dựng lại point Qdrant cho các release chưa retire từ Postgres và `embedding_cache` (ví dụ sau khi xoá Qdrant) |

Sau này API upload corpus cá nhân gọi đúng `ImportKnowledgeBundle`, không có đường ingest thứ hai.

### 8.2 Luồng `ImportKnowledgeBundle`

1. **Validate** bundle (§5.3).
2. **Không có thay đổi**: nếu `bundle_digest` và `CHUNKER_VERSION` giống release hiện hành thì báo "không có thay đổi" và dừng.
3. **Upsert danh tính**: `collections`, `documents`, `sections` theo key (chỉ cập nhật metadata).
4. **Section revision**: tính ID, insert nếu chưa có.
5. **Chunk và enrich**: `chunk_section()`, tính `chunk_version_id`, insert nếu chưa có.
6. **Tạo release** `status=building`, ghi `release_chunks`, `glossary_entries`, `colloquial_mappings`.
7. **Embedding** cho từng `embedding_text_sha256` của release:
   1. Có trong `embedding_cache` với model hiện tại → dùng.
   2. Có trong `embeddings/<model_slug>.jsonl` của bundle, model và dims khớp → ghi vào cache.
   3. Còn thiếu → gọi embedding endpoint theo lô (`corpus.embed_batch_size`, mặc định 32; `corpus.embed_max_concurrent`, mặc định 1), ghi cache sau mỗi lô. Chạy lại lệnh là resume.
8. **Index Qdrant**: upsert point cho chunk version chưa có point; chunk đã có point thì thêm `release_id` vào payload `release_ids`.
9. **Kiểm tra**: số point có `release_id` của release bằng số `release_chunks` → `status=ready`, ghi `stats`.
10. **Publish** (khi có `--publish`): một transaction đặt `current_release_id` và `published_at`.

Mọi bước idempotent theo ID băm; lỗi giữa chừng để lại release `building`, chạy lại tiếp tục từ cache và point đã có.

### 8.3 Qdrant

| Mục | Thiết kế |
| --- | --- |
| Collection | `chunks_<model_slug>` cho mỗi embedding model; vector `dense_vector` (dims theo model, cosine) và sparse `bm25_sparse_vector` (modifier IDF) |
| Alias | `chunks_current`; chỉ đổi khi đổi embedding model |
| Point ID | `chunk_version_id` |
| Payload | `collection_id` (keyword, `is_tenant=true`), `release_ids[]` (keyword), `document_id`, `section_id`, `kind` (keyword). **Không chứa text** |
| BM25 | Upsert `models.Document(text=embedding_text, model="Qdrant/bm25")`; Qdrant server tự suy luận |
| Metadata collection | `{embedding_model, dims}`; backend kiểm tra khi khởi động |
| Truy vấn | Filter `collection_id ∈ phạm vi` và, với từng collection, `release_ids = current_release_id` |

### 8.4 Dung lượng

`embedding_cache` cho khoảng 25k chunk × 2560 chiều × 4 byte ≈ 256 MB. Đổi lại, dựng lại Qdrant hoặc đổi layout không bao giờ phải embed lại, và build release mới chỉ embed chunk có `embedding_text` thay đổi.

### 8.5 `gc`

1. Retire mọi release không phải release hiện hành và không nằm trong `--keep` release gần nhất.
2. Gỡ `release_id` của release đã retire khỏi payload; xoá point có `release_ids` rỗng (giữ thống kê IDF của BM25 sạch).
3. Xoá `release_chunks`, `glossary_entries`, `colloquial_mappings` của release đã retire.
4. Xoá `chunk_versions` không còn trong `release_chunks` nào; chunk đang được `message_citations` tham chiếu bị FK `RESTRICT` chặn và được giữ lại. Tương tự cho `section_revisions`.
5. `embedding_cache` không bị dọn trong đợt này.

## 9. Retrieval trong backend

Domain retrieval giữ nguyên các port `Retriever`, `Reranker`, `Hydrator` và `RetrievalService`.

| Thành phần | Thay đổi |
| --- | --- |
| `Hit`, `Chunk` | Giữ `embedding_text` (reranker chấm trên trường này như hiện nay); `chunk_id` → `chunk_version_id`; thêm `release_id`, `collection_id`, `document_key`, `section_key`, `section_revision_id`, `ordinal`, `start_page`, `end_page` cho từng chunk; trang dùng `None` thay cho 0 |
| Port mới `CorpusReader` | `load_chunks(ids) -> list[ChunkRecord]`, `section_chunks(release_id, section_revision_id, around: int | None, radius: int) -> list[Chunk]`, `current_releases(collection_ids) -> dict[collection_id, release_id]` |
| `QdrantHybridRetriever` | Chế độ `hybrid` (RRF dense + BM25), `dense`, `bm25` (chỉ sparse, làm baseline evaluation); thêm filter §8.3; nhận ID + score từ Qdrant rồi gọi `CorpusReader.load_chunks` để dựng `Hit` (text, header, trang, thuật ngữ, colloquial) |
| `PostgresHydrator` | Thay `QdrantHydrator`: `full_section` lấy toàn bộ `release_chunks` của section revision theo `ordinal`; `chunk_window` lấy `ordinal ± radius`; `search_only` trả rỗng |
| `Reranker` | Giữ nguyên |
| Phạm vi | Setting `retrieval.collections = ["formulary"]`; release hiện hành đọc từ Postgres mỗi lần search (một query có index, không cache) |
| Setting | Bỏ `retrieval.collection_alias`; thêm `retrieval.qdrant_collection = "chunks_current"`, `retrieval.collections` |
| Health | `/health` thêm check `corpus`: collection Qdrant tồn tại, metadata `embedding_model`/`dims` khớp setting, mọi collection trong phạm vi có release hiện hành. Sai thì `degraded` với lý do `CORPUS_NOT_READY` |
| Audit | `retrieval_runs.corpus_version` → `release_ids jsonb` (map collection → release); `retrieval_hits.chunk_id`/`section_id` → `chunk_version_id uuid`, `section_key text`. Audit **không có FK** tới corpus để không chặn `gc` |
| Citation | `citations_from` dựng `Citation` từ `Evidence`: `index, chunk_version_id, release_id, strategy, block_chunk_version_ids` (theo đúng thứ tự mô hình đã đọc) và thông tin hiển thị; lưu vào `message_citations` (spec A) |

## 10. seed-pipeline

### 10.1 Đổi tên

- Thư mục `corpus-pipeline/` → `seed-pipeline/`; package `corpus_pipeline` → `seed_pipeline`; CLI `corpus` → `seed`; `[project].name = "seed-pipeline"`.
- Slug dataset Kaggle `corpus-pipeline-rag-final` → `seed-pipeline-bundle`; tiền tố workspace `corpus-pipeline-kaggle-`/`corpus-pipeline-stage-` → `seed-pipeline-kaggle-`/`seed-pipeline-stage-`.
- Cập nhật `README.md` gốc, `.pre-commit-config.yaml` (hook pyrefly), `backend/README.md`, `seed-pipeline/README.md`, `seed-pipeline/docs/guides/*`. Spec và plan cũ của backend là lịch sử, giữ nguyên tên cũ.

### 10.2 Giữ nguyên

Snapshot An Khang, xử lý PDF, clean, tách section, curate bảng Docling, tích hợp An Khang, validate và deep audit của tri thức có cấu trúc.

### 10.3 Lệnh mới

| Lệnh | Việc |
| --- | --- |
| `seed bundle export --output <dir>` | Từ canonical sections và blocks, xuất `knowledge-bundle/v1`: gom document, gán `blocks[].kind` và `retrieval` từ các quy tắc hiện có, chép glossary và colloquial mapping, ghi manifest. Validate bằng `pharma_agent.domain.corpus.bundle` trước khi ghi |
| `seed bundle embed --bundle <dir> --backend kaggle\|local --model <model>` | Chạy `chunk_section` của backend ở máy local để lấy các cặp `(embedding_text_sha256, embedding_text)` duy nhất, dùng lại hạ tầng embed Kaggle hiện có (checkpoint theo hash, resume), ghi `embeddings/<model_slug>.jsonl` và cập nhật manifest. Kernel Kaggle chỉ nhận text, không import backend |

### 10.4 Evaluation

- `seed evaluation build` (bộ câu hỏi gold, sinh câu hỏi, rejudge) giữ nguyên.
- `seed retrieve --run <name>` thay các retriever riêng bằng `build_retrieval_service(settings)` của backend, chạy trên Postgres và Qdrant dev đã import release. Các chế độ `bm25`, `dense`, `hybrid` và tham số K được truyền vào qua cấu hình retrieval của backend.
- `seed embed queries` và `seed rerank` trên Kaggle giữ nguyên; candidate bundle lấy từ retrieval của backend.
- Định danh run ghi thêm `release_id`, `chunker_version`, `embedding_model`.
- Gold label theo section key, không đổi.

### 10.5 Xoá

`compile_unified_chunks`, `build_rag_metadata.build_qdrant_payload`, `qdrant_payload_contract.py`, gói `vector_store/` (upload, ingest, point ID namespace, cache embedding chunk cũ), contract `chunks.jsonl` trong `rag-final/`, lệnh `corpus vectors upload` và `corpus embed chunks`, schema app cũ `integrations/postgres/schema/rag_app_schema.sql`.

### 10.6 Tài liệu

`downstream.md` viết lại: output của seed-pipeline là knowledge bundle; mục 10 thay bằng "corpus lúc chạy nằm trong schema `corpus` của backend, nạp bằng `pharma-agent corpus import`".

## 11. Chuyển đổi (môi trường dev)

1. **Backend**: schema `corpus`, domain corpus, import, Qdrant layout mới, `CorpusReader`, `PostgresHydrator`, audit, setting và health.
2. **seed-pipeline**: đổi tên, `seed bundle export`.
3. **Kiểm tra khớp chunk** (§7.3): chạy trên bundle xuất từ build hiện tại, phải trùng 100%. Máy hiện chưa có `data/heavy/processed/`, cần chạy `seed build` trước.
4. **Embedding**: `seed bundle embed --backend kaggle --model qwen3-embedding:4b-fp16`.
5. **Reset dev**: xoá collection Qdrant `thesis_chunks_*`, dựng lại DB từ migration, chạy `pharma-agent corpus import <bundle> --collection formulary --publish`.
6. **Kiểm tra chất lượng**: chạy benchmark hybrid + rerank qua evaluation mới; so với kết quả `hybrid-qwen4b-p50-k30-rrf2` + rerank gần nhất. Chấp nhận khi Hit@10 và MRR không giảm quá 1 điểm phần trăm.
7. **Dọn**: xoá code và tài liệu cũ theo §10.5, §10.6; cập nhật `docker-compose.yml` (bỏ `PHARMA_RETRIEVAL__COLLECTION_ALIAS`, thêm hướng dẫn import).

## 12. Testing

| Loại | Nội dung |
| --- | --- |
| Unit domain | `chunk_section` theo từng `kind` (bảng giữ header, danh mục không vỡ mục, prose dài); fallback section không có block; hàm danh tính cho cùng kết quả với cùng input và khác khi đổi `CHUNKER_VERSION`/glossary; hydrate policy; validate bundle với từng loại lỗi |
| Unit application | `ImportKnowledgeBundle` với repository/index/embedder giả: dùng vector từ cache, từ bundle, gọi embedder cho phần thiếu; import lại không sinh dữ liệu; bundle không đổi thì dừng; trạng thái release |
| Integration (testcontainers Postgres + Qdrant, embedder giả) | Bundle fixture nhỏ có bảng và danh mục: import → search hybrid → hydrate `full_section`/`chunk_window` → publish/rollback đổi kết quả → `gc` giữ chunk có citation (FK) và xoá point thừa → `reindex` sau khi xoá collection |
| Contract | Test của seed-pipeline validate bundle xuất ra bằng model của backend; fixture bundle dùng chung được kiểm tra ở cả hai phía |
| Khớp dữ liệu cũ | §7.3 và §11 bước 6, chạy một lần khi chuyển đổi |

## 13. Ngoài scope

- Upload và xử lý corpus cá nhân (parse file, form nhập tri thức), incremental sync (CocoIndex hoặc tương đương), hàng đợi worker.
- Trang public của corpus và SEO.
- Tự động hoá chuyển đổi embedding model (tạo collection mới, đổi alias).
- Dọn `embedding_cache`.

## 14. Nhật ký quyết định

| Quyết định | Lựa chọn khác đã cân nhắc | Lý do |
| --- | --- | --- |
| Postgres là nguồn chính, Qdrant là index | Chỉ lưu snapshot text trong tin nhắn; ID theo hash nhưng vẫn giữ corpus trong Qdrant | Chỉ cách này sửa gốc danh tính, đồng thời phục vụ corpus cá nhân và trang public. Qdrant không có versioning dữ liệu trong collection |
| Backend sở hữu corpus lúc chạy | Package riêng `corpus-store` dùng chung; backend sở hữu schema nhưng pipeline tự ghi SQL; pipeline sở hữu schema | Ranh giới người dùng chốt: phần runtime vào backend, phần tạo dữ liệu gốc tách riêng. Tránh backend kéo torch/docling |
| Bàn giao tri thức có cấu trúc | Bàn giao chunk hoàn chỉnh | Một bộ chunker cho cả dữ liệu gốc lẫn dữ liệu người dùng |
| Vector tính trước theo `(model, hash)` | Backend xuất file riêng cho Kaggle; backend tự embed toàn bộ | Backend không biết Kaggle; seed nhanh nhờ GPU; định dạng import dùng chung |
| Evaluation ở seed-pipeline, gọi retrieval backend | Chuyển retrieve và metrics sang backend | Evaluation là công cụ nghiên cứu offline dùng Kaggle; vẫn đo đúng retrieval chạy thật |
| Một collection Qdrant mỗi model, phân vùng payload | Một collection mỗi release với alias | Khuyến nghị multitenancy chính thức của Qdrant; publish và rollback chỉ đổi con trỏ trong Postgres; sẵn cho corpus cá nhân |
| Citation lưu tham chiếu | Lưu bản sao `cited_text` như Anthropic Citations API | Chunk version bất biến và FK `RESTRICT` bảo đảm không mất nội dung, nên không cần lưu trùng |
| Xem trước citation ở mức khối | Chọn câu khớp bằng code; chấm bằng embedding/reranker; bắt LLM trích câu | Người dùng chọn mức khối; toàn văn xem ở panel chi tiết |
