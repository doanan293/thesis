# Chuẩn hoá dữ liệu seed-pipeline

Ngày: 2026-09-14. Phạm vi: `seed-pipeline/data/`, code seed-pipeline đọc và ghi dữ liệu đó, phần backend chứa dữ liệu corpus (prompt, skill, `corpus gc`), tài liệu liên quan.

## 1. Bối cảnh

`seed-pipeline/data/` nặng 18 GB, gần hết nằm trong `data/heavy/` (bị Git bỏ qua). Dữ liệu do pipeline cũ (`corpus-pipeline`) và pipeline mới sinh ra lẫn lộn:

- Mỗi run evaluation bị tách thành hai cây cùng tên: `data/retrieval_eval/<run>` (Git) và `data/heavy/retrieval_eval/<run>` (payload); manifest metrics ghi hai lần.
- Code sinh tên file và thư mục từ hash: report metrics, biến thể rerank, cache điểm rerank, profile Kaggle, lock, checkpoint, id snapshot.
- Code còn nhánh tương thích ngược (run schema 1, bản ghi cache chưa niêm phong, tên kernel cũ).
- 9 run evaluation hiện có đều do pipeline cũ sinh; 8 run ghi đường dẫn tuyệt đối tới `corpus-pipeline`.
- Tờ hướng dẫn sử dụng thuốc được gọi theo tên website nguồn ("An Khang") trong key tài liệu và section, `source.url`, system prompt, skill, bộ gold, code, test và tài liệu. Production không được lộ nguồn này.

Đã xong trước spec này:

- `seed build` theo contract `rag-final-v3` (build `1f286160109101e4`); nội dung section giống hệt build cũ.
- Sửa `split_table_markdown` của backend; parity với `chunks.jsonl` cũ đạt 0 lệch trên 24.955 chunk.
- Bundle `formulary` có `embeddings/qwen3_embedding_4b_fp16.jsonl`, chuyển từ cache vector cũ, không suy luận lại; stack app đã import release `formulary #1`.
- Đã xoá rác: lock mồ côi trong `heavy/.work/`, input tạm `heavy/.work/bundle-embed/`, `heavy/processed/.rag-final.publish.lock`, 5 profile Kaggle mà code hiện tại từ chối nạp hoặc không khớp chính sách cache.

## 2. Mục tiêu và ngoài phạm vi

Mục tiêu:

1. Layout 5 thư mục cấp cao, tên đọc được; không đường dẫn file hay thư mục cục bộ nào chứa hash.
2. Mọi số liệu evaluation được sinh lại bằng CLI hiện tại và đối chiếu với số cũ; riêng run OpenAI nhập lại candidates cũ.
3. Không còn dữ liệu legacy và code tương thích ngược.
4. Dữ liệu không nằm trong Git được lưu và khôi phục bằng lệnh qua Kaggle dataset private.
5. Tờ hướng dẫn thuốc được gọi là `leaflet` trong mọi định danh, prompt, code, test và tài liệu; nguồn gốc chỉ ghi trong manifest nguồn nội bộ (`sources/leaflets/`) và phần phương pháp của report. Dữ liệu vào DB, API và agent không chứa tên hay URL của website nguồn.

Ngoài phạm vi: thuật toán retrieval và rerank; tên kernel và dataset Kaggle mà job tạo trên Kaggle (tài nguyên từ xa, cần duy nhất theo job); collection `chunks_fake_embedding_4d` và alias `e2e_chunks_current` của test E2E; spec và plan cũ trong `docs/superpowers/` (bản ghi lịch sử cố định) và lịch sử Git.

## 3. Hiện trạng (kiểm ngày 2026-09-14)

| Đường dẫn | Dung lượng | Tình trạng |
| --- | ---: | --- |
| `heavy/raw/duoc-thu-quoc-gia-viet-nam.pdf` | 39 MB | PDF Dược thư |
| `heavy/raw/ankhang/snapshots/ankhang-2026-07-24-97f5b5c43eee.tar.zst` | 12 MB | 2.434 file HTML, 377 MB khi giải nén |
| `heavy/processed/rag-final/` | 76 MB | Build v3 hiện hành |
| `heavy/processed/evaluation/` | 9,8 MB | Bộ gold `section_retrieval_eval.jsonl` (sha256 `b26d9fd7…`), `patient_queries.json` |
| `heavy/bundles/formulary/` | 386 MB | Bundle hiện hành, embeddings model 4B |
| `heavy/cache/text_embeddings/` | 1,4 GB | Model 4B, 24.946 bản ghi |
| `heavy/cache/vector_embeddings/` | 5,0 GB | Legacy; 5 model, mỗi model phủ đủ 24.946 text, `model_sha256` và số chiều khớp catalog |
| `heavy/cache/query_embeddings/` | 2,0 GB | 5 model × 10.000 query, đúng dạng `<model>.jsonl`, khớp catalog |
| `heavy/cache/rerank_scores/` | 1,2 GB | Layout cũ `<model>/<hash>.jsonl`; 5 file khớp catalog, 1 file gemma `native_rerank` không khớp |
| `heavy/runtime_kaggle_profiles/` | 64 KB | Còn 4 profile schema 3 hợp lệ (gemma, bge-m3, qwen3 0,6B, 8B) |
| `heavy/migration/` | 322 MB | Bản sao build cũ, dùng cho parity |
| `heavy/retrieval_eval/` | 7,5 GB | Payload của 9 run cũ |
| `retrieval_eval/`, `manifests/`, `resources/` | 2,4 MB | Git: metadata run cũ, provenance, đầu vào tự duy trì |

Mọi cache query, rerank và text đều không còn bản ghi chưa niêm phong (`load_records(..., allow_legacy=True)` trả `has_legacy=False` cho cả 12 file).

Tên nguồn "ankhang" hiện có trong: 2.406 key tài liệu `leaflet:ankhang:*` kèm `source.url` của website; 2.406 key section `brand:ankhang:*` và `section_keys` của colloquial mapping; 3.941/10.000 dòng bộ gold (`query_id`, `eval_group`, `source_family`, `source_subcategory`, `eval_tags`, `notes`, id mong đợi); system prompt (`domain/agent/prompts.py`), skill `brand-to-generic`, docstring `domain/corpus/bundle.py`; 15 file code và 7 file test seed-pipeline, 8 file test backend; `seed-pipeline/README.md`, `docs/guides/downstream.md`, runbook migration, `README.md` ở root. Citation trả cho frontend chỉ có `source` ("Tờ hướng dẫn sử dụng"), tiêu đề, section, trang và snippet. `pharma-agent corpus gc` không xoá dòng `documents` và `sections` mồ côi.

## 4. Layout đích

```
seed-pipeline/data/
├── README.md
├── sources/          Dữ liệu gốc
│   ├── duoc-thu-quoc-gia-viet-nam.pdf
│   ├── leaflets/     manifest.json, urls/{all_urls.txt, drug_urls.txt}, html/<nhóm>/<slug>.html
│   ├── curation/     docling_tables.jsonl, table_continuation_overrides.json, table_duplicate_overrides.json
│   └── colloquial_mappings.json, term_glossary.json, vietnamese_valid_syllables.json
├── corpus/
│   ├── rag-final/    kết quả `seed build`
│   └── formulary/    knowledge bundle bàn giao cho `pharma-agent corpus import`
├── evaluation/
│   ├── gold/         section_retrieval_eval.jsonl, patient_queries.json
│   └── runs/<run>/   run.json, candidates/, rerank/<model>/, reports/
├── cache/
│   ├── text_embeddings/<model>.jsonl
│   ├── query_embeddings/<model>.jsonl
│   ├── rerank_scores/<model>.jsonl
│   └── kaggle_profiles/<workload>/<model>.json
└── work/             scratch: build workspace, lock, staging archive
```

`<model>` là `ModelSpec.slug` trong catalog (ví dụ `qwen3_embedding_4b_fp16`).

`sources/leaflets/manifest.json` (schema `leaflet-source-v1`) là nơi duy nhất ghi nguồn gốc: sitemap URL, ngày crawl, sha256 danh sách URL, và với mỗi file HTML `path`, `size`, `sha256`, `source_url`. Nó thuộc dữ liệu nội bộ của seed-pipeline, không đi vào bundle.

### 4.1 Git và archive

Git chỉ giữ file nhỏ tự duy trì hoặc cần cho report; phần còn lại nằm trong archive Kaggle (§9). `seed-pipeline/.gitignore` thay dòng `data/heavy/` bằng:

```gitignore
# Large or generated data lives in the Kaggle archive (`seed data push` / `seed data pull`).
data/sources/**/*.pdf
data/sources/leaflets/html/
data/corpus/**
!data/corpus/**/
!data/corpus/**/manifest.json
!data/corpus/**/validation_report.json
data/evaluation/gold/
data/evaluation/runs/**
!data/evaluation/runs/**/
!data/evaluation/runs/**/run.json
!data/evaluation/runs/**/manifest.json
!data/evaluation/runs/**/report.md
data/cache/
data/work/
```

Nội dung archive là mọi file dưới `data/` mà Git bỏ qua, trừ `data/work/`. Hook `check-added-large-files` (2 MB) giữ nguyên để chặn commit nhầm.

## 5. Quy tắc đặt tên

Tên chỉ ghép từ tên run, slug model, tham số đọc được và nội dung dữ liệu; không dùng hash và không dùng tên website nguồn. Hash, digest và identity đầy đủ nằm trong `manifest.json` hoặc trong bản ghi.

| Artifact | Đường dẫn |
| --- | --- |
| Run | `evaluation/runs/<run>/run.json` |
| Candidates | `evaluation/runs/<run>/candidates/candidates.jsonl`, `manifest.json` |
| Biến thể rerank | `evaluation/runs/<run>/rerank/<model>/` |
| Report metrics | `evaluation/runs/<run>/reports/baseline/top<K>-window<N>/`, `reports/rerank/<model>/top<K>-window<N>/` gồm `report.md`, `manifest.json`, `metrics.jsonl` |
| Cache text, query | `cache/text_embeddings/<model>.jsonl`, `cache/query_embeddings/<model>.jsonl` |
| Cache điểm rerank | `cache/rerank_scores/<model>.jsonl` (bản ghi mang `model_sha256`, `request_contract_sha256`) |
| Profile Kaggle | `cache/kaggle_profiles/<workload>/<model>.json` |
| Checkpoint | `<output_dir>/.checkpoints/<model>.jsonl`; bản ghi đầu mang identity |
| Lock | `work/locks/<đường dẫn đích so với data/, "/" thay bằng "__">.lock` |
| Nguồn tờ hướng dẫn | `sources/leaflets/html/<nhóm>/<slug>.html`, `sources/leaflets/manifest.json` |

Định danh trong dữ liệu:

| Định danh | Dạng |
| --- | --- |
| Key tài liệu tờ hướng dẫn | `leaflet:<nhóm>:<slug>` |
| Key section tờ hướng dẫn | `leaflet:<nhóm>:<slug>` |
| `source` tài liệu tờ hướng dẫn | `{"title": "Tờ hướng dẫn sử dụng", "url": null}` |
| Bộ gold | `query_id` `leaflet-…`, `leaflet-alias-…`, `leaflet-alias-any-…`; `eval_group` `leaflet`; `source_family` `leaflet_brand`; `source_subcategory` `leaflet_<khía cạnh>`; tag `leaflet`, `leaflet_brand`, `leaflet_<khía cạnh>` |

Xung đột:

- Run, biến thể rerank, report, checkpoint: đường dẫn đã có mà identity khác thì lệnh dừng với lỗi nêu rõ đường dẫn và trường khác nhau; `--force` thay thế nguyên tử.
- Profile Kaggle: identity khác được coi như chưa có; benchmark mới ghi đè, vì profile sinh lại được.
- Cache text, query, rerank: nhiều bản ghi trong một file, khoá theo nội dung như hiện tại; không đổi.

## 6. Thay đổi code

### 6.1 `seed-pipeline/src/seed_pipeline/config/paths.py`

Hằng số mới: `SOURCES_DIR`, `LEAFLETS_DIR`, `LEAFLETS_HTML_DIR`, `LEAFLETS_MANIFEST_PATH`, `LEAFLETS_URLS_DIR`, `SOURCES_CURATION_DIR`, `FORMULARY_PDF_PATH`, `CORPUS_DIR`, `RAG_FINAL_DIR`, `BUNDLE_DIR` (`corpus/formulary`), `EVALUATION_DIR`, `GOLD_DIR`, `RUNS_DIR`, `CACHE_DIR`, `TEXT_EMBEDDING_CACHE_DIR`, `QUERY_EMBEDDING_CACHE_DIR`, `RERANK_SCORE_CACHE_DIR`, `KAGGLE_PROFILE_DIR`, `WORK_DIR`, `BUILD_WORK_DIR` (`work/build`), `EVALUATION_CHUNKS_PATH` (`work/evaluation-chunks/chunks.jsonl`), `LOCK_DIR` (`work/locks`). Hàm `run_dir(run)` thay `retrieval_run_roots`.

Xoá: `HEAVY_DATA_DIR`, `HEAVY_RAW_DIR`, `RAW_DIR`, `RAW_ANKHANG_*`, `RESOURCES_*`, `RAW_CURATION_DIR`, `MANIFESTS_DIR`, `RETRIEVAL_EVAL_DIR`, `HEAVY_RETRIEVAL_EVAL_DIR`, `DATA_RUNS_DIR`, `RETRIEVAL_EVAL_RUNS_DIR`, `RUNTIME_PROFILE_DIR`, `ANKHANG_MARKDOWN_INTERIM_DIR` (đổi tên theo `leaflet`), `rerank_score_cache_path`, `query_embedding_bundle_dir`, `retrieval_run_roots`. `MIGRATION_DIR` còn đến giai đoạn 4 và bị xoá cùng parity (§10).

### 6.2 Run evaluation

- `run.json` schema 3 gồm `identity`, `origin` (`backend` hoặc `imported`), candidates, biến thể rerank theo slug model và report. Chỉ đọc schema 3; xoá đọc schema 1 và 2, `LegacyRerankReference`, `legacy_status`, `migrate_legacy_rerank`.
- Một cây duy nhất cho mỗi run; bỏ file `candidates-manifest.json` và `metrics-manifest.json` trùng lặp.
- `seed retrieve`, `rerank`, `metrics` ghi và đọc theo §5.

### 6.3 Đặt tên không hash

Sửa các chỗ đang ghép hash vào đường dẫn: `evaluation/metrics_artifacts.py` (`_report_parent`, `_report_dir`), biến thể rerank trong `evaluation/rerank_service.py` và `rerank_artifacts.py`, `runtime/runtime_profiles.py` (`RuntimeProfileStore.path`), `integrations/kaggle/job_lock.py`, checkpoint trong `evaluation/query_embedding_service.py` và `rerank_service.py`. Tên thư mục output trên Kaggle (`integrations/kaggle/stages.py`) và slug kernel/dataset từ xa giữ nguyên.

### 6.4 Nguồn tờ hướng dẫn giữ nguyên dạng

- `seed source crawl` ghi HTML vào `sources/leaflets/html/` và cập nhật `sources/leaflets/manifest.json`; sitemap URL, domain và mọi giá trị riêng của website nguồn đọc từ manifest (tuỳ chọn `--sitemap-url` ghi đè), không còn trong code.
- Các thư mục trung gian của build (`TEXT_INTERIM_DIR`, `DOCLING_INTERIM_DIR`, `CANONICAL_INTERIM_DIR`…) chuyển vào `BUILD_WORK_DIR`.
- `seed build` thay `--snapshot-archive` và `--snapshot-manifest` bằng `--leaflets-dir` (mặc định `sources/leaflets`); kiểm sha256 từng file HTML theo manifest rồi đọc trực tiếp, không giải nén.
- Manifest `rag-final` thay `snapshot_id` và `snapshot_sha256` bằng `leaflet_source` (`manifest_sha256`, `file_count`).
- Xoá `artifacts/snapshot.py` (`pack_snapshot`, `verify_snapshot`, `extract_snapshot`) cùng test.

### 6.5 Tên trung tính

- seed-pipeline: đổi tên module, hàm, hằng số, schema và giá trị định danh theo `leaflet` (ví dụ `corpus/crawling/integrate_ankhang.py` → `integrate_leaflets.py`, `ANKHANG_*` → `LEAFLET_*`); `bundle/export.py` sinh key và `source` theo §5; bộ sinh evaluation (`build_section_retrieval_eval.py`, `patient_query_generation.py`, `section_eval_schema.py`, `metrics_service.py`) sinh `query_id`, nhóm, family, tag và ghi chú theo §5.
- backend: câu trong `domain/agent/prompts.py` thành "…dựa trên Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng thuốc"; `skills/brand-to-generic/SKILL.md` và docstring `domain/corpus/bundle.py` dùng "tờ hướng dẫn sử dụng"; test backend dùng key mới.
- Test fixture seed-pipeline (`tests/fixtures/rag_final_small/`) dùng key mới.

### 6.6 `pharma-agent corpus gc` dọn danh tính mồ côi

Sau khi xoá section revision mồ côi, `purge_retired` xoá tiếp các dòng `sections` không còn section revision nào, rồi các dòng `documents` không còn section nào, trong collection đó; `PurgeResult` thêm `sections_deleted` và `documents_deleted`. Test integration trên Postgres thật: import bundle A, import bundle B có key khác, `gc --keep 1` để lại đúng tài liệu và section của B.

### 6.7 Xoá code tương thích ngược

- `cache/jsonl_records.py`: tham số `allow_legacy` và nhánh bản ghi chưa niêm phong; `query_embedding_cache.py` và `rerank_score_cache.py` bỏ `rewrite_legacy` và việc viết lại bản ghi.
- `evaluation/query_embedding_artifact.py`: nhánh gộp query JSONL cũ; `integrations/kaggle/workers/query_embed.py`: `_legacy_query_records`.
- `integrations/kaggle/kernels.py`: tham chiếu kernel dự phòng `sha256[:8]`.
- Lệnh `seed evaluation rejudge-current`, `evaluation/rejudge_service.py`, `evaluation/rejudging.py` và test của chúng (không module nào khác dùng).

### 6.8 Lệnh `seed data`

`seed data push` và `seed data pull` theo §9, dùng lớp Kaggle sẵn có (`DatasetService`, profile tài khoản trong `seed-pipeline/.env`) và `zstandard` đã có trong dependency.

## 7. Chuyển dữ liệu (một lần)

Chạy bằng script đặt trong `data/work/migration/` (bị Git bỏ qua, không commit). Script đọc các định dạng cũ mà code mới không còn hỗ trợ. Dữ liệu trong `data/heavy/` chỉ được đọc và chép, không sửa, và giữ nguyên tới giai đoạn 4; "Bỏ" trong bảng nghĩa là không chép sang layout mới. File nhỏ trong Git được `git mv`, lịch sử Git giữ bản cũ. Mỗi bước kiểm checksum bản ghi, độ phủ và in số lượng; sai lệch thì dừng.

### 7.1 Ánh xạ định danh

Một hàm cố định, áp cho mọi dữ liệu mang định danh cũ:

- `brand:ankhang:X` và `leaflet:ankhang:X` → `leaflet:X` (key section, key tài liệu, `section_id`, `chunk_id` có hậu tố `:chunk-NNN`).
- `query_id` `ankhang-N`, `ankhang-alias-N`, `ankhang-alias-any-N` → `leaflet-N`, `leaflet-alias-N`, `leaflet-alias-any-N`.
- `eval_group` `ankhang` → `leaflet`; `source_family` `ankhang_brand` → `leaflet_brand`; `source_subcategory` và tag `ankhang_<x>` → `leaflet_<x>`, tag `ankhang` → `leaflet`; ghi chú: "An Khang" → "tờ hướng dẫn".

Kiểm tra: ánh xạ là đơn ánh trên từng tập (không hai định danh cũ nào về cùng một định danh mới), số bản ghi giữ nguyên, sau ánh xạ không còn chuỗi `ankhang` (không phân biệt hoa thường) trong dữ liệu đích.

### 7.2 Bảng chuyển

| Nguồn | Đích | Việc và kiểm tra |
| --- | --- | --- |
| `heavy/raw/duoc-thu-quoc-gia-viet-nam.pdf` | `sources/duoc-thu-quoc-gia-viet-nam.pdf` | Sao chép; sha256 bằng `source_pdf_sha256` trong manifest rag-final (`2aa81c84…`) |
| `heavy/raw/ankhang/snapshots/ankhang-2026-07-24-97f5b5c43eee.tar.zst` | `sources/leaflets/html/<nhóm>/<slug>.html` | Giải nén 2.434 file; sha256 và kích thước từng file khớp manifest nguồn cũ |
| `manifests/source/ankhang-2026-07-24-97f5b5c43eee.manifest.json` | `sources/leaflets/manifest.json` | `git mv` rồi viết lại theo schema `leaflet-source-v1`: giữ ngày crawl, sha256 danh sách URL, từng file; thêm sitemap URL |
| `resources/ankhang/{all_urls,drug_urls}.txt` | `sources/leaflets/urls/` | `git mv` |
| `resources/{colloquial_mappings,term_glossary,vietnamese_valid_syllables}.json`, `resources/curation/*` | `sources/` | `git mv` (không chứa tên nguồn) |
| `heavy/processed/evaluation/section_retrieval_eval.jsonl`, `patient_queries.json` | `evaluation/gold/` | Kiểm sha256 cũ `b26d9fd7…`, áp §7.1, ghi sha256 mới vào kết quả giai đoạn |
| `heavy/cache/text_embeddings/` và `vector_embeddings/` (bge-m3, gemma-300m, qwen3 0,6B, 8B) | `cache/text_embeddings/<model>.jsonl` | Ghi qua `TextEmbeddingCache`; `verify_record` từng bản ghi nguồn; `text_hash` là sha256 của `embedding_text`; đủ 24.946 cho mỗi model. Khoá theo nội dung nên không chịu ảnh hưởng của §7.1 |
| `heavy/cache/query_embeddings/*` | `cache/query_embeddings/*` | Áp §7.1 cho `query_id`, ghi lại qua `QueryEmbeddingCache` (niêm phong lại); `query_hash` và vector không đổi |
| `heavy/cache/rerank_scores/<model>/<hash>.jsonl` (5 file khớp catalog) | `cache/rerank_scores/<model>.jsonl` | Áp §7.1 cho `query_id` và `chunk_id`, ghi lại qua `RerankScoreCache`; mọi bản ghi khớp `model_sha256` và `request_contract_sha256` của catalog; `query_hash`, `document_hash`, điểm không đổi |
| `heavy/cache/rerank_scores/bge_reranker_v2_gemma_f16/d80693…jsonl` (`native_rerank`) | — | Bỏ; không khớp catalog, không report nào dùng |
| `heavy/runtime_kaggle_profiles/rerank/<model>/<sha>.json` | `cache/kaggle_profiles/rerank/<model>.json` | Sao chép; nạp được bằng `RuntimeProfile.from_dict` |
| `manifests/corpus/*` | — | Bỏ; `corpus/rag-final/manifest.json` được Git giữ |

### 7.3 Build lại và app

1. Bằng CLI mới: `seed build`, `seed validate`, `seed bundle export`, rồi `seed bundle embed --backend local` cho cả 5 model (toàn bộ từ cache, `actions=['cached']`).
2. Parity: script áp §7.1 lên bản sao `heavy/migration/rag-final-chunks.jsonl` vào `work/migration/`, rồi `seed bundle parity --old-chunks` trên bản đã ánh xạ phải đạt 0 lệch.
3. Bundle không còn chuỗi `ankhang`; mọi tài liệu tờ hướng dẫn có `source.url` rỗng.
4. Stack app: `pharma-agent corpus import … --publish` tạo release mới (key mới nên chunk version mới; vector dùng lại từ bundle, không embed); `pharma-agent corpus gc --collection formulary --keep 1`; xoá collection và alias Qdrant `thesis_chunks_qwen3_embedding_4b_fp16*` của pipeline cũ; không bảng nào trong schema `corpus` còn chuỗi `ankhang`.

## 8. Môi trường đánh giá và dựng lại thí nghiệm

### 8.1 Môi trường

```bash
POSTGRES_PORT=5434 QDRANT_HTTP_PORT=6335 QDRANT_GRPC_PORT=6336 \
  docker compose -p thesis-eval up -d postgres qdrant
```

`pharma-agent migrate`, `pharma-agent corpus import` và `seed retrieve` trỏ tới môi trường này qua biến môi trường `PHARMA_POSTGRES__DSN`, `PHARMA_QDRANT__URL`, `PHARMA_RETRIEVAL__EMBEDDING__MODEL`, `PHARMA_RETRIEVAL__EMBEDDING__DIMENSION` (được ưu tiên hơn `backend/.env`). Không cần llama.cpp: vector chunk từ bundle, vector query từ cache, `seed retrieve` tắt rerank. Xong việc: `docker compose -p thesis-eval down -v`.

### 8.2 Ma trận run

| Run | Release | Lệnh |
| --- | --- | --- |
| `dense-bge-m3-k30`, `dense-gemma300m-k30`, `dense-qwen06b-k30`, `dense-qwen4b-k30`, `dense-qwen8b-k30` | Model tương ứng, import và publish lần lượt | `seed retrieve --retriever dense --candidate-k 30`; `seed metrics --top-k 30` |
| `bm25-qwen4b-k30` | 4B | `seed retrieve --retriever bm25 --candidate-k 30`; `seed metrics --top-k 30` |
| `hybrid-qwen4b-p50-k30-rrf2`, `hybrid-qwen4b-p50-k30-rrf60` | 4B | `seed retrieve --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2` (hoặc `60`); `seed metrics --top-k 30` |
| 5 biến thể rerank trên `hybrid-qwen4b-p50-k30-rrf2` | 4B | `seed rerank --model <reranker> --dry-run`, rồi `seed rerank`; `seed metrics --model <reranker> --top-k 30` |
| `dense-text-embedding-3-large-k30` | — | Script một lần chép candidates cũ, áp §7.1 cho `chunk_id` và `section_id`, `origin: imported`, giữ `embedding_model` và `collection_name` cũ; `seed metrics --top-k 30` |

Thứ tự: import và chạy 4 model dense khác trước, model 4B sau cùng, để `bm25`, `hybrid` và rerank dùng release 4B đang được publish.

Rerank: nếu `--dry-run` báo còn cặp query–chunk chưa có trong cache, dừng, báo số cặp và chỉ chạy model (CPU hoặc Kaggle) khi người dùng đồng ý.

### 8.3 Đối chiếu

Ngưỡng theo runbook migration: Hit@10 không giảm quá 1 điểm phần trăm, MRR không giảm quá 0,01; tăng không giới hạn. Run `imported` phải trùng số cũ (sai khác chỉ do làm tròn). Ánh xạ §7.1 là đơn ánh nên nhãn gold không đổi nghĩa.

| Run / biến thể | Hit@10 cũ | MRR cũ | Hit@10 tối thiểu | MRR tối thiểu |
| --- | ---: | ---: | ---: | ---: |
| `bm25-qwen4b-k30` | 77,98% | 0,5550 | 76,98% | 0,5450 |
| `dense-gemma300m-k30` | 83,12% | 0,5879 | 82,12% | 0,5779 |
| `dense-bge-m3-k30` | 90,95% | 0,7333 | 89,95% | 0,7233 |
| `dense-qwen06b-k30` | 91,85% | 0,7042 | 90,85% | 0,6942 |
| `dense-qwen4b-k30` | 95,21% | 0,7784 | 94,21% | 0,7684 |
| `dense-qwen8b-k30` | 95,14% | 0,7948 | 94,14% | 0,7848 |
| `dense-text-embedding-3-large-k30` (imported) | 94,85% | 0,7506 | 94,85% | 0,7506 |
| `hybrid-qwen4b-p50-k30-rrf60` | 92,45% | 0,6784 | 91,45% | 0,6684 |
| `hybrid-qwen4b-p50-k30-rrf2` | 95,67% | 0,7242 | 94,67% | 0,7142 |
| rrf2 + `bge-reranker-v2-gemma:f16` | 95,30% | 0,7570 | 94,30% | 0,7470 |
| rrf2 + `bge-reranker-v2-m3:f16` | 95,04% | 0,7982 | 94,04% | 0,7882 |
| rrf2 + `qwen3-reranker:0.6b-fp16` | 96,18% | 0,7823 | 95,18% | 0,7723 |
| rrf2 + `qwen3-reranker:4b-fp16` | 96,98% | 0,8060 | 95,98% | 0,7960 |
| rrf2 + `qwen3-reranker:8b-fp16` | 89,75% | 0,4797 | 88,75% | 0,4697 |

Run không đạt thì dừng: so candidates cũ (sau ánh xạ §7.1) và mới theo `section_id` từng query, không xoá gì. Bảng đối chiếu cũ/mới đưa vào mô tả commit của giai đoạn 3 để dẫn trong report.

## 9. Archive Kaggle

Dataset private `<owner>/seed-pipeline-data`, `<owner>` là username của profile `--kaggle-account` (mặc định `KAGGLE_ACCOUNT_DEFAULT`).

`seed data push [--kaggle-account ACC] [--message TEXT]`:

1. Liệt kê file bằng `git ls-files --others --ignored --exclude-standard -- data`, bỏ `data/work/`.
2. Ghi luồng tar nén zstd vào `data/work/archive/`, cắt thành `seed-pipeline-data.tar.zst.part-0001`, `…-0002`, mỗi phần tối đa 1,9 GiB.
3. Ghi `archive-manifest.json`: thời điểm, commit Git, danh sách file (`path`, `size`, `sha256`), danh sách phần (`name`, `size`, `sha256`).
4. `kaggle datasets create` lần đầu, `kaggle datasets version` các lần sau, `--dir-mode skip`, không `--public`.

`seed data pull [--kaggle-account ACC] [--force]`: tải các phần và manifest vào `data/work/archive/`, kiểm sha256 từng phần, ghép và giải nén vào `data/`, kiểm sha256 từng file. File đã có mà khác checksum thì dừng, trừ khi có `--force`.

Chia phần vì giới hạn công khai của Kaggle: 200 GB mỗi dataset và 200 GB tổng dung lượng private mỗi tài khoản; diễn đàn Kaggle còn báo giới hạn khoảng 2 GB mỗi file và 1.000 file mỗi lần upload qua API. Lần push đầu cần người dùng xác nhận trước khi chạy.

## 10. Dọn legacy (giai đoạn 4, sau khi §8.3 đạt)

- Dữ liệu: toàn bộ `data/heavy/`, `data/retrieval_eval/`, `data/manifests/corpus/` (`resources/` và manifest nguồn đã được `git mv` ở §7).
- Code chỉ dùng cho migration: lệnh `seed bundle parity`, `bundle/parity.py`, `tests/bundle/test_parity_data.py`, `tests/bundle/test_parity.py`, `MIGRATION_DIR`, runbook `docs/guides/migration-2026-09.md` và các liên kết tới nó.

## 11. Tài liệu

- `data/README.md`: layout, cái gì trong Git, cái gì trong archive, `seed data push/pull`, vai trò của `sources/leaflets/manifest.json`.
- `seed-pipeline/README.md`, `docs/guides/cli-reference.md`, `downstream.md`, `workflow-local-kaggle.md`, `workflow-local-only.md`: đường dẫn mới (bundle ở `data/corpus/formulary`, gold ở `data/evaluation/gold`, run ở `data/evaluation/runs`) và cách gọi trung tính "tờ hướng dẫn sử dụng".
- Guide mới `docs/guides/evaluation.md`: dựng `thesis-eval`, ma trận run §8.2, ngưỡng §8.3.
- `README.md` ở root và `backend/README.md`: đường dẫn `pharma-agent corpus import ../seed-pipeline/data/corpus/formulary`, cách gọi nguồn trung tính.

## 12. Kiểm thử và tiêu chí hoàn thành

- Test mới:
  - không đường dẫn artifact nào khớp `[0-9a-f]{12,}`;
  - xung đột identity báo lỗi, `--force` thay thế;
  - `origin` của run;
  - `seed data push/pull` với runner giả, gồm chia phần và kiểm checksum khi ghép;
  - `seed build` đọc thư mục HTML và từ chối file lệch sha256 trong manifest;
  - bundle export sinh key `leaflet:<nhóm>:<slug>` và `source.url` rỗng;
  - `corpus gc` xoá section và tài liệu mồ côi (integration);
  - test chính sách repo: không file nào trong `backend/src`, `backend/skills`, `seed-pipeline/src`, `frontend/app` chứa `ankhang` hay `an khang` (không phân biệt hoa thường).
- Test hiện có cập nhật theo layout, tên và key mới; test của code bị xoá bị xoá cùng.
- `uv run pytest -q` và `-m integration` của backend, `uv run pytest -q` của seed-pipeline, `ruff`, `pyrefly --min-severity warn`, `pre-commit run --all-files` sạch; không thêm `noqa`, `type: ignore` hay tắt rule.
- Parity 0 lệch trên bundle mới với bản sao `chunks.jsonl` đã ánh xạ (§7.3) trước giai đoạn 4.
- Bảng §8.3 đạt cho 9 run và 5 biến thể rerank.
- Không bảng nào trong schema `corpus` của stack app còn chuỗi `ankhang`.
- `seed data pull` vào một bản clone mới tái tạo đúng mọi file bị bỏ qua (so sha256).
- Trước mỗi commit kiểm file đã stage; không file dữ liệu lớn hay file cục bộ nào vào Git.

## 13. Giai đoạn

1. **Code**: §6 và test §12; parity và `MIGRATION_DIR` còn nguyên.
2. **Chuyển dữ liệu**: §7, gồm build lại, embed từ cache, parity, import lại và dọn stack app. Chạy ngay sau giai đoạn 1 vì CLI mới không đọc layout cũ.
3. **Đánh giá**: §8, dừng ở mọi cổng (cặp rerank thiếu, run không đạt ngưỡng).
4. **Dọn và archive**: §10, §11, rồi `seed data push` sau khi người dùng xác nhận.

Mỗi giai đoạn là một hoặc vài commit riêng.

## 14. Rủi ro

| Rủi ro | Xử lý |
| --- | --- |
| Retrieval qua backend lệch retrieval cũ quá ngưỡng | Dừng ở §8.3, so candidates theo `section_id`, không xoá legacy |
| Retrieval mới trả cặp query–chunk chưa có điểm rerank | Cổng xác nhận trước khi chạy model |
| Ánh xạ định danh gây trùng key | Kiểm đơn ánh ở §7.1 trước khi ghi; trùng thì dừng |
| Đổi key làm đổi chunk version và digest bundle | Import tạo release mới; vector dùng lại từ bundle; `gc` mở rộng dọn danh tính cũ |
| Đổi câu trong system prompt làm đổi hành vi agent | Chỉ thay tên nguồn bằng mô tả nội dung; test prompt hiện có phải qua |
| Kaggle từ chối kích thước phần hoặc số file | Dừng và báo; chỉnh kích thước phần |
| Giữa giai đoạn 1 và 2, CLI không đọc được dữ liệu cũ | Làm liên tiếp; dữ liệu cũ vẫn nguyên cho tới giai đoạn 4 |
