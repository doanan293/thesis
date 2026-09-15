# Rerank native tối ưu, bỏ completion_logprobs

Ngày: 2026-09-15. Phạm vi: reranker của seed-pipeline (catalog, contract, client, worker và benchmark Kaggle, local backend, `seed retrieve`, `seed metrics`), reranker của backend (`llama_cpp_reranker`, settings), `compose.yaml`, `.env.example`, tài liệu liên quan.

## 1. Bối cảnh

- Ngày 15/09 ba reranker Qwen3 được chuyển sang `native_rerank` (`POST /v1/rerank`):
  - Bản 8b chấm bằng `completion_logprobs` cho kết quả sai (MRR 0,4797). File GGUF 8b là bản convert classifier (có `cls.output.weight`, không có `output.weight`). Qwen3-Reranker-8B không dùng chung ma trận embedding cho lớp đầu ra, nên llama.cpp lấy nhầm `token_embd.weight` làm lớp đầu ra.
  - File 4b của mradermacher không có classifier head nên không phục vụ được `/v1/rerank`. File đã được thay bằng bản f16 của Voodisss.
- Qwen3-Reranker là LLM. Theo model card, điểm là softmax trên logit "yes" và "no" ở token cuối. `/v1/rerank` tính đúng phép này qua `cls.output.weight`. Model card cũng ghi instruction theo đúng task thường tăng 1–5%.
- Đo trên Kaggle T4 với cùng file GGUF, native chậm hơn logprobs: 0.6b 15,2 so với 18,5 cặp/giây, 8b 1,3 so với 1,7.
- Nguyên nhân nằm ở cấu hình và phần mềm:
  - **llama.cpp không dùng lại cache prompt với rank pooling.** `server_slot::can_split()` trả `false`, nên server bỏ qua nhánh cache prompt và tính lại toàn bộ prompt cho mỗi tài liệu. Server vẫn xếp được tài liệu của nhiều slot vào cùng một ubatch.
  - **Search space chỉ quét số request đồng thời phía client.** Server đặt `-np` 4 (0.6b), 2 (4b, 8b), `-ub` 2048, `-c = 4096 × np`.
  - **Benchmark gửi một tài liệu mỗi request.** Job chấm thật gửi 30 tài liệu mỗi request, nhưng mỗi lúc chỉ `-np` tài liệu được tính.
- `compose.yaml` đặt `LLAMA_ARG_UBATCH` 512, thấp hơn độ dài prompt trung vị (726 token). Với `/v1/rerank`, phần lớn tài liệu sẽ bị từ chối "input is too large".
- `seed rerank --backend local` chấm từng tài liệu một, và local chưa có benchmark.
- `reuse_sha256` của job rerank chứa `runtime_profile`, nên đổi cấu hình runtime là mất checkpoint dù điểm không phụ thuộc cấu hình.

Quyết định của người dùng:
- Bỏ hẳn `completion_logprobs`; giữ kết quả `bge-reranker-v2-gemma` hiện có làm kết quả cuối cùng.
- Giữ f16 ở mọi nơi.
- Tối ưu cả Kaggle lẫn CPU production bằng benchmark đầu-cuối.
- Gom tài liệu thành khối để tính một lượt.
- Chọn instruction bằng phép đo trên tập con.

## 2. Mục tiêu và ngoài phạm vi

Mục tiêu:

1. Seed-pipeline và backend chỉ còn một protocol reranker là `native_rerank`.
2. Server llama.cpp gom nhiều tài liệu vào một lần tính; cấu hình được chọn bằng benchmark đầu-cuối trên Kaggle T4 và CPU local.
3. Đổi cấu hình runtime không làm mất điểm đã chấm.
4. Instruction được chọn bằng số đo trên tập con, theo quy tắc quyết định chốt trước khi đo.
5. Chấm lại `qwen3-reranker` 0.6b, 4b, 8b trên 300.000 cặp, rồi cập nhật báo cáo.

Ngoài phạm vi: lượng tử hoá; đổi engine suy luận (vLLM, TEI); thêm hoặc đổi model; chấm lại Gemma; nâng runtime Kaggle từ `b9637` lên `b10920` (thay vào đó dùng bộ cờ có ở cả hai bản); spec và plan cũ trong `docs/superpowers/`.

## 3. Số liệu đầu vào (kiểm ngày 2026-09-15)

Độ dài prompt rerank, đo bằng tokenizer `Qwen/Qwen3-Reranker-0.6B` với template `rerank` gốc, trên 12.000 cặp của 400 câu hỏi ngẫu nhiên thuộc run `hybrid-qwen4b-p50-k30-rrf2`:

| Chỉ số | Token |
| --- | ---: |
| Trung bình | 663 |
| p50 / p90 / p99 | 726 / 1.117 / 1.328 |
| Lớn nhất | 1.664 |
| Phần đầu chung (system, instruction, query) | 88 |
| Một câu hỏi, 30 tài liệu | khoảng 19.900 |

File model, tất cả đều có `cls.output.weight`, `pooling_type` rank và template `rerank` gốc:

| Model catalog | File | Nguồn | sha256 |
| --- | --- | --- | --- |
| `qwen3-reranker:0.6b-fp16` | `qwen3-reranker-0.6b-f16.gguf` | `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp` | `fa726a72…` |
| `qwen3-reranker:4b-fp16` | `qwen3-reranker-4b-f16.gguf` | `Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp` | `c4de2e3e…` |
| `qwen3-reranker:8b-fp16` | `qwen3-reranker-8b-f16.gguf` | `sinjab/Qwen3-Reranker-8B-F16-GGUF` | `a53322f7…` |

Hành vi llama.cpp đã xác minh trong mã nguồn:
- Kaggle chạy `b9637`; compose chạy `b10920`.
- Khi bật `--kv-unified`, `n_ctx_seq = n_ctx`: mỗi chuỗi dùng chung cả vùng KV.
- Tối đa 256 chuỗi song song (`LLAMA_MAX_SEQ`).
- `--kv-unified-per-slot` chỉ có ở `b10920`.

Tài khoản Kaggle:
- `acc1` là `KAGGLE_SHARED_OWNER`, là tài khoản duy nhất tạo được dataset dependency, trong đó có dataset model.
- `acc2` và `acc3` chỉ dùng được dataset đã tồn tại.

Máy production: Intel i5-13420H (12 luồng, AVX2, AVX-VNNI), 23 GB RAM, không có GPU.

## 4. Thiết kế

### 4.1 Bỏ `completion_logprobs`

Seed-pipeline xoá:
- `CompletionScoring`, `qwen3_rerank_contract`, `bge_gemma_rerank_contract`, `build_qwen3_yes_no_prompt`, `build_bge_gemma_yes_no_prompt`, `build_qwen_rerank_prompt`.
- Trong `LlamaCppClient`: `completion_payload`, `rerank_completion*`, tra token id, parse logprobs và timing.
- Nhánh completion trong `workers/rerank.py`, `workers/benchmark.py` và local reranker.
- Nhánh completion trong `server_policy.inference_cache_policy` (cache prompt, slot similarity) và `compare_cache_arms` cùng cổng `cache_comparison` trong `auto_profile`.
- Tham số `protocol` của `_reranker` trong catalog. `RerankContract` chỉ còn định danh protocol `native_rerank`; template `rerank` nằm trong file GGUF và đã được sha256 của model bao phủ.

Backend xoá:
- `LlamaCppCompletionReranker`.
- Giá trị `completion_logprobs` trong `RerankSettings.protocol`, chỉ còn `native_rerank | none`.
- Nhánh completion trong test Langfuse tracing và test reranker, cùng các dòng tương ứng trong spec backend và `.env.example`.

Kết quả Gemma:
- Giữ nguyên `data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/bge_reranker_v2_gemma_f16/` đã commit; báo cáo dùng số của nó như mọi reranker khác.
- Bỏ `bge-reranker-v2-gemma:f16` khỏi catalog và khỏi `rerank_variants` trong `run.json`.
- Xoá `rerank/bge_reranker_v2_gemma_f16/` và `data/cache/rerank_scores/bge_reranker_v2_gemma_f16.jsonl` (bản sao còn trong dataset Kaggle `seed-pipeline-data`), cùng `data/cache/kaggle_profiles/rerank/bge_reranker_v2_gemma_f16.json`.

### 4.2 Server gom khối

Mọi server reranker, trên Kaggle lẫn trong compose, chạy với:

```text
--reranking --kv-unified -np <slots> -c <ub> -b <ub> -ub <ub>
```

- Flash attention để mặc định `auto`. Bản 8b giữ `--tensor-split 1,1` trên hai GPU.
- **`-c` bằng `-ub`, không nhân theo số slot.** Mỗi task rerank trả KV ngay sau một lượt tính, nên vùng KV chỉ cần chứa số token đang tính cùng lúc.
- **`-ub` tối thiểu 2048**, vì prompt dài nhất là 1.664 token và mỗi tài liệu phải nằm trọn trong một ubatch.
- Trong `RuntimeCandidate` của reranker: `server_slots` là `-np`; `physical_batch_size = logical_batch_size = context_per_slot` là `-ub`; `request_batch_size` là số tài liệu mỗi request.
- **`concurrency` không quét** mà tính bằng `ceil(server_slots / request_batch_size) + 1` cho mỗi server, để slot luôn đầy.

Giá trị cố định và giá trị được quét:

| Nơi chạy | `-np` | `-ub` được quét | Tài liệu mỗi request | Tham số khác |
| --- | ---: | --- | ---: | --- |
| Kaggle T4, 0.6b (2 server) | 64 | 8192 / 16384 / 32768 | 30 | |
| Kaggle T4, 4b (2 server) | 32 | 8192 / 16384 | 30 | |
| Kaggle T4, 8b (1 server, 2 GPU) | 16 | 4096 / 8192 | 30 | |
| CPU local, 4b | 16 | 4096 / 8192 / 16384 | 15 | `--threads` 8 / 12 |

### 4.3 Benchmark đầu-cuối

- `BenchmarkLevel` mang một `RuntimeCandidate` đầy đủ. `recommend` trả về chính candidate đó, và `auto_profile` lưu thẳng, không dò ngược theo `(batch_size, concurrency)`.
- **Tải đo:** các nhóm câu hỏi lấy phân tầng theo tổng số ký tự, mỗi request là một nhóm đủ ứng viên.
  - Kaggle: 32 nhóm × 30 tài liệu = 960 cặp mỗi mức.
  - CPU: 6 nhóm × 15 tài liệu mỗi mức.
  - Mỗi mức chạy khởi động một nhóm không tính giờ, và khởi động lại server khi đổi mức.
- **Mỗi mức ghi:** số cặp/giây, p50 và p95 thời gian mỗi request, trạng thái, loại lỗi, và đuôi log server khi lỗi.
- **Kiểm tra điểm:** mức hợp lệ đầu tiên là mốc. Mỗi mức sau so điểm của cùng mẫu với mốc; nếu `max_abs_score_delta` lớn hơn `1e-3` thì mức đó ghi `invalid` với lỗi `score_mismatch`. Việc này bảo đảm đổi cấu hình không đổi điểm, là tiền đề của mục 4.4.
- **Chọn cấu hình:** Kaggle chọn số cặp/giây cao nhất. CPU chọn p95 thấp nhất; nếu bằng nhau thì chọn số cặp/giây cao hơn. Mọi mức đều lỗi thì stage dừng và in đuôi log.
- **Lưu profile:**
  - Kaggle giữ `data/cache/kaggle_profiles/rerank/<slug>.json`; identity có sha của search space mới nên profile cũ tự hết hiệu lực.
  - Local lưu `data/cache/local_profiles/rerank/<slug>.json`, với `machine_shape` là tên CPU đọc từ `/proc/cpuinfo` và `runtime` là tag image llama.cpp trong compose.

### 4.4 Chấm trên Kaggle

- Worker gửi một request `/v1/rerank` cho mỗi câu hỏi, kèm đủ 30 ứng viên, với số request đồng thời theo mục 4.2. Hai server của 0.6b và 4b dùng chung hàng đợi.
- `JobIdentity` của stage rerank giữ `runtime_profile` trong `payload` (mỗi cấu hình có kernel riêng), nhưng `reuse_payload` bỏ `runtime_parameters.runtime_profile`. Checkpoint và artifact dở dang vì vậy được dùng lại qua các cấu hình.
- Lỗi `Dependency owner … does not match dataset service owner …` được bổ sung hướng dẫn: chạy lần đầu bằng tài khoản `KAGGLE_SHARED_OWNER` để tạo dataset.

### 4.5 Local CPU

- `seed rerank --backend local` gửi một request cho mỗi câu hỏi qua service `llama-reranker` của compose. Nếu có profile local thì dùng nó, không thì dùng mặc định của compose.
- Lệnh mới `seed rerank --backend local --benchmark --model <model> --run <run>`:
  - Với mỗi mức, dựng lại `llama-reranker` bằng `LLAMA_ARG_N_PARALLEL`, `LLAMA_ARG_UBATCH`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_CTX_SIZE`, `LLAMA_ARG_KV_UNIFIED`, `LLAMA_ARG_THREADS`.
  - Đo theo mục 4.3, rồi in cấu hình được chọn.
- `compose.yaml` bỏ `LLAMA_ARG_KV_UNIFIED_PER_SLOT` và đặt mặc định bằng kết quả đo trên i5-13420H; chú thích ghi rõ máy đo. `.env.example` liệt kê cùng các biến để máy khác ghi đè sau khi tự chạy benchmark.

### 4.6 Backend

- `NativeReranker` giữ nguyên: mỗi vòng search gửi một request chứa toàn bộ ứng viên.
- Compose đặt `PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT` = 1. `RERANK_MAX_CANDIDATES` giữ 15. `PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS` tối thiểu gấp 2 lần p95 CPU đo được.

### 4.7 Đo instruction trên tập con

1. **Tập con.** `seed retrieve --sample 1000 --sample-seed 0` lấy phân tầng 10% mỗi `eval_group`: 500 `formulary`, 250 `leaflet`, 100 `chunk_risk`, 50 `patient_natural`, 50 `noisy_confuser`, 50 `multi_intent`. Mẫu được ghi vào identity của run. Run tạo ra là `hybrid-qwen4b-p50-k30-rrf2-sample1000`. `--limit` giữ nguyên nghĩa là N dòng đầu.
2. **File tiếng Việt y tế.** Tạo `qwen3-reranker-0.6b-f16-vi-medical.gguf` từ file gốc bằng `gguf-new-metadata` (gói `gguf` 0.19.0).
   - `--chat-template` nhận danh sách JSON gồm template `default` chép từ file gốc, và template `rerank` giống hệt bản gốc trừ dòng `<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query`.
   - Lệnh đầy đủ và sha256 của file tạo ra ghi trong `docs/guides/evaluation.md`.
   - Catalog thêm model thử nghiệm `qwen3-reranker:0.6b-fp16-vi-medical`.
3. **Chấm.** Chấm cả hai model 0.6b trên 30.000 cặp của tập con với profile Kaggle mới, rồi chạy `seed metrics`.
4. **Quy tắc quyết định.** Bootstrap theo cặp trên 1.000 câu hỏi (10.000 lần lấy mẫu lại, seed 0) cho hiệu MRR@30 giữa bản tiếng Việt và bản gốc.
   - Cận dưới khoảng tin cậy 95% lớn hơn 0: dùng instruction tiếng Việt.
   - Ngược lại: giữ template gốc.
5. **Sau quyết định.**
   - Nếu tiếng Việt thắng: tạo file tương ứng cho 4b và 8b. Ba model `qwen3-reranker:{0.6b,4b,8b}-fp16` trỏ tới file mới; xoá file gốc, cache điểm và checkpoint dùng template gốc; tên file mặc định trong compose đổi theo.
   - Nếu template gốc thắng: xoá model thử nghiệm và file của nó.
   - Report của run tập con được giữ làm bằng chứng cho báo cáo.

### 4.8 Lỗi và phục hồi

- **Một mức benchmark hết bộ nhớ, server không lên, hoặc điểm lệch mốc:** mức đó ghi `invalid`, chạy tiếp mức sau.
- **Job chấm hết ngân sách 6 giờ:** artifact dở dang được gộp vào cache; chạy lại lệnh là tiếp tục từ checkpoint nhờ `reuse_sha256` mới.
- **Server rớt giữa chừng:** giữ hành vi hiện tại, đóng artifact dở dang để có thể phục hồi.
- **Lệnh chạy lâu** (benchmark, chấm Kaggle, tải model, `seed data push/pull`) chạy trong tmux, mỗi job một cửa sổ và có file log.

## 5. Thứ tự triển khai

1. Làm mục 4.1–4.6; toàn bộ test, ruff, pyrefly của hai project pass.
2. Benchmark Kaggle cho `qwen3-reranker:0.6b-fp16` với search space mới.
3. Đo instruction theo mục 4.7 và chốt template.
4. Chấm toàn bộ 300.000 cặp: 0.6b trên `acc1`; 4b trên `acc3` (dataset model đã được `acc1` tạo); 8b trên `acc2`, chạy nối nhiều phiên và chia sang tài khoản khác nếu vượt quota tuần.
5. Benchmark CPU local cho `qwen3-reranker:4b-fp16`, rồi ghi mặc định vào `compose.yaml` và `.env.example`.
6. Với mỗi model, chạy `seed metrics` và kiểm tra phân bố điểm: điểm phải trải từ gần 0 tới gần 1, trung vị chênh lệch điểm trong một câu hỏi phải lớn hơn 0,9. Sau đó cập nhật bảng 5.1–5.3 và nhận xét trong `report/report.md`, cùng bảng trong `docs/guides/evaluation.md`. Bảng tốc độ 5.3 ghi số native đã tối ưu cho Kaggle và p95 CPU.

## 6. Kiểm thử

Unit test seed-pipeline:
- **Catalog:** mọi reranker là `native_rerank`; không còn `bge-reranker-v2-gemma:f16`; model thử nghiệm (nếu còn) trỏ đúng file và sha256.
- **Lệnh server:** reranker có đủ `--reranking --kv-unified -np -c -b -ub`, với `-c` bằng `-ub`.
- **`inference_cache_policy`:** không còn trường dành cho completion.
- **Benchmark:** mức mang đủ candidate; chọn theo số cặp/giây (Kaggle) và p95 (local); bỏ qua mức `invalid`; phát hiện `score_mismatch`; báo lỗi khi mọi mức lỗi.
- **Worker native:** request theo nhóm câu hỏi; số request đồng thời theo công thức mục 4.2; đóng artifact dở dang khi server rớt.
- **`JobIdentity`:** đổi `runtime_profile` giữ nguyên `reuse_sha256` nhưng đổi `sha256`.
- **Local:** chấm theo nhóm; lệnh benchmark truyền đúng biến môi trường cho từng mức, kiểm bằng runner giả.
- **`seed retrieve --sample`:** số câu mỗi `eval_group` đúng tỉ lệ; cùng seed ra cùng mẫu; mẫu ghi vào identity.
- **`seed metrics`:** chạy được trên run không còn variant Gemma; report Gemma không bị ghi đè.

Unit test backend: `RerankSettings` từ chối `completion_logprobs`; composition dựng `NativeReranker`; test tracing chỉ còn nhánh native.

Test tích hợp (marker `integration`, bỏ qua khi thiếu `ai-models/gguf/qwen3-reranker-0.6b-f16.gguf` hoặc Docker):
- Chạy image `ghcr.io/ggml-org/llama.cpp:server-b10920` với bộ cờ mục 4.2.
- Gửi `/v1/rerank` gồm một tài liệu đúng, một tài liệu lạc đề, và một tài liệu khoảng 1.600 token.
- Điểm nằm trong [0, 1], tài liệu đúng có điểm cao hơn tài liệu lạc đề, không có lỗi "input is too large".

Lệnh bắt buộc: `pytest` (warning coi là lỗi), `ruff check`, `ruff format --check`, `pyrefly check` ở cả hai project; pre-commit chạy từ thư mục gốc repo.

## 7. Tài liệu

- `seed-pipeline/docs/guides/evaluation.md`: chỉ còn rerank native; benchmark Kaggle và local; run tập con; phép đo instruction (lệnh `gguf-new-metadata`, sha256, quy tắc quyết định); bảng kết quả mới.
- `seed-pipeline/docs/guides/cli-reference.md`: `seed rerank --backend local --benchmark`, `seed retrieve --sample/--sample-seed`.
- `seed-pipeline/docs/guides/workflow-local-kaggle.md`, `workflow-local-only.md`: lệnh rerank mới, tài khoản tạo dataset model, chạy trong tmux.
- `seed-pipeline/data/README.md`: thêm `cache/local_profiles/`.
- `backend/docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md`: bảng settings (`native_rerank | none`), bỏ mô tả `completion_logprobs`, cập nhật mục 5 của nhật ký quyết định.
- `README.md` và `.env.example` ở gốc repo: biến llama.cpp của reranker và cách chạy lại benchmark CPU.

## 8. Rủi ro

- **Bộ nhớ T4 với `-ub` lớn** (4b, 8b): benchmark gặp mức thiếu bộ nhớ thì ghi `invalid` và chọn mức nhỏ hơn.
- **CPU tăng tốc ít:** trên CPU tổng khối lượng tính gần như cố định. Báo cáo ghi p95 thực đo, không suy ra từ số GPU.
- **Phép đo instruction có nhiễu** trên 1.000 câu: quy tắc bootstrap chốt trước; nếu không rõ thì giữ template gốc.
- **Quota bản 8b:** có thể cần nhiều tuần. Checkpoint dùng lại được qua phiên và qua tài khoản, nhờ `reuse_sha256` không phụ thuộc cấu hình và nhờ dataset do `acc1` sở hữu.
- **Hai phiên bản llama.cpp:** số tốc độ Kaggle (`b9637`) và CPU (`b10920`) được báo cáo riêng, không so với nhau.
- **Chênh lệch số thực khi gom khối:** giới hạn bằng kiểm tra `score_mismatch` trong benchmark (mục 4.3).
