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
- **Orchestrator Kaggle chạy nối tối đa 10 phiên cho một job, nhưng chỉ trên một tài khoản.** `CheckpointInheritanceService` chỉ tìm checkpoint tốt nhất trong các tài khoản một lần lúc bắt đầu. Không có bước nào đọc quota GPU.
- **Sự cố ngày 15/09:**
  - Kernel 0.6b (`doanvanan0209/rerank-5f22fcadeede1072`) dừng ở giây 7.659 vì llama-server ngừng (`recoverable model-server termination`). Worker đóng artifact dở dang 94.950/300.000 cặp, dù ngân sách phiên còn gần 4 giờ.
  - WSL khởi động lại lúc 19:24, làm mất session tmux, tiến trình ở máy và log trong `/tmp` trước khi artifact được tải về.
- **Quota GPU ngày 15/09** (làm mới 2026-09-19): acc1 còn 27,58 giờ, acc2 29,61 giờ, acc3 30,00 giờ.

Quyết định của người dùng:
- Bỏ hẳn `completion_logprobs`; giữ kết quả `bge-reranker-v2-gemma` hiện có làm kết quả cuối cùng.
- Giữ f16 ở mọi nơi.
- Tối ưu cả Kaggle lẫn CPU production bằng benchmark đầu-cuối.
- Gom tài liệu thành khối để tính một lượt.
- Chọn instruction bằng phép đo trên tập con.
- Job Kaggle tự chuyển tài khoản theo quota GPU; khi cả ba tài khoản không đủ quota thì dừng và báo thời điểm làm mới.

## 2. Mục tiêu và ngoài phạm vi

Mục tiêu:

1. Seed-pipeline và backend chỉ còn một protocol reranker là `native_rerank`.
2. Server llama.cpp gom nhiều tài liệu vào một lần tính; cấu hình được chọn bằng benchmark đầu-cuối trên Kaggle T4 và CPU local.
3. Đổi cấu hình runtime không làm mất điểm đã chấm.
4. Instruction được chọn bằng số đo trên tập con, theo quy tắc quyết định chốt trước khi đo.
5. Chấm lại `qwen3-reranker` 0.6b, 4b, 8b trên 300.000 cặp, rồi cập nhật báo cáo.
6. Job Kaggle chạy nối nhiều phiên trên nhiều tài khoản theo quota GPU, không mất điểm khi đổi tài khoản, đổi cấu hình, server rớt hoặc máy khởi động lại.

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
--reranking --kv-unified -np <slots> -c <ub × (slots + 1)> -b <ub> -ub <ub>
```

- Flash attention để mặc định `auto`. Bản 8b giữ `--tensor-split 1,1` trên hai GPU.
- **`-ub` tối thiểu 4096.** Mỗi tài liệu phải nằm trọn trong một ubatch. Đo bằng `/tokenize` trên toàn bộ 300.000 cặp của `hybrid-qwen4b-p50-k30-rrf2`: prompt dài nhất 2.074 token, p99 1.335, p50 692. Đúng một cặp vượt 2.048, và khi `-ub` là 2048 thì cặp đó làm hỏng cả job (xem mục lỗi bên dưới).
- **`-c` bằng `-ub` nhân (số slot + 1).** Mỗi slot được giữ trọn một ubatch, nên `-ub` vừa là giới hạn prompt vừa là đơn vị tính KV; không còn hằng số phụ thuộc dữ liệu.
  - Tính xong một tài liệu, llama-server vẫn giữ KV của prompt đó trong slot.
  - Slot đang chờ chỉ bỏ KV cũ khi tài liệu mới của nó được xếp vào batch. Khi hết chỗ, server chỉ dọn slot rảnh.
  - Nếu `-c` chỉ bằng `-ub`, KV cũ của các slot đang chờ lấp đầy vùng KV và server báo `Context size has been exceeded.`
  - Lỗi này đã gặp trên Kaggle ở mức `-np 64 -c 8192` và tái hiện được trên CPU với `b10920`. Mã `b9637` và `b10920` xử lý giống nhau.
- **Các mức chỉ khác nhau ở số slot**, `-ub` giữ 4096. Thêm slot cho phép nhiều tài liệu vào chung một batch nhưng tốn thêm KV; model càng lớn thì càng ít slot.
- **Prompt dài hơn `-ub` là lỗi dữ liệu, không phải lỗi server.** llama-server trả HTTP 500 `input (N tokens) is too large to process`. Client đánh dấu phản hồi này là không retry được, nên worker dừng hẳn kèm thông báo thay vì restart server ba lần và mất cả phiên GPU.
- Trong `RuntimeCandidate` của reranker:
  - `server_slots` là `-np`.
  - `physical_batch_size = logical_batch_size` là `-ub`.
  - `context_per_slot` là 2048 token mỗi slot giữ.
  - `request_batch_size` là số tài liệu mỗi request.
- **`concurrency` được quét cùng số slot.** Một batch chỉ gom tối đa `-np` tài liệu, nên ít slot thì ubatch và GPU đều bỏ trống; số request đồng thời quyết định slot có luôn đầy hay không. Đo trên Kaggle cho thấy GPU chỉ chạy ~30% ở mức ít slot, nên cả hai chiều đều phải đo thay vì suy ra bằng công thức.

Bộ nhớ đo bằng `llama-server` `b10920` trên CPU. Compute buffer ở `-ub 32768` ra 21.135 MiB, khớp lần cấp phát lỗi trên T4.
- Compute buffer khoảng 0,6 MiB mỗi token `-ub` với cả 0.6b, 4b và 8b. Phần theo `-c` chỉ là mask 2 byte cho mỗi ô `ub × c`.
- KV: 0,109 MiB mỗi token `-c` với 0.6b, 0,141 MiB với 4b và 8b.
- T4 còn trống 14.806 MiB:
  - Ở `-np 64 -ub 16384`, server nạp được nhưng hết bộ nhớ lúc tính.
  - Ở `-ub 32768`, server không cấp phát được compute buffer.

Các mức được quét, ghi dạng (`-np`, `-ub`):

| Nơi chạy | Mức | Tài liệu mỗi request | Tham số khác |
| --- | --- | ---: | --- |
| Kaggle T4, 0.6b (2 server) | (4, 4096, 2) / (8, 4096, 2) / (8, 4096, 4) / (16, 4096, 4) | 30 | |
| Kaggle T4, 4b (2 server) | (2, 4096, 2) / (4, 4096, 2) / (4, 4096, 4) | 30 | |
| Kaggle T4, 8b (1 server, 2 GPU) | (2, 4096, 2) / (4, 4096, 4) / (8, 4096, 4) | 30 | |
| CPU local, 4b | (4, 4096, 1) | 15 | `--threads` 8 / 12 |

### 4.3 Benchmark đầu-cuối

- `BenchmarkLevel` mang một `RuntimeCandidate` đầy đủ. `recommend` trả về chính candidate đó, và `auto_profile` lưu thẳng, không dò ngược theo `(batch_size, concurrency)`.
- **Tải đo:** các nhóm câu hỏi lấy phân tầng theo tổng số ký tự, mỗi request là một nhóm đủ ứng viên.
  - Kaggle: 32 nhóm × 30 tài liệu = 960 cặp mỗi mức.
  - CPU: 6 nhóm × 15 tài liệu mỗi mức.
  - Mỗi mức chạy khởi động một nhóm không tính giờ, và khởi động lại server khi đổi mức.
- **Mỗi mức ghi:** số cặp/giây, p50 và p95 thời gian mỗi request, trạng thái, loại lỗi, và đuôi log server khi lỗi.
- **Kiểm tra thứ hạng:** mức hợp lệ đầu tiên là mốc. Mỗi mức sau xếp lại ứng viên của từng câu hỏi theo điểm; nếu thứ tự khác mốc thì mức đó ghi `invalid` với lỗi `rank_mismatch`, còn chấm thiếu hoặc thừa cặp thì ghi `score_mismatch`. `max_abs_score_delta` vẫn được ghi cho mọi mức để đưa vào báo cáo. Lý do không so điểm tuyệt đối: điểm phụ thuộc cách llama.cpp gom tài liệu vào batch nên hai mức của cùng model lệch 0,02–0,07 trên T4, trong khi pipeline chỉ dùng điểm để xếp hạng.
- **Chọn cấu hình:** Kaggle chọn số cặp/giây cao nhất. CPU chọn p95 thấp nhất; nếu bằng nhau thì chọn số cặp/giây cao hơn. Mọi mức đều lỗi thì stage dừng và in đuôi log.
- **Lưu profile:**
  - Kaggle giữ `data/cache/kaggle_profiles/rerank/<slug>.json`; identity có sha của search space mới nên profile cũ tự hết hiệu lực.
  - Local lưu `data/cache/local_profiles/rerank/<slug>.json`, với `machine_shape` là tên CPU đọc từ `/proc/cpuinfo` và `runtime` là tag image llama.cpp trong compose.

### 4.4 Chấm trên Kaggle

- Worker gửi một request `/v1/rerank` cho mỗi câu hỏi, kèm đủ 30 ứng viên, với số request đồng thời theo mục 4.2. Hai server của 0.6b và 4b dùng chung hàng đợi.
- `JobIdentity` của stage rerank giữ `runtime_profile` trong `payload` (mỗi cấu hình có kernel riêng), nhưng `reuse_payload` bỏ `runtime_parameters.runtime_profile`. Checkpoint và artifact dở dang vì vậy được dùng lại qua các cấu hình.
- Dataset dependency chưa tồn tại được tạo bằng context của `KAGGLE_SHARED_OWNER`, theo mục 4.9, nên lỗi `Dependency owner … does not match dataset service owner …` không còn xảy ra khi chạy bằng tài khoản khác.

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
2. **File tiếng Việt y tế.** Tạo `qwen3-reranker-0.6b-f16-vimed.gguf` từ file gốc bằng `gguf-new-metadata` (gói `gguf` 0.19.0).
   - `--chat-template` nhận danh sách JSON gồm template `default` chép từ file gốc, và template `rerank` giống hệt bản gốc trừ dòng `<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query`.
   - Lệnh đầy đủ và sha256 của file tạo ra ghi trong `docs/guides/evaluation.md`.
   - Catalog thêm model thử nghiệm `qwen3-reranker:0.6b-fp16-vimed`.
   - Hậu tố `-vimed` giữ slug dataset Kaggle của file trong giới hạn 50 ký tự (`-vi-medical` cho bản 0.6b dài 52 ký tự).
   - Số cũ của 0.6b và 4b (MRR 0,7823 và 0,8060) được chấm bằng `completion_logprobs` với instruction tiếng Việt y tế, còn template `rerank` của GGUF dùng instruction mặc định, nên phép đo này cũng cho biết instruction có giải thích được chênh lệch đó hay không.
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
- **Server rớt giữa phiên:** worker khởi động lại llama-server tối đa 3 lần mỗi phiên (chờ `/health`), rồi chấm tiếp các cặp còn thiếu. Hết số lần thử mới đóng artifact dở dang như hiện tại.
- **Lệnh chạy lâu** (benchmark, chấm Kaggle, tải model, `seed data push/pull`) chạy trong tmux, mỗi job một cửa sổ.
- **Log bền:** mỗi lệnh `seed rerank` ghi thêm vào `data/work/logs/rerank/<model-slug>.log`, có timestamp: tài khoản được chọn, quota trước và sau phiên, kernel, tiến độ, lỗi. Log không nằm trong `/tmp`, nên còn sau khi máy khởi động lại. Chạy lại lệnh sẽ nối vào kernel đang chạy nhờ cơ chế reconcile sẵn có.

### 4.9 Nhiều phiên, nhiều tài khoản theo quota

- **Tuỳ chọn `--kaggle-account auto`** cho `seed rerank --backend kaggle`. Giá trị `accN` cụ thể vẫn dùng được như trước.
- **Chọn tài khoản trước mỗi phiên:**
  - Chạy `kaggle quota -v` bằng thông tin xác thực của từng profile `accN` trong `.env`, đọc dòng `GPU` của CSV `resource,used,remaining,total,refreshAt`.
  - Bỏ các tài khoản đang bị job khác khoá, rồi chọn tài khoản có `remaining` lớn nhất; nếu bằng nhau thì chọn số `accN` nhỏ hơn.
  - Ngân sách phiên = min(`--budget-seconds`, `remaining` − 0,5 giờ). Chỉ nộp kernel khi ngân sách ít nhất 1 giờ.
- **Hết quota:** nếu không tài khoản nào đủ 1 giờ, lệnh kết thúc với trạng thái `incomplete` và in bảng quota kèm `refreshAt` của từng tài khoản. Checkpoint đã được đồng bộ, nên chạy lại sau khi quota làm mới là chấm tiếp.
- **Khoá tài khoản:** `data/work/locks/kaggle-accounts/<accN>.lock` (flock) giữ suốt phiên. Nhiều lệnh `seed rerank` cho các model khác nhau chạy song song sẽ tự chia tài khoản, mỗi tài khoản một phiên GPU.
- **Dataset dependency:** dataset chưa tồn tại (model, input) luôn được tạo bằng context của `KAGGLE_SHARED_OWNER`, bất kể tài khoản nào chạy phiên. Người dùng không phải chạy lại bằng `acc1` như lỗi của bản 4b.
- **Checkpoint theo phiên:**
  - Sau mỗi phiên, artifact (kể cả dở dang) được gộp vào cache điểm ở máy trước tiên, rồi mới publish checkpoint.
  - Trước mỗi phiên, nguồn có nhiều cặp nhất trong {cache điểm ở máy, checkpoint của mọi tài khoản} được publish làm checkpoint của tài khoản sắp chạy. `CheckpointInheritanceService` được mở rộng thêm nguồn cache ở máy và chạy lại ở mỗi phiên, không chỉ lúc bắt đầu.
  - Cache ở máy là nguồn gốc, nên đổi tài khoản, đổi cấu hình hay đổi identity job đều không mất điểm.
- **Số phiên:** vòng lặp chạy tới khi xong hoặc hết quota, không còn giới hạn cứng 10 phiên. `--max-runs` vẫn giữ để đặt giới hạn khi cần.

## 5. Thứ tự triển khai

1. Làm mục 4.1–4.6, 4.8 và 4.9; toàn bộ test, ruff, pyrefly của hai project pass.
2. Tải output kernel `doanvanan0209/rerank-5f22fcadeede1072` (94.950 cặp của 0.6b, template gốc) và gộp vào cache điểm ở máy.
3. Benchmark Kaggle cho `qwen3-reranker:0.6b-fp16` với search space mới.
4. Đo instruction theo mục 4.7 và chốt template. Nếu instruction tiếng Việt thắng, 94.950 cặp ở bước 2 bị xoá cùng các dữ liệu dùng template gốc.
5. Chấm toàn bộ 300.000 cặp của ba model bằng `--kaggle-account auto`, mỗi model một cửa sổ trong tmux, chạy song song.
6. Benchmark CPU local cho `qwen3-reranker:4b-fp16`, rồi ghi mặc định vào `compose.yaml` và `.env.example`.
7. Với mỗi model, chạy `seed metrics` và kiểm tra phân bố điểm: điểm phải trải từ gần 0 tới gần 1, trung vị chênh lệch điểm trong một câu hỏi phải lớn hơn 0,9. Sau đó cập nhật bảng 5.1–5.3 và nhận xét trong `report/report.md`, cùng bảng trong `docs/guides/evaluation.md`. Bảng tốc độ 5.3 ghi số native đã tối ưu cho Kaggle và p95 CPU.

## 6. Kiểm thử

Unit test seed-pipeline:
- **Catalog:** mọi reranker là `native_rerank`; không còn `bge-reranker-v2-gemma:f16`; model thử nghiệm (nếu còn) trỏ đúng file và sha256.
- **Lệnh server:** reranker có đủ `--reranking --kv-unified -np -c -b -ub`, với `-b` bằng `-ub` và `-c` bằng `-ub` nhân (số slot + 1).
- **`inference_cache_policy`:** không còn trường dành cho completion.
- **Benchmark:** mức mang đủ candidate; chọn theo số cặp/giây (Kaggle) và p95 (local); bỏ qua mức `invalid`; phát hiện `rank_mismatch` và `score_mismatch`; báo lỗi khi mọi mức lỗi.
- **Worker native:** request theo nhóm câu hỏi; số request đồng thời theo công thức mục 4.2; đóng artifact dở dang khi server rớt.
- **`JobIdentity`:** đổi `runtime_profile` giữ nguyên `reuse_sha256` nhưng đổi `sha256`.
- **Local:** chấm theo nhóm; lệnh benchmark truyền đúng biến môi trường cho từng mức, kiểm bằng runner giả.
- **`seed retrieve --sample`:** số câu mỗi `eval_group` đúng tỉ lệ; cùng seed ra cùng mẫu; mẫu ghi vào identity.
- **`seed metrics`:** chạy được trên run không còn variant Gemma; report Gemma không bị ghi đè.
- **Chọn tài khoản** (runner giả trả CSV quota):
  - Chọn tài khoản còn nhiều giờ nhất; bằng nhau thì chọn số nhỏ hơn.
  - Bỏ tài khoản đang bị khoá.
  - Ngân sách phiên trừ 0,5 giờ dự phòng; không nộp kernel khi ngân sách dưới 1 giờ.
  - Hết quota thì kết thúc `incomplete` và in `refreshAt`.
- **Khoá tài khoản:** hai job không lấy cùng một tài khoản cùng lúc.
- **Checkpoint qua tài khoản:**
  - Artifact dở dang được gộp vào cache ở máy.
  - Phiên kế tiếp trên tài khoản khác nhận checkpoint từ nguồn có nhiều cặp nhất; số cặp đã có không bao giờ giảm.
- **Dataset dependency:** dataset thiếu được tạo bằng context của `KAGGLE_SHARED_OWNER` khi tài khoản chạy phiên là tài khoản khác.
- **Worker:** server rớt được khởi động lại tối đa 3 lần rồi chấm tiếp; quá số lần thì đóng artifact dở dang.
- **Log:** ghi vào `data/work/logs/rerank/<model-slug>.log`, có tài khoản và quota của từng phiên.

Unit test backend: `RerankSettings` từ chối `completion_logprobs`; composition dựng `NativeReranker`; test tracing chỉ còn nhánh native.

Test tích hợp (marker `integration`, bỏ qua khi thiếu `ai-models/gguf/qwen3-reranker-0.6b-f16.gguf` hoặc Docker):
- Chạy image `ghcr.io/ggml-org/llama.cpp:server-b10920` với bộ cờ mục 4.2.
- Gửi `/v1/rerank` gồm một tài liệu đúng, một tài liệu lạc đề, và một tài liệu khoảng 1.600 token.
- Điểm nằm trong [0, 1], tài liệu đúng có điểm cao hơn tài liệu lạc đề, không có lỗi "input is too large".

Lệnh bắt buộc: `pytest` (warning coi là lỗi), `ruff check`, `ruff format --check`, `pyrefly check` ở cả hai project; pre-commit chạy từ thư mục gốc repo.

## 7. Tài liệu

- `seed-pipeline/docs/guides/evaluation.md`: chỉ còn rerank native; benchmark Kaggle và local; run tập con; phép đo instruction (lệnh `gguf-new-metadata`, sha256, quy tắc quyết định); bảng kết quả mới.
- `seed-pipeline/docs/guides/cli-reference.md`: `seed rerank --backend local --benchmark`, `seed retrieve --sample/--sample-seed`.
- `seed-pipeline/docs/guides/workflow-local-kaggle.md`, `workflow-local-only.md`: lệnh rerank mới, `--kaggle-account auto` và cách đọc bảng quota khi dừng, chạy trong tmux, vị trí log.
- `seed-pipeline/data/README.md`: thêm `cache/local_profiles/`.
- `backend/docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md`: bảng settings (`native_rerank | none`), bỏ mô tả `completion_logprobs`, cập nhật mục 5 của nhật ký quyết định.
- `README.md` và `.env.example` ở gốc repo: biến llama.cpp của reranker và cách chạy lại benchmark CPU.

## 8. Rủi ro

- **Bộ nhớ T4 với `-ub` lớn** (4b, 8b): benchmark gặp mức thiếu bộ nhớ thì ghi `invalid` và chọn mức nhỏ hơn.
- **CPU tăng tốc ít:** trên CPU tổng khối lượng tính gần như cố định. Báo cáo ghi p95 thực đo, không suy ra từ số GPU.
- **Phép đo instruction có nhiễu** trên 1.000 câu: quy tắc bootstrap chốt trước; nếu không rõ thì giữ template gốc.
- **Quota bản 8b:** có thể cần nhiều tuần. Checkpoint dùng lại được qua phiên và qua tài khoản, nhờ `reuse_sha256` không phụ thuộc cấu hình và nhờ dataset do `acc1` sở hữu.
- **Hai phiên bản llama.cpp:** số tốc độ Kaggle (`b9637`) và CPU (`b10920`) được báo cáo riêng, không so với nhau.
- **Chênh lệch số thực khi gom khối:** không chặn được, chỉ kiểm soát bằng kiểm tra thứ hạng trong benchmark (mục 4.3) và bằng việc mỗi run chấm trọn ở một mức duy nhất.
- **Số quota Kaggle cập nhật trễ:** quota được đọc lại trước mỗi phiên và có 0,5 giờ dự phòng. Nếu Kaggle vẫn cắt phiên vì hết quota, artifact dở dang được gộp như khi hết ngân sách.
- **Giới hạn phiên GPU đồng thời của mỗi tài khoản:** khoá tài khoản bảo đảm mỗi tài khoản chỉ chạy một phiên từ pipeline.
