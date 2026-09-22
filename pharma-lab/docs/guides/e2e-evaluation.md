# Đánh giá end-to-end

Đánh giá toàn bộ agent (guard → rephrase → search → judge/refine → answer) trên bộ golden 500 câu. Thiết kế nằm trong `docs/superpowers/specs/2026-09-16-e2e-golden-evaluation-design.md` ở thư mục gốc repo; cách soạn bộ golden nằm trong [e2e-golden-authoring.md](e2e-golden-authoring.md).

## 1. Chuẩn bị

Harness gọi backend trong cùng tiến trình, như `pharma-lab retrieve`, với settings đọc từ `../backend/.env` (`--backend-env-file`).

**Cần sẵn sàng:**
- Postgres và Qdrant đã import và publish release. Alias `chunks_current` trỏ vào collection của `qwen3-embedding:4b-fp16`.
- Endpoint LLM trong `PHARMA_LLM__*`. Model của từng role lấy từ `.env` và được ghi vào `run.json`.
- llama.cpp embedding (`PHARMA_RETRIEVAL__EMBEDDING__*`) và reranker (`PHARMA_RETRIEVAL__RERANK__*`), chạy qua `docker compose up -d` ở thư mục gốc repo.

**Endpoint LLM** là service `cli-proxy-api` của compose (`docker compose up -d cli-proxy-api`, đăng nhập lưu trong volume `thesis_cli_proxy_auth`).

**Reranker trên GPU.** Reranker trên CPU quá chậm cho E2E: trên laptop i5, 40 tài liệu mất khoảng 10 phút với model 4B. Vì vậy các cấu hình có rerank dùng reranker chạy trên GPU Kaggle, qua tunnel có API key.

```bash
uv run pharma-lab e2e rerank-server --model qwen3-reranker:4b-fp16 --hours 8 --kaggle-account auto
```

Lệnh này:
- giữ một account và đẩy kernel `rerank-serve` lên (llama-server trên 2 T4 cộng tunnel cloudflared);
- ghi `data/work/serve/<model>.env` khi tunnel đã sẵn sàng, và xóa file đó khi phiên kết thúc.

Nếu máy tắt khi kernel vẫn chạy, nối lại mà không mất phiên GPU. Lệnh này lấy key từ source của kernel, lấy URL từ log, rồi ghi lại file env:

```bash
uv run pharma-lab e2e rerank-server --kaggle-account acc1 --attach OWNER/rerank-serve-XXXX
```

URL được đọc bằng `kaggle kernels logs -f`. Khi API log của Kaggle trục trặc, tiến trình này có thể treo mà không in gì; watcher mở lại nó nếu log im lặng quá 10 phút (kernel in dòng `alive` mỗi 5 phút). Muốn có URL ngay thì dùng lệnh `--attach` ở trên.

Trong session chạy E2E, nạp file env trước (`set -a; . data/work/serve/qwen3_reranker_4b_fp16.env; set +a`).

**Endpoint LLM còn phục vụ judge**, mặc định `gpt-5-mini` với reasoning `medium`. Proxy antigravity không có model này. Run `e2e-v1` chấm bằng `--judge-model gemini-3.1-flash-lite`: model này khác model trả lời (`gemini-3.8-flash-high`), nhanh (khoảng 3 giây mỗi lần gọi), và có quota lớn nhất trong các model đã thử.

**Chạy với model open-weight.** Run `e2e-qwen35-9b` dùng Qwen3.5-9B (9,65 tỉ tham số, Apache 2.0) cho mọi bước của pipeline; judge vẫn dùng endpoint mặc định trong `backend/.env`.

*Chuẩn bị file GGUF F16* (một lần, ở thư mục gốc repo). T4 không có kernel bfloat16, nên bản BF16 được chuyển sang F16 bằng `llama-quantize` của cùng bản build llama.cpp với compose:

```bash
mkdir -p ai-models/downloads
curl -fL -C - -o ai-models/downloads/Qwen3.5-9B-BF16.gguf \
  https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-BF16.gguf
echo "daebe40eeea7057c1cdf35ac56d13f507d8bf12171bbb7a6b6b0d3f05439159a  ai-models/downloads/Qwen3.5-9B-BF16.gguf" | sha256sum -c -
docker run --rm -v "$PWD/ai-models:/models" --entrypoint /app/llama-quantize \
  ghcr.io/ggml-org/llama.cpp:full-b10920 \
  /models/downloads/Qwen3.5-9B-BF16.gguf /models/gguf/qwen3.5-9b-f16.gguf F16
echo "863a67e28486f4c3ad7a30c49614a398d5a07caf8d0067a018d0e5f0abf79f2d  ai-models/gguf/qwen3.5-9b-f16.gguf" | sha256sum -c -
```

Kích thước và sha256 của file F16 được ghi trong catalog (`qwen3.5:9b-f16`). So từng tensor với file BF16: 250/427 tensor được đổi (8,95 tỉ giá trị, trọng số lớn nhất 1,07), sai lệch tuyệt đối lớn nhất 3,0×10⁻⁸, 17.320 giá trị nhỏ hơn khoảng 3×10⁻⁸ bị làm tròn về 0; các tensor F32 giữ nguyên. Chạy thử bằng `server-b10920` trên CPU cho output JSON đúng schema và không sinh phần suy luận khi `enable_thinking` là `false`. Lần chạy stage đầu tiên tự đưa file lên dataset Kaggle `vector-cache-gguf-qwen3-5-9b-f16`. Bản llama.cpp trên Kaggle (b9637) đã hỗ trợ kiến trúc `qwen35`.

*Phục vụ trên Kaggle.* Model 17,9 GB được chia đôi trên hai T4 (4 slot, mỗi slot 16.384 token). Hai GPU dành cho LLM, nên reranker chạy ở một kernel khác:

```bash
uv run pharma-lab e2e llm-server --model qwen3.5:9b-f16 --hours 10 --kaggle-account auto
uv run pharma-lab e2e rerank-server --model qwen3-reranker:4b-fp16 --hours 10 --kaggle-account auto
```

`llm-server` ghi `data/work/serve/qwen3_5_9b_f16.env`: endpoint của cả sáu bước trỏ vào tunnel, `EXTRA_BODY` tắt chế độ suy luận mặc định của Qwen3.5, và timeout 600 giây. Trong tmux session của run, nạp cả hai file env rồi chạy thử `--limit 20` trước:

```bash
set -a; . data/work/serve/qwen3_5_9b_f16.env; . data/work/serve/qwen3_reranker_4b_fp16.env; set +a
uv run pharma-lab e2e run --run e2e-qwen35-9b --config full --limit 20 --deadline-seconds 900
```

Biến môi trường được ưu tiên hơn `backend/.env`, nên các run khác không bị ảnh hưởng. `EXTRA_BODY` và tên model được ghi vào `run.json`.

*Qua API thay cho Kaggle.* Cùng model có trên DeepInfra (key `DEEPINFRA_API_KEY` trong `pharma-lab/.env`); khi đó đặt endpoint từng bước bằng tay:

```bash
set -a; . ./.env; set +a
for ROLE in GUARDRAIL REPHRASE JUDGE REFINE ANSWER SUMMARIZER; do
  export PHARMA_LLM__ROLES__${ROLE}__BASE_URL=https://api.deepinfra.com/v1/openai
  export PHARMA_LLM__ROLES__${ROLE}__API_KEY=$DEEPINFRA_API_KEY
  export PHARMA_LLM__ROLES__${ROLE}__MODEL=Qwen/Qwen3.5-9B
  export PHARMA_LLM__ROLES__${ROLE}__EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
done
```

**Cấu hình ablation chỉ đặt được từ harness.** Backend không có biến môi trường nào để tắt bước của agent.

| `--config` | Rephrase | Judge/refine | Reranker |
| --- | --- | --- | --- |
| `full` | bật | bật | bật |
| `one-step` | tắt | tắt | bật |
| `no-judge-refine` | bật | tắt | bật |
| `no-rephrase` | tắt | bật | bật |
| `no-rerank` | bật | bật | tắt |
| `closed-book` | – | – | – |

`one-step` là guard → search (câu hỏi gốc) → answer, dùng cùng retriever, prompt trả lời và ngân sách evidence như `full`.

`closed-book` là baseline LLM thuần: cùng model của vai trò answer, nhận lịch sử hội thoại và câu hỏi nhưng không có guard, retrieval hay tài liệu nào. Nó đo phần mà corpus và pipeline đóng góp. Judge chấm nội dung câu trả lời như mọi cấu hình khác; faithfulness và các metric trích dẫn không xác định được khi không có context nên để trống. Cấu hình này không gọi retrieval nên không cần reranker server.

**Phiên bản prompt.** Run `e2e-v2` và `e2e-qwen35-9b-v2` dùng prompt answer đã sửa: bước answer luôn thấy câu hỏi gốc của người dùng (kèm câu viết lại khi rephrase có đổi), và quy tắc cho người hỏi là người dân không còn yêu cầu viết ngắn gọn. Khi câu hỏi không bị viết lại, prompt giữ nguyên từng byte (test `test_answer_prompt_is_unchanged_when_the_question_was_not_rewritten`), nên `one-step` và `no-rephrase` của bản cũ được chép sang run mới thay vì chạy lại. Không chạy tiếp một run cũ bằng code mới: harness bỏ qua câu đã có và sẽ trộn hai phiên bản.

## 2. Bộ golden

```bash
uv run pharma-lab e2e golden sample
uv run pharma-lab e2e golden check data/evaluation/e2e/authoring/answerable-01.authored.jsonl
uv run pharma-lab e2e golden resample --slots e2e-ans-0211,e2e-ans-0222
uv run pharma-lab e2e golden build
```

`golden build` ghi `data/evaluation/e2e/golden_e2e.jsonl` và `golden_e2e.manifest.json`; manifest được commit, bộ golden lưu cùng `pharma-lab data push`.

**Sửa bộ golden khi đã có run.** Sửa item trong các file `*.authored.jsonl`, kiểm bằng `golden check`, rồi `golden build`. Build giữ bản cũ thành `golden_e2e.<sha256>.jsonl`. Lần `e2e run` hoặc `e2e judge` kế tiếp trên một run cũ so hai bản theo từng item: item đổi lượt hội thoại thì câu trả lời và điểm bị thay (`status=superseded`) và được chạy lại, item chỉ đổi đáp án chuẩn hoặc hành vi mong đợi thì chỉ được chấm lại, các item khác giữ nguyên. Vì vậy chạy lại đúng các lệnh `e2e run` (không cần `--retry-errors`) rồi `e2e judge` của mọi cấu hình. Sau đó `e2e calibration refresh --run NAME` đưa các câu mù bị ảnh hưởng về trạng thái chưa chấm; chấm lại chúng trước khi `calibration score`. Không sửa golden khi một tiến trình `e2e judge` còn đang chạy trên run đó, vì nó chấm theo bản cũ đã nạp.

Lần rà soát ngày 22/09/2026 sửa 57 item theo tiêu chí cố định, áp như nhau cho mọi hệ thống: câu hỏi không nêu đối tượng (mẫu "tên biệt dược này"), gợi ý "(phần …)" lệch với câu hỏi, câu chép từ tài liệu không phải câu hỏi, câu hỏi "tra ở đâu" trong khi đáp án là nội dung, slug lọt vào câu hỏi (42 item sửa câu hỏi); ý chính nằm ngoài phạm vi câu hỏi hoặc không có trong đoạn trích (15 item chỉ sửa đáp án). Lỗi chính tả, không dấu và tên thuốc dễ nhầm của các nhóm `noisy_confuser`, `patient_natural` là độ khó có chủ đích nên giữ nguyên.

## 3. Chạy, chấm và báo cáo

Mỗi cấu hình chạy trong một tmux session riêng; chạy lại cùng lệnh sẽ resume.

```bash
uv run pharma-lab e2e run --run e2e-v1 --config full --limit 20        # chạy thử
uv run pharma-lab e2e run --run e2e-v1 --config full
uv run pharma-lab e2e run --run e2e-v1 --config full --retry-errors    # chạy lại các câu lỗi
uv run pharma-lab e2e judge --run e2e-v1 --config full --judge-model gemini-3.1-flash-lite
uv run pharma-lab e2e calibration export --run e2e-v1
uv run pharma-lab e2e calibration score --run e2e-v1
uv run pharma-lab e2e report --run e2e-v1
```

**`e2e run`** ghi vào `data/evaluation/e2e/runs/<run>/<config>/`:
- `run.json`: identity của lần chạy, gồm release, model theo role, cấu hình retrieval/rerank và bộ golden. Identity khác thì phải đặt tên run mới.
- `answers.jsonl`: mỗi câu một dòng, gồm câu trả lời, evidence đã đưa vào prompt, trích dẫn quy về `section_id`, token theo role và latency.

**`--limit N`** lấy N câu rải đều trên bộ golden.

**`--deadline-seconds S`** thay deadline của một lượt (mặc định lấy từ budget của backend, 90 giây). Dùng khi embedding/reranker chạy trên CPU hoặc endpoint LLM chậm; giá trị được ghi vào `run.json`, và latency báo cáo vẫn là thời gian thật.

**`e2e judge`** ghi `judgments.jsonl` và `judge.json` (model judge). Chấm lại một cấu hình bằng judge khác cần `--force`.

**Chỉ số.** Tên và định nghĩa theo các bài gốc; metric in đậm là metric chính, có mặt trong `main.tex` và `ablation.tex`.

| Metric | Định nghĩa | Nguồn | Áp dụng |
| --- | --- | --- | --- |
| **truthfulness** | Trung bình điểm theo lớp: perfect = 1, acceptable = 0,5, missing = 0, incorrect = −1 | CRAG (NeurIPS 2024) | answerable, multi-turn, unanswerable |
| **perfect / missing / hallucination rate**, acceptable rate | Tỉ lệ từng lớp. Câu cần trả lời: có ý chính bị nói sai → incorrect; không trả lời hoặc không nêu được ý chính nào → missing; nêu đủ → perfect; còn lại → acceptable. Câu unanswerable: từ chối → perfect, còn lại → incorrect | CRAG | như trên |
| **nugget recall** | Tỉ lệ ý chính được câu trả lời nêu đúng | TREC 2024 RAG | câu cần trả lời |
| contradiction rate | Tỉ lệ câu có ít nhất một ý chính bị nói sai | – | câu cần trả lời |
| **severe harm rate**, harmful error rate | Tỉ lệ câu có trả lời mà lỗi sai được judge xếp mức severe (hoặc minor/severe) | CHART (2025), TRIPOD-LLM (2025) | câu có trả lời |
| **faithfulness** | Tỉ lệ claim được evidence trong prompt hỗ trợ | RAGAS (EACL 2024) | câu có trả lời |
| **citation recall** | Số câu văn có trích dẫn được đoạn trích hỗ trợ, chia cho tổng số câu văn có nội dung; câu không trích dẫn tính là không được hỗ trợ | ALCE (EMNLP 2023) | câu có trả lời |
| supported citation rate | Tỉ lệ câu văn có trích dẫn được chính các đoạn đó hỗ trợ | ALCE | câu có trích dẫn |
| cited-source precision / recall | Trích dẫn trỏ vào section gold hoặc chunk được file nhãn chấp nhận | – | câu cần trả lời |
| **response rate**, over-refusal rate | Tỉ lệ câu cần trả lời có (hoặc không có) câu trả lời grounded | CRAG, RGB | câu cần trả lời |
| **negative rejection rate** | Câu unanswerable được từ chối, bằng chế độ trả lời hoặc bằng lời | RGB (AAAI 2024) | unanswerable |
| **redirect accuracy** | Câu ngoài phạm vi được chuyển hướng | – | out-of-scope |
| **injection defence rate**, attack success rate | Bị chặn hoặc trả lời an toàn / làm theo lệnh chèn | AgentDojo (NeurIPS 2024) | injection |
| answer relevancy, factual correctness F1 | Metric phụ của RAGAS | RAGAS | câu có trả lời |
| **tokens / LLM calls per turn**, latency p50/p95 | Chi phí; latency chỉ để tham khảo vì môi trường không phải production | – | tất cả |

**Thống kê:**
- **Khoảng tin cậy:** Wilson cho tỉ lệ (vẫn đúng với các nhóm 10 câu), bootstrap percentile 10.000 lần cho giá trị trung bình.
- **So sánh ablation:** hiệu số có cặp so với `full`, kiểm định hoán vị dấu (sign-flip; Smucker et al., CIKM 2007), p-value hiệu chỉnh Holm trên các cấu hình.
- **Metric dựa trên judge:** có thêm ước lượng PPI (Angelopoulos et al., Science 2023; ARES, NAACL 2024) từ mẫu hiệu chỉnh.
- **Provenance:** `provenance.json` ghi `run.json` và `judge.json` của từng cấu hình (model, thiết lập suy luận, ngày chạy), cùng checksum file nhãn relevance.

**Hiệu chỉnh:**
1. `calibration export` rút ngẫu nhiên 50 câu cho mỗi cấu hình (mặc định `full` và `one-step`; lặp `--config` để chọn, ví dụ thêm `--config closed-book`), trộn thứ tự và che danh tính, ghi vào `calibration/items.jsonl` kèm khóa ánh xạ riêng. Run `e2e-v2` và `e2e-qwen35-9b-v2` hiệu chỉnh cả ba cấu hình của bảng chính: `--config full --config one-step --config closed-book`.
2. Người chấm hiệu chỉnh ghi `calibration/grades.jsonl` theo schema `Grade` trong `pharma_lab/e2e/calibration.py`: ý chính, faithfulness, từng câu trích dẫn, injection, từ chối, mức tác hại.
3. `calibration score` ghi `agreement.json` và `ppi.json`:
   - nhãn nhị phân: percent agreement, Cohen's κ và Gwet's AC1;
   - điểm liên tục: Spearman ρ;
   - mọi chỉ số kèm CI bootstrap, ngưỡng tin cậy 0,6;
   - `ppi.json`: ước lượng PPI.

**`e2e report`** ghi các file sau vào `runs/<run>/reports/`:
- `main.csv|tex`: CSV có mọi cấu hình; bảng LaTeX chỉ gồm `full` và `one-step`, kèm CI;
- `ablation.csv|tex` (Δ so với `full`, paired bootstrap): CSV có CI và p-value; bảng LaTeX chỉ ghi Δ, dấu * khi p Holm < 0,05;
- `by_group.csv`;
- `calibration.csv|tex`;

Các file `.tex` dùng booktabs, có `\label` (`tab:e2e-main`, `tab:e2e-ablation`, `tab:e2e-calibration`) và tên metric/cấu hình dễ đọc, để copy thẳng vào `report/tables/` (xem `report/README.md`).
- `errors.md` (20 câu tệ nhất của `full`, gắn nhãn lỗi retrieval / citation / answer).
