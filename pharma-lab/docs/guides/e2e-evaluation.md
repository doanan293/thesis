# Đánh giá end-to-end

Đánh giá toàn bộ agent (guard → rephrase → search → judge/refine → answer) trên bộ golden 500 câu. Thiết kế nằm trong `docs/superpowers/specs/2026-09-16-e2e-golden-evaluation-design.md` ở thư mục gốc repo; cách soạn bộ golden nằm trong [e2e-golden-authoring.md](e2e-golden-authoring.md).

## 1. Chuẩn bị

Harness gọi backend trong cùng tiến trình, như `pharma-lab retrieve`, với settings đọc từ `../backend/.env` (`--backend-env-file`).

**Cần sẵn sàng:**
- Postgres và Qdrant đã import và publish release. Alias `chunks_current` trỏ vào collection của `qwen3-embedding:4b-fp16`.
- Endpoint LLM trong `PHARMA_LLM__*`. Model của từng role lấy từ `.env` và được ghi vào `run.json`.
- llama.cpp embedding (`PHARMA_RETRIEVAL__EMBEDDING__*`) và reranker (`PHARMA_RETRIEVAL__RERANK__*`), chạy qua `docker compose up -d` ở thư mục gốc repo.

**Endpoint LLM còn phục vụ judge**, mặc định `gpt-5-mini` với reasoning `medium`. Model khác thì truyền `--judge-model`.

**Cấu hình ablation chỉ đặt được từ harness.** Backend không có biến môi trường nào để tắt bước của agent.

| `--config` | Rephrase | Judge/refine | Reranker |
| --- | --- | --- | --- |
| `full` | bật | bật | bật |
| `one-step` | tắt | tắt | bật |
| `no-judge-refine` | bật | tắt | bật |
| `no-rephrase` | tắt | bật | bật |
| `no-rerank` | bật | bật | tắt |

`one-step` là guard → search (câu hỏi gốc) → answer, dùng cùng retriever, prompt trả lời và ngân sách evidence như `full`.

## 2. Bộ golden

```bash
uv run pharma-lab e2e golden sample
uv run pharma-lab e2e golden check data/evaluation/e2e/authoring/answerable-01.authored.jsonl
uv run pharma-lab e2e golden resample --slots e2e-ans-0211,e2e-ans-0222
uv run pharma-lab e2e golden build
```

`golden build` ghi `data/evaluation/e2e/golden_e2e.jsonl` và `golden_e2e.manifest.json`; manifest được commit, bộ golden lưu cùng `pharma-lab data push`.

## 3. Chạy, chấm và báo cáo

Mỗi cấu hình chạy trong một tmux session riêng; chạy lại cùng lệnh sẽ resume.

```bash
uv run pharma-lab e2e run --run e2e-v1 --config full --limit 20        # chạy thử
uv run pharma-lab e2e run --run e2e-v1 --config full
uv run pharma-lab e2e run --run e2e-v1 --config full --retry-errors    # chạy lại các câu lỗi
uv run pharma-lab e2e judge --run e2e-v1 --config full
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

**Chỉ số:**

| Chỉ số | Cách chấm |
| --- | --- |
| Faithfulness, factual correctness, answer relevancy | RAGAS 0.4.3 |
| Key-fact recall, contradiction, citation support, injection | judge có output cấu trúc |
| Behaviour, citation precision/recall | tính bằng code |

**Hiệu chỉnh:**
1. `calibration export` chọn 100 câu mù (50 của `full`, 50 của `one-step`) vào `calibration/items.jsonl`, kèm khóa ánh xạ riêng.
2. Người chấm hiệu chỉnh ghi `calibration/grades.jsonl` theo schema `Grade` trong `pharma_lab/e2e/calibration.py`.
3. `calibration score` tính Cohen's κ và Spearman ρ, kèm CI bootstrap. Ngưỡng tin cậy là 0,6.

**`e2e report`** ghi các file sau vào `runs/<run>/reports/`:
- `main.csv|tex`;
- `ablation.csv|tex` (Δ so với `full`, paired bootstrap);
- `by_group.csv`;
- `calibration.csv|tex`;
- `errors.md` (20 câu tệ nhất của `full`, gắn nhãn lỗi retrieval / citation / answer).
