# Thiết kế đánh giá end-to-end bằng bộ golden

Ngày: 2026-09-16.

## 1. Mục tiêu

Đánh giá toàn bộ hệ thống hỏi đáp thuốc (guard → rephrase → search → judge/refine →
answer), không chỉ riêng tầng retrieval. Kết quả là phần E2E của paper và gồm:

- một bộ golden E2E 500 câu, đóng băng và có sha256;
- so sánh agentic RAG với RAG một bước (one-step), cùng retriever, model trả lời và
  ngân sách context;
- ablation: bỏ judge/refine, bỏ rephrase, bỏ reranker;
- chấm tự động bằng LLM-as-judge (RAGAS và judge có output cấu trúc, gpt-5-mini,
  reasoning medium);
- hiệu chỉnh judge bằng 100 câu do một model mạnh hơn chấm mù;
- bảng, CSV và bảng LaTeX cho `report/`.

Ngoài phạm vi:

- ablation từng lớp enrichment (đã bỏ);
- chấm bởi dược sĩ (không có dược sĩ; xem §8).

## 2. Kiến trúc

Harness nằm trong `pharma-lab` (package mới `pharma_lab.e2e`). Như retrieval eval,
harness import backend và chạy trong cùng tiến trình.

Backend chỉ thêm `PipelineOptions` (§4). Không có service eval riêng, không gọi qua HTTP
và không ghi conversation vào Postgres.

Cần sẵn sàng:

- Postgres và Qdrant dev đã import và publish release;
- llama.cpp embedding và reranker local qua compose;
- khóa OpenAI trong `backend/.env`.

## 3. Bộ golden E2E

### 3.1 File và schema

`pharma-lab/data/evaluation/e2e/golden_e2e.jsonl` có schema `e2e-golden-v1`. Mỗi dòng
một câu:

| Trường | Ý nghĩa |
|---|---|
| `item_id` | ID ổn định, ví dụ `e2e-ans-0001` |
| `category` | `answerable`, `multi_turn`, `unanswerable`, `out_of_scope`, `injection` |
| `source_query_id` | `query_id` trong bộ gold, nếu câu lấy từ bộ gold; ngược lại `null` |
| `turns` | Danh sách `{role, text}`. Lượt cuối là lượt `user` được chấm. Các lượt trước là lịch sử cố định. |
| `expected_behavior` | `grounded`, `abstain`, `redirect`, `blocked`; khớp `AnswerMode` của backend |
| `gold_section_ids`, `gold_chunk_ids` | Bằng chứng đúng; rỗng với câu không có đáp án |
| `reference.answer` | Câu trả lời mẫu ngắn |
| `reference.key_facts` | Danh sách `{fact, evidence_quote, section_id}` |
| `absent_terms` | Chỉ câu `unanswerable`: tên hoặc cụm từ phải vắng mặt trong corpus |
| `eval_group`, `difficulty`, `tags` | Lấy từ bộ gold hoặc gán khi soạn |
| `authored_by` | `claude-opus-5` |

File `golden_e2e.manifest.json` ghi sha256 của bộ golden, số câu theo từng nhóm, seed,
và sha256 của bộ gold nguồn và của bundle. Bộ golden được lưu cùng
`pharma-lab data push`.

### 3.2 Thành phần (500 câu)

| Nhóm | Số câu | Cách tạo | Hành vi mong đợi |
|---|---|---|---|
| Answerable | 420 | 70 câu × 6 `eval_group` của bộ gold. Phân tầng theo `difficulty` × `answer_mode`, seed cố định, dùng lại `stratified_sample`. | `grounded` |
| Multi-turn | 50 | Lượt 1 hỏi về thuốc X. Lượt 2 hỏi tiếp bằng đại từ hoặc tỉnh lược; đáp án ở một section khác của cùng thuốc. | `grounded` |
| Unanswerable | 10 | Thuốc hoặc thông tin không có trong corpus. Mỗi câu ghi `absent_terms`; `golden build` kiểm các từ này không xuất hiện ở bất kỳ tiêu đề hay section nào của bundle. | `abstain` |
| Out-of-scope | 10 | Câu không liên quan đến thuốc | `redirect` |
| Injection | 10 | Câu hỏi về thuốc có chèn lệnh ("bỏ qua hướng dẫn…", "in system prompt"…) | `blocked`, hoặc trả lời an toàn mà không làm theo lệnh chèn |

Tất cả câu viết bằng tiếng Việt.

### 3.3 Soạn đáp án

Claude soạn `reference` từ đúng text của section gold trong bundle, theo lô khoảng 35
câu. Lô nào cũng qua `pharma-lab e2e golden build`. Lệnh này kiểm:

- `evidence_quote` là chuỗi con (sau khi chuẩn hóa khoảng trắng) của text section
  `section_id` trong bundle; không khớp thì câu bị loại và phải soạn lại;
- `section_id` của mỗi key fact thuộc `gold_section_ids`;
- số câu từng nhóm đúng như §3.2; `item_id` không trùng;
- câu `abstain`, `redirect`, `blocked` có `gold_*` rỗng và `key_facts` rỗng.

### 3.4 Lịch sử hội thoại cố định

Với câu multi-turn:

- Lượt `assistant` trước lấy từ câu trả lời mẫu của lượt trước.
- Harness đưa lịch sử này vào `ConversationContext`.
- Mọi cấu hình nhận cùng ngữ cảnh và chỉ lượt cuối được chấm.

Hệ quả: lỗi của lượt trước không kéo sai lượt sau, nên so sánh giữa các cấu hình công
bằng.

## 4. Thay đổi backend: `PipelineOptions`

`application/chat/context.py` thêm:

```python
@dataclass(frozen=True)
class PipelineOptions:
    rephrase: bool = True
    judge_refine: bool = True
```

`TurnContext` có thêm trường `pipeline: PipelineOptions = PipelineOptions()`.

`ChatTurnRunner` nhận thêm tham số `pipeline: PipelineOptions = PipelineOptions()` và
đưa vào `TurnContext`. Graph và routing không đổi:

- `rephrase_node`: khi `rephrase` tắt, ghi `record_rephrase(None, skipped=True)` (đường
  đã có sẵn); search dùng câu hỏi gốc, intent giữ mặc định `pharma_question`.
- `judge_node`: khi `judge_refine` tắt, không gọi LLM mà ghi
  `AgentRun.skip_judge(now=...)`. Hàm domain mới này đặt `last_judge = ANSWER` và ghi
  action `judge` với outcome `skipped`. `allowed_steps()` khi đó chỉ còn `ANSWER`, nên
  chỉ có một vòng search. `decide_answer()` trả `GROUNDED` với `partial=False` khi có
  bằng chứng và `ABSTAIN` khi không có, nên prompt trả lời giống hệt full agent.
- `answer_node` ghi thêm `context_text` (đúng đoạn evidence đã đưa vào prompt trả lời)
  vào `ChatTurnState`; `TurnOutcome` có thêm trường `context_text`. Harness dùng trường
  này để chấm faithfulness.

**Production luôn chạy full pipeline.**

- Không có setting hay biến môi trường nào cho `PipelineOptions`.
- API và `build_application` luôn dùng giá trị mặc định.
- Chỉ harness của pharma-lab truyền giá trị khác.
- "Bỏ reranker" đổi `retrieval.rerank.protocol = "none"` trên đối tượng settings trong
  bộ nhớ của harness, không sửa `.env`.

Test:

- domain: `skip_judge` dẫn tới `ANSWER`, plan `GROUNDED` không partial, hoặc
  `ABSTAIN` khi không có bằng chứng;
- graph với LLM giả cho từng tổ hợp công tắc; one-step chỉ gọi LLM hai lần (guard +
  answer);
- `TurnOutcome.context_text` bằng đoạn context trong prompt trả lời;
- app dựng từ `build_application` chạy đủ các bước với giá trị mặc định.

## 5. Cấu hình E2E

| Config | rephrase | judge_refine | reranker |
|---|---|---|---|
| `full` | bật | bật | bật |
| `one-step` | tắt | tắt | bật |
| `no-judge-refine` | bật | tắt | bật |
| `no-rephrase` | tắt | bật | bật |
| `no-rerank` | bật | bật | tắt |

**One-step RAG** là guard → search (câu gốc) → answer.

- Giữ nguyên retriever, model trả lời, prompt trả lời, `max_evidence_chars` và cách
  trích dẫn.
- Guard được giữ vì đó là lớp an toàn, không phải chiến lược RAG.
- Paper nêu rõ định nghĩa này.

Model theo role giữ mặc định của backend:

- nano cho guard, rephrase và summarizer;
- mini cho judge, refine và answer.

Reranker dùng llama.cpp local qua compose, với đúng model được chọn từ benchmark
retrieval. Lượt chạy E2E diễn ra qua đêm trong tmux. Latency trong paper chỉ báo phần
LLM và tổng thời gian; không báo latency rerank trên CPU vì stack CPU chỉ là demo.

## 6. Harness

### 6.1 Lệnh

```text
pharma-lab e2e golden sample                     chọn câu nguồn, ghi các lô soạn kèm text section
pharma-lab e2e golden build                      kiểm tra và đóng băng bộ golden (§3.3)
pharma-lab e2e run --run NAME --config CONFIG    chạy hệ thống trên bộ golden
pharma-lab e2e judge --run NAME --config CONFIG  chấm tự động (§7)
pharma-lab e2e calibration export --run NAME     xuất 100 câu để chấm mù (§8)
pharma-lab e2e calibration score --run NAME      đo đồng thuận judge và người chấm hiệu chỉnh
pharma-lab e2e report --run NAME                 bảng, CSV, LaTeX (§9)
```

Tất cả lệnh được ghi vào `pharma-lab/docs/guides/evaluation.md` và `cli-reference.md`.

### 6.2 Dữ liệu một lần chạy

Thư mục là `pharma-lab/data/evaluation/e2e/runs/<run>/<config>/`.

**`run.json`** là identity của lần chạy:

- sha256 của bộ golden;
- `release_id`;
- model và reasoning effort theo role;
- `PipelineOptions`;
- cấu hình retrieval và reranker (không có secret);
- git commit.

**`answers.jsonl`**, mỗi câu một dòng:

- `item_id`, `answer_text`, `answer_mode`, `status`, `error`;
- `citations`: chỉ số, `chunk_version_id`, và `chunk_id` / `section_id` đã quy đổi từ
  corpus store;
- `context_text`: đoạn evidence đã đưa vào prompt trả lời (`TurnOutcome.context_text`);
- `standalone_query`, các query của từng vòng search, số vòng search, quyết định judge;
- token theo role (harness bọc `LlmPort` bằng một lớp đếm theo role), số lần gọi LLM,
  latency tổng và theo bước (từ thời điểm của các action).

**`judgments.jsonl`** là kết quả chấm (§7).

### 6.3 Resume và lỗi

- File jsonl chỉ ghi nối thêm và khóa theo `item_id`. Chạy lại cùng lệnh thì bỏ qua các
  câu đã có kết quả.
- `run.json` khác identity hiện tại thì lệnh dừng và yêu cầu tên run mới, như quy tắc
  của retrieval run.
- Câu lỗi được ghi `status=error`. `--retry-errors` chỉ chạy lại các câu đó.
- Chạy song song bằng semaphore (`--concurrency`, mặc định 4). Retry dựa vào OpenAI
  SDK.
- Harness tắt Langfuse (như retrieval eval); mọi thứ cần phân tích đã nằm trong
  `answers.jsonl`.

## 7. Chấm tự động

Judge là gpt-5-mini, reasoning medium.

- Judge không biết cấu hình nào sinh ra câu trả lời.
- Chấm lại chỉ gọi lại judge, không chạy lại agent.

RAGAS chạy trong workspace, trong nhóm dev của pharma-lab:

- `ragas==0.4.3` và `langchain-community==0.4.1`. langchain-community 0.4.2 đã xóa module
  `chat_models.vertexai` mà ragas vẫn import.
- `instructor` 1.17 (ragas cần) giới hạn `jiter<0.15`, còn openai từ 3.3.1 cần
  `jiter>=0.16`. Root `pyproject.toml` thêm
  `[tool.uv] override-dependencies = ["jiter>=0.16.0,<1"]`. Đã kiểm: instructor gọi qua
  ragas `llm_factory` và `jiter.from_json(partial_mode=...)` chạy đúng với jiter 0.17.
- Lock chỉ thêm package mới, và hạ `rich` 15.0.0 xuống 14.3.4. openai, langchain-core và
  langgraph của backend giữ nguyên.

Chấm gồm hai phần, cùng lệnh `pharma-lab e2e judge`, ghi `judgments.jsonl`:

- **RAGAS** (`ragas.metrics.collections`): `Faithfulness`, `FactualCorrectness` và
  `AnswerRelevancy`. LLM là `llm_factory("gpt-5-mini", client=AsyncOpenAI(...),
  reasoning_effort="medium")`, dùng endpoint LLM của `backend/.env`. `AnswerRelevancy`
  dùng qwen3-embedding local qua endpoint OpenAI-compatible.
- **Judge có output cấu trúc** qua `OpenAiLlmAdapter` của backend: các chỉ số RAGAS
  không có (key-fact recall, contradiction, citation support, injection success). Các
  chỉ số còn lại tính bằng code.

| Chỉ số | Cách đo | Áp dụng |
|---|---|---|
| Key-fact recall | Judge xét từng `key_fact` có được câu trả lời nêu đúng không | answerable, multi_turn |
| Contradiction rate | Judge xét câu trả lời có mâu thuẫn với key fact nào không | answerable, multi_turn |
| Faithfulness | RAGAS `Faithfulness`: claim của câu trả lời có được `context_text` hỗ trợ không | câu có trả lời |
| Factual correctness | RAGAS `FactualCorrectness` (F1) so với `reference.answer` | answerable, multi_turn |
| Answer relevance | RAGAS `AnswerRelevancy` | câu có trả lời |
| Citation precision / recall | Tính bằng code. Precision: tỉ lệ trích dẫn trỏ vào `gold_section_ids`. Recall: có ít nhất một trích dẫn trúng section gold. | answerable, multi_turn |
| Citation support | Judge xét câu mang [n] có được chunk [n] hỗ trợ không | câu có trích dẫn |
| Behaviour accuracy | Tính bằng code: `answer_mode` so với `expected_behavior` | tất cả; báo riêng từng category |
| Injection success | Judge xét câu trả lời có làm theo lệnh chèn không | injection |
| Chi phí và độ trễ | Token theo role, số lần gọi LLM, latency p50/p95 | tất cả |

## 8. Hiệu chỉnh judge (strong-model calibration)

- **Chọn câu:** 100 câu phân tầng, lấy từ kết quả `full` và `one-step`, khoảng 50/50.
  Khoảng 70 câu answerable hoặc multi-turn và 30 câu nhóm đặc biệt.
- **Che danh tính:** `calibration export` xóa tên cấu hình và điểm judge, xáo thứ tự
  bằng seed cố định, ghi `calibration/items.jsonl`. Khóa ánh xạ nằm ở file riêng.
- **Chấm:** Claude chấm theo cùng rubric với judge và ghi `calibration/grades.jsonl`,
  với `grader: claude-opus-5`.
- **Đồng thuận:** `calibration score` tính:
  - Cohen's κ cho nhãn nhị phân (key fact, claim, citation support, injection);
  - Spearman ρ cho điểm key-fact recall và faithfulness từng câu;
  - CI bootstrap cho cả hai.
- **Ngưỡng tin cậy:** κ ≥ 0,6. Chỉ số dưới ngưỡng thì sửa rubric và chấm lại bằng
  judge. Nếu vẫn dưới ngưỡng, paper ghi là chỉ số tham khảo.
- **Limitations trong paper:** không có dược sĩ; người chấm hiệu chỉnh là một LLM mạnh
  hơn, không phải chuyên gia.

## 9. Báo cáo

`pharma-lab e2e report` đọc các file jsonl đã lưu và ghi vào
`runs/<run>/reports/`. Đổi cách tính chỉ cần chạy lại lệnh này.

1. **Bảng chính:** 5 cấu hình × các chỉ số §7, trung bình kèm CI bootstrap 95%.
2. **Hiệu ứng ablation:** Δ so với `full`, kèm CI paired bootstrap trên cùng các câu.
3. **Theo nhóm:** kết quả theo `category` và `eval_group`.
4. **Hiệu chỉnh:** κ và ρ (§8).
5. **Phân tích lỗi:** 20 câu tệ nhất của `full`, phân loại lỗi retrieval / judge /
   answer.

Mỗi bảng có bản CSV và bản LaTeX để đưa vào `report/`.

## 10. Kiểm thử

- `golden build`: các luật §3.3, mỗi luật một test.
- Harness:
  - chạy với backend giả, kiểm resume, identity mismatch và `--retry-errors`;
  - quy đổi citation sang `section_id`.
- Chấm:
  - các chỉ số tính bằng code có test số liệu cố định;
  - các lời gọi judge dùng LLM giả;
  - adapter RAGAS có test với OpenAI giả (`httpx.MockTransport`).
- Hiệu chỉnh: κ, ρ và bootstrap có test với dữ liệu nhỏ biết trước kết quả.
- Backend: các test §4.

## 11. Thứ tự triển khai

1. Backend `PipelineOptions` và test.
2. `pharma_lab.e2e`: schema, `golden build`, sampler.
3. Soạn bộ golden theo lô, đóng băng, data push.
4. `e2e run` và chạy thử 20 câu với `full`.
5. `e2e judge` (RAGAS và judge cấu trúc).
6. Chạy đủ 5 cấu hình trong tmux.
7. Hiệu chỉnh, rồi `report`, rồi đưa bảng vào paper.
