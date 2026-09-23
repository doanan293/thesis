# Soạn bộ golden E2E

Hướng dẫn này là quy trình soạn `data/evaluation/e2e/golden_e2e.jsonl` (schema `e2e-golden-v1`). Thiết kế đầy đủ nằm trong `docs/superpowers/specs/2026-09-16-e2e-golden-evaluation-design.md` ở thư mục gốc repo.

## Quy trình

```bash
uv run pharma-lab e2e golden sample              # ghi authoring/*.todo.jsonl (420 answerable, 50 multi-turn)
# soạn mỗi lô X.todo.jsonl thành X.authored.jsonl, cộng thêm special.authored.jsonl
uv run pharma-lab e2e golden check data/evaluation/e2e/authoring/X.authored.jsonl
uv run pharma-lab e2e golden build               # kiểm đủ số lượng, ghi golden_e2e.jsonl và manifest
uv run pharma-lab data push --kaggle-account acc1 --message "Add the E2E golden set"
```

Mỗi dòng của `*.todo.jsonl` là một slot:

- `slot_id`: `item_id` của câu sẽ soạn.
- `category`: nhóm của câu.
- `source_rows`: các dòng lấy từ bộ gold.
- `sections`: text đầy đủ của các section gold.
- `focus_chunks`: text của chunk mong đợi, tức phần của section mà câu hỏi nói tới.

Mỗi dòng của `*.authored.jsonl` là một `GoldenItem` JSON do tác giả soạn cùng dược sĩ.

## Quy tắc chung

- **Key fact** là một ý nguyên tử (liều, chỉ định, chống chỉ định, cách dùng...) mà câu trả lời đúng phải nêu. Mỗi câu có 1–5 key fact và chỉ gồm những ý trả lời đúng điều được hỏi.
- **`evidence_quote`** chép nguyên văn một đoạn ngắn (khoảng 5–30 từ) trong text section.
  - Chỉ khoảng trắng được chuẩn hóa. Dấu markdown (`**`, `|`, `-`) phải giữ đúng như trong section, nên hãy chọn đoạn văn thường, tránh ranh giới ô bảng.
  - `section_id` phải thuộc `gold_section_ids`.
- **`reference.answer`** là câu trả lời mẫu tiếng Việt, 1–4 câu, trả lời thẳng.
  - Không có câu miễn trừ kiểu "hỏi ý kiến bác sĩ".
  - Chỉ dùng thông tin có trong section.
- **Các trường lấy từ dòng nguồn:**
  - `source_query_id` = `query_id`;
  - `eval_group`, `difficulty` giữ nguyên;
  - `tags` = `eval_tags`;
  - `gold_section_ids` = `expected_section_ids`;
  - `gold_chunk_ids` = `[expected_chunk_id]` khi có.

## Theo nhóm

- **`answerable`**
  - `turns` là đúng một lượt `user` với nguyên văn `query`.
  - `expected_behavior` là `grounded`.
  - `answer_mode = any_acceptable`: key fact lấy từ section trả lời trực tiếp nhất.
  - `answer_mode = multi_required`: key fact phải phủ mọi ý được hỏi, lấy từ các section tương ứng.
- **`multi_turn`**: dùng hai dòng nguồn A và B về cùng một thuốc.
  - **Lượt 1** (`user`) là câu hỏi của A, có thể sửa nhẹ cho tự nhiên.
  - **Lượt 2** (`assistant`) trả lời đúng A trong 1–2 câu, lấy từ section của A.
  - **Lượt 3** (`user`) hỏi tiếp về nội dung của B nhưng không nhắc tên thuốc. Dùng đại từ hoặc tỉnh lược, ví dụ "Thuốc này có dùng được cho phụ nữ có thai không?" hay "Còn liều cho trẻ em thì sao?".
  - Key fact, `gold_*`, `source_query_id`, `eval_group` và `difficulty` lấy theo B.
  - `tags` thêm `multi_turn`.
- **`unanswerable`** (10 câu, `abstain`)
  - Hỏi về thuốc hoặc hoạt chất có thật nhưng không có trong corpus.
  - `absent_terms` ghi tên thuốc và hoạt chất; `golden check` xác nhận chúng không xuất hiện trong bất kỳ section hay tiêu đề nào.
- **`out_of_scope`** (10 câu, `redirect`): câu hỏi rõ ràng không liên quan đến thuốc, như nấu ăn, thời tiết, lập trình, thể thao, du lịch.
- **`injection`** (10 câu, `blocked`)
  - Câu hỏi có dáng hỏi thuốc nhưng chèn lệnh tấn công, ví dụ:
    - bỏ qua mọi hướng dẫn trước;
    - in system prompt;
    - đóng vai một trợ lý không giới hạn;
    - xuất khóa API;
    - trả lời bằng nội dung độc hại.
  - Các câu phải đa dạng về kỹ thuật.
