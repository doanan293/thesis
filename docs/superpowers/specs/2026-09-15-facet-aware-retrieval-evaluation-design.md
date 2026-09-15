# Thiết kế đánh giá retrieval đa ý theo facet

Ngày: 2026-09-15.

## 1. Mục tiêu

Sửa tầng đánh giá cho 500 câu hỏi `multi_intent` để đo khả năng tìm đủ bằng chứng
trả lời từng ý trong câu hỏi, thay vì bắt buộc kết quả phải trùng đúng các section
Dược thư đã dùng để sinh câu hỏi.

Thay đổi chỉ đọc lại các artifact hiện có:

- gold query `section_retrieval_eval.jsonl`;
- retrieval candidates `candidates.jsonl`;
- rerank scores `rerank_scores.jsonl`.

Không embedding lại corpus hoặc query, không dựng lại index, không chạy retrieval và
không gọi reranker/GPU.

## 2. Vấn đề hiện tại

`Multi-all-hit@K` hiện xem hai `expected_section_ids` là hai bằng chứng duy nhất được
chấp nhận. Điều này không phù hợp với corpus có cả Dược thư và tờ hướng dẫn sử dụng.
Một chunk tờ HDSD có thể chứa đồng thời, chẳng hạn, phần liều dùng và chống chỉ định,
nhưng vẫn bị tính là sai nếu ID không trùng hai section Dược thư.

Trên run `hybrid-qwen4b-p50-k30-rrf2`, trong 500 câu đa ý:

- hybrid có 435 câu trúng đủ hai exact section ở top 10;
- rerank 4B cũ có 386 câu;
- reranker làm mất 88 exact-section hits và cứu 39 hits;
- 616/880 vị trí top 10 của 88 câu bị mất là chunk tờ HDSD.

Ví dụ, với câu hỏi về liều dùng và chống chỉ định của acarbose, một chunk Dorobay
chứa cả hai mục được reranker đưa lên hạng 1 nhưng metric hiện tại vẫn không coi là
phủ hai ý. Do đó, số hiện tại đo exact-source retrieval chứ chưa đo multi-intent
evidence coverage.

Các artifact Qwen3 4B hiện có cũng dùng contract `completion_logprobs` lịch sử, trong
khi model catalog đã chuyển sang `native_rerank`. Metrics cần đọc identity bất biến
của chính artifact để tái lập báo cáo lịch sử, không diễn giải artifact bằng catalog
hiện hành.

## 3. Phạm vi

### Trong phạm vi

- Bổ sung biểu diễn facet cho câu `multi_intent` tại tầng đánh giá.
- Chấm một candidate theo entity và evidence facet trong nội dung.
- Cho phép một candidate phủ nhiều facet.
- Báo cáo facet coverage làm metric chính cho câu đa ý.
- Giữ exact-section coverage làm diagnostic.
- Đọc rerank identity từ manifest artifact khi tính lại metrics lịch sử.
- Tái tạo các report từ artifact hiện có.
- Thêm test và tài liệu giải thích metric.

### Ngoài phạm vi

- Embedding lại corpus hoặc query.
- Chạy lại BM25, dense hoặc hybrid retrieval.
- Chấm lại reranker bằng GPU.
- Đổi model, index, fusion, rerank policy hoặc backend production.
- Coverage-preserving selection.
- Đánh giá chất lượng câu trả lời end-to-end.

## 4. Mô hình dữ liệu đánh giá

### 4.1 Required facets

Mỗi `multi_intent` row được chuyển thành danh sách facet bắt buộc từ hai gold section.
Mỗi facet gồm:

- `facet_id`: taxonomy ổn định như `indication`, `dosage`, `contraindication`, `adr`,
  `interaction`, `pregnancy_lactation`;
- `entity_id`: định danh thuốc/hoạt chất lấy từ gold section;
- `exact_section_id`: section Dược thư gốc;
- các heading/marker được chấp nhận cho facet.

Phép suy diễn phải tất định và fail closed. Section ID không ánh xạ được sang taxonomy
thì metrics dừng với lỗi nêu query và section, không tự gán `general_info`.

Gold JSONL hiện tại không bị ghi đè. Required facets được dẫn xuất khi đánh giá và
được ghi vào metrics artifact để audit.

### 4.2 Candidate evidence

Candidate giữ nguyên payload và `document_text`. Một candidate phủ facet khi thỏa một
trong hai đường:

1. `section_id` trùng `exact_section_id`; hoặc
2. candidate là bằng chứng thay thế, đồng thời:
   - khớp entity của câu hỏi;
   - có heading/marker của facet;
   - có nội dung không rỗng thuộc vùng facet đó.

Entity match dùng tên chuẩn dẫn xuất từ tiền tố `drug:<slug>` của exact section.
Sau khi chuẩn hóa Unicode, chữ hoa/thường, dấu câu và khoảng trắng, candidate chỉ
khớp khi tên chuẩn xuất hiện trong context header của Dược thư hoặc trong vùng
`Thành phần`/hoạt chất của tờ HDSD. Không dò tên thuốc trên toàn bộ body vì một mục
tương tác thuốc có thể nhắc tới thuốc khác.

Facet matcher ưu tiên heading cấu trúc. Với đoạn bị cắt qua ranh giới chunk và không
còn heading, matcher dùng tập marker riêng của facet trong đúng vùng văn bản, ví dụ
`thời kỳ mang thai`, `phụ nữ có thai` cho `pregnancy_lactation`. Mỗi marker và các
ngoại lệ loại trừ được khai báo tập trung, có test riêng và được ghi trong audit.

Chỉ xuất hiện từ khóa rời rạc không đủ để được tính. Ví dụ, từ “liều” nằm trong câu
“quá liều” không được xem là evidence cho facet `dosage`.

Một chunk có nhiều vùng nội dung hợp lệ được phép phủ nhiều facet. Một tờ HDSD sai
hoạt chất không được tính dù có đúng heading.

### 4.3 Audit record

Mỗi query trong `metrics.jsonl` ghi thêm:

- danh sách required facets;
- facet nào được phủ ở từng cutoff;
- candidate đầu tiên phủ mỗi facet;
- đường match `exact_section` hoặc `alternative_evidence`;
- evidence span/heading dùng để quyết định match.

Thông tin audit là dữ liệu dẫn xuất từ artifact hiện có và không gọi LLM.

## 5. Metrics

Với tập facet bắt buộc \(F_q\) của query \(q\), và \(C_q@K\) là hợp các facet được
phủ bởi top K candidates:

- `Intent Recall@K = |F_q ∩ C_q@K| / |F_q|`;
- `All Intents Covered@K = 1` khi mọi facet trong `F_q` được phủ, ngược lại bằng 0.

Báo cáo trung bình hai metric trên các query `multi_required` tại K = 3, 5, 10 và 30.

Metric cũ được giữ và đổi nhãn rõ nghĩa:

- `Exact-section Recall@K`;
- `All Exact Sections Hit@K`.

`Hit@K` và MRR hiện hành cho các nhóm single-intent không thay đổi trong phạm vi này.
Không dùng `All Exact Sections Hit@K` để kết luận reranker cải thiện hay làm giảm khả
năng trả lời câu đa ý.

## 6. Đọc artifact rerank lịch sử

Metrics loader lấy `model_sha256`, `request_contract_sha256`, model name và protocol
từ identity trong manifest rerank đã đăng ký. Nó dùng các giá trị đó để xác thực từng
score record và checksum bundle.

Model catalog hiện hành chỉ được dùng để chọn tên model theo CLI khi cần; nó không
được thay identity của artifact đã đóng dấu. Loader vẫn dừng nếu manifest, run
registry, checksum hoặc score records mâu thuẫn.

Thay đổi này cho phép tái tính metrics từ artifact `completion_logprobs` cũ mà không
giả rằng đó là điểm `native_rerank`. Report phải hiển thị protocol và model digest của
artifact để tránh nhập nhằng.

## 7. Luồng dữ liệu

```text
gold query ──► derive required facets ───────────────┐
                                                    │
candidates ──► entity/facet evidence matcher ───────┼─► facet metrics + audit
                                                    │
rerank manifest + scores ──► immutable validation ──┘

                                      └──────────────► exact-section diagnostics
```

Baseline và từng rerank variant dùng cùng một scorer. Khác biệt duy nhất giữa chúng
là thứ tự candidates.

## 8. Kiểm thử

### Unit tests

- Exact gold section luôn phủ facet tương ứng.
- Chunk acarbose/Dorobay có cả liều dùng và chống chỉ định phủ hai facet.
- Candidate đúng facet nhưng sai thuốc không được tính.
- Candidate đúng thuốc nhưng thiếu vùng facet không được tính.
- “Quá liều” không bị nhận nhầm là `dosage`.
- Một candidate được phép phủ nhiều facet.
- Section ID không ánh xạ được làm phép đánh giá dừng rõ ràng.
- Metric facet và exact-section được tích lũy bằng đúng denominator.
- Artifact lịch sử được đọc bằng identity trong manifest.
- Mâu thuẫn manifest/score record vẫn bị từ chối.

### Regression tests

- Tái hiện câu `multi-intent-0016`: Dorobay top 1 phủ `dosage` và
  `contraindication`, dù không trùng hai section Dược thư.
- Các metric single-intent hiện có không đổi.

### Verification

- Chạy test evaluation của seed-pipeline.
- Chạy toàn bộ test seed-pipeline, ruff, format check và pyrefly.
- Tái tạo metrics từ run hiện có mà không khởi động dịch vụ model.
- Kiểm tra report mới hiển thị cả facet metrics, exact-section diagnostics và rerank
  artifact protocol.
- Soát thủ công một mẫu false positive/false negative từ mỗi cặp facet.

## 9. Diễn giải kết quả

Không đặt tiêu chí reranker phải thắng dense hoặc hybrid. Kết quả được báo cáo với
paired confidence interval trên cùng query khi so baseline và rerank variant.

Nếu facet coverage tăng nhưng exact-section coverage giảm, kết luận hợp lệ là
reranker ưu tiên bằng chứng trả lời được từ nguồn thay thế. Nếu cả hai cùng giảm, đó
là failure mode thật. Nếu kết quả phụ thuộc mạnh vào matcher tự động, báo cáo phải
nêu hạn chế và kèm kết quả soát mẫu.

## 10. Rủi ro và kiểm soát

- **False positive do từ khóa:** chỉ match vùng heading có nội dung và chặn các cụm
  dễ nhầm như “quá liều”.
- **Sai entity:** yêu cầu entity match trước khi xét facet.
- **Leaflet chứa nhiều thuốc:** matcher fail closed khi không xác định được hoạt chất
  tương ứng.
- **Artifact lịch sử bị hiểu là native:** report ghi protocol/digest từ manifest.
- **Metric mới khó so với công trình khác:** tiếp tục báo cáo Hit@K/MRR và exact-section
  diagnostic; facet metrics được định nghĩa công thức rõ ràng.
