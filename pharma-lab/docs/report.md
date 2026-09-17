# Báo cáo tiến độ: phần AI của hệ thống hỏi đáp thuốc

Cập nhật 15/09/2026. Gồm dữ liệu, retrieval, đánh giá và backend agent (không gồm frontend). Số liệu lấy
từ manifest, validation report và metrics trong `seed-pipeline/data/`.

## 1. Tóm tắt

| Hạng mục | Kết quả |
| --- | --- |
| Nguồn dữ liệu | Dược thư Quốc gia Việt Nam (PDF 1.668 trang) và 2.434 trang tờ hướng dẫn sử dụng thuốc |
| Corpus | 3.120 tài liệu, 12.326 section, 24.955 chunk, khoảng 42,7 triệu ký tự |
| Enrichment | Tất định, không gọi LLM khi index: context header, glossary 71 thuật ngữ, 2.409 colloquial mapping |
| Bộ test retrieval | 10.000 câu hỏi có nhãn section/chunk đúng, 6 nhóm, 12 dạng câu hỏi, 17 loại ý định |
| Benchmark | 6 embedding model, BM25/dense/hybrid, 5 reranker: 14 cấu hình, mỗi cấu hình chạy đủ 10.000 câu |
| Cấu hình tốt nhất | Hybrid (dense `qwen3-embedding:4b` + BM25, RRF k=2) + `qwen3-reranker:4b`: Hit@3 88,51%, Hit@10 96,98%, MRR 0,8060 |
| Backend | Agentic RAG trên LangGraph chạy được qua CLI và HTTP API, dùng đúng cấu hình retrieval ở trên |
| Chất lượng code | Backend 477 test, seed-pipeline 469 test; ruff và pyrefly strict, không tắt rule |
| Bước tiếp theo | Đánh giá end-to-end agent (mục 7) |

## 2. Dữ liệu

Xử lý offline trong `seed-pipeline`; đầu ra là knowledge bundle (`knowledge-bundle/v1`) mà backend nạp
bằng `pharma-agent corpus import`.

```text
PDF Dược thư ──► PyMuPDF text ──► làm sạch markdown ──► tách chuyên luận/mục ─┐
              └► Docling tables ──► curate bảng ──────────────────────────────┤
HTML tờ HDSD ──► BeautifulSoup ──► markdown mục "Hướng dẫn sử dụng" ───────────┤
                                                                               ▼
                                          canonical blocks ──► validation (rag-final-v3)
                                                                               ▼
                               knowledge bundle ──► chunk + enrichment (chunker của backend)
                                                                               ▼
                                                         embedding 5 model (Kaggle GPU)
```

### 2.1 Dược thư Quốc gia Việt Nam

| Thông số | Giá trị |
| --- | ---: |
| Ấn bản | Xuất bản lần thứ 2, NXB Y học, 2018 |
| Số trang PDF | 1.668 |
| Chuyên luận thuốc | 689 |
| Chuyên luận chung (hướng dẫn, phụ lục) | 25 |
| Section | 9.920 |
| Bảng trích bằng Docling và curate | 258 |
| Override bảng nối trang / bảng trùng | 6 / 54 |

Xử lý: sửa từ bị tách sai và lỗi dấu câu khi trích text, tách theo mục chuẩn của Dược thư, curate bảng
nối trang hoặc bị lặp, dựng canonical blocks có số trang để trích dẫn.

### 2.2 Tờ hướng dẫn sử dụng thuốc

| Thông số | Giá trị |
| --- | ---: |
| Ngày crawl | 24/07/2026 |
| Trang HTML | 2.434 (38 nhóm thuốc) |
| Parse được mục "Hướng dẫn sử dụng" | 2.407 |
| Tờ hướng dẫn vào corpus | 2.406 |

- Nhóm lớn nhất: tim mạch – huyết áp (279), kháng sinh (218), thần kinh (163), ung thư (162).
- HTML gốc được kiểm sha256 theo `manifest.json`; định danh trong corpus chỉ mô tả nội dung
  (`leaflet:<nhóm>:<slug>`).

### 2.3 Kiểm tra chất lượng

Validation bản build hiện tại: 0 lỗi, 0 cảnh báo; 32 nhóm section trùng văn bản (quy chế, bảo quản lặp
giữa các chuyên luận) đã được duyệt.

## 3. Chunking, enrichment và embedding

- Một chunker duy nhất trong backend (`chunker-v1`), seed-pipeline import lại, nên chunk khi đánh giá và
  chunk trên production là một.
- Cắt theo section rồi theo block: prose theo đoạn/câu, bảng theo hàng (lặp header), mục lục không cắt
  giữa mục; tối đa 3.000 ký tự.
- `chunk_version_id` (UUIDv5 từ nội dung và phiên bản chunker) giữ citation ổn định qua các lần build.

| Loại tài liệu | Section | Chunk | Ký tự TB / chunk | Trung vị |
| --- | ---: | ---: | ---: | ---: |
| Chuyên luận thuốc (Dược thư) | 9.881 | 10.680 | 783 | 402 |
| Chuyên luận chung (Dược thư) | 39 | 477 | 2.728 | 2.953 |
| Tờ hướng dẫn sử dụng | 2.406 | 13.798 | 2.393 | 2.750 |
| **Tổng** | **12.326** | **24.955** | | |

Hydrate policy (agent đọc bao nhiêu văn bản khi một chunk được chọn):

| Chiến lược | Điều kiện | Số section |
| --- | --- | ---: |
| `full_section` | Section ≤ 16.000 ký tự: đọc cả section | 11.631 |
| `chunk_window` | Section dài hơn: đọc chunk trúng và 1 chunk mỗi bên | 694 |
| `search_only` | Chỉ dùng để tìm (mục lục) | 1 |

Enrichment ghép tất định vào `embedding_text`, không dùng LLM (chi phí index bằng 0, lặp lại được):

| Lớp | Nội dung | Phạm vi |
| --- | --- | --- |
| Context header | Tên thuốc > đường dẫn mục, ví dụ `Paracetamol > Liều lượng và cách dùng` | Mọi chunk |
| Colloquial mapping | Tên gọi dân gian ("Panadol đỏ"), dấu hiệu nhận biết ("hộp màu đỏ"), tên sản phẩm | 2.409 mapping, phủ toàn bộ 13.798 chunk tờ hướng dẫn |
| Glossary | Viết tắt và thuật ngữ kèm diễn giải, ví dụ `ADR = tác dụng không mong muốn; Adverse Drug Reactions` | 71 thuật ngữ; 11.035 chunk (44,2%) |

Embedding tính trên Kaggle GPU (T4), cache theo `sha256(embedding_text)`:

| Model | Chiều vector | Kích thước file |
| --- | ---: | ---: |
| `embeddinggemma:300m` | 768 | 105 MB |
| `bge-m3:567m-fp16` | 1.024 | 139 MB |
| `qwen3-embedding:0.6b-fp16` | 1.024 | 139 MB |
| `qwen3-embedding:4b-fp16` | 2.560 | 344 MB |
| `qwen3-embedding:8b-fp16` | 4.096 | 548 MB |

Toàn bộ dữ liệu (khoảng 17 GB: nguồn, corpus, evaluation, cache) được đóng gói zstd có checksum trên
Kaggle dataset riêng tư (`seed data push` / `pull`); git chỉ giữ manifest và báo cáo.

## 4. Bộ test retrieval

10.000 câu hỏi tiếng Việt sinh tất định từ corpus bằng quy tắc và template (không dùng LLM), mỗi câu gắn
section đúng và khi cần là chunk đúng.

| Nhóm | Số câu | Kiểm tra điều gì |
| --- | ---: | --- |
| `formulary` | 5.000 | Câu hỏi về chuyên luận Dược thư |
| `brand_product_qa` | 2.500 | Câu hỏi theo tên biệt dược trên tờ hướng dẫn |
| `chunk_level_retrieval` | 1.000 | Phải tìm đúng chunk trong section dài |
| `patient_natural` | 500 | Câu hỏi tự nhiên của người bệnh |
| `noisy_confuser` | 500 | Mất dấu, gõ sai, tên thuốc dễ nhầm |
| `multi_intent` | 500 | Một câu hỏi cần nhiều section |

- Độ khó: dễ 985, trung bình 8.447, khó 568.
- Ý định chính: thông tin chung 2.792, ADR 965, chống chỉ định 822, mang thai/cho con bú 779, liều dùng 738.
- Metrics: **Hit@K** (chunk trả về phải nằm trong cửa sổ 3 chunk quanh chunk đúng khi cần), **MRR**, và
  **Multi-all-hit@K** (đủ mọi section trong top K) cho 500 câu bắt buộc nhiều section.

## 5. Kết quả retrieval

Evaluation gọi đúng `RetrievalService` của backend trên Postgres và Qdrant thật. Mỗi reranker chấm
300.000 cặp query–chunk (10.000 câu × top 30) trên Kaggle GPU.

### 5.1 Kết quả tổng

| Cấu hình | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR | Multi-all-hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (Qdrant sparse) | 60,31% | 67,43% | 77,97% | 90,61% | 0,5550 | 25,00% |
| Dense `embeddinggemma:300m` | 67,36% | 75,41% | 83,80% | 90,61% | 0,5902 | 54,80% |
| Dense `bge-m3` | 81,08% | 85,96% | 90,95% | 94,55% | 0,7334 | 74,40% |
| Dense `qwen3-embedding:0.6b` | 78,28% | 84,85% | 91,88% | 97,18% | 0,7042 | 83,20% |
| Dense `qwen3-embedding:4b` | 85,18% | 90,16% | 95,22% | 98,45% | 0,7786 | **97,60%** |
| Dense `qwen3-embedding:8b` | 86,38% | 90,98% | 95,26% | 98,29% | 0,7952 | **97,60%** |
| Dense `text-embedding-3-large` (OpenAI, trả phí) | 82,63% | 89,02% | 94,85% | 98,17% | 0,7506 | 97,40% |
| Hybrid qwen3-4b + BM25, RRF k=60 | 74,57% | 82,97% | 92,44% | 98,61% | 0,6786 | 60,60% |
| Hybrid qwen3-4b + BM25, RRF k=2 | 83,70% | 90,24% | 95,67% | **99,09%** | 0,7242 | 87,00% |
| Hybrid k=2 + `bge-reranker-v2-gemma` | 84,55% | 90,19% | 95,30% | 99,09% | 0,7570 | 80,40% |
| Hybrid k=2 + `bge-reranker-v2-m3` | 86,88% | 90,91% | 95,04% | 99,09% | 0,7982 | 81,20% |
| Hybrid k=2 + `qwen3-reranker:0.6b` | 86,31% | 91,21% | 96,18% | 99,09% | 0,7823 | 87,20% |
| **Hybrid k=2 + `qwen3-reranker:4b`** | **88,51%** | **92,97%** | **96,98%** | 99,09% | **0,8060** | 77,20% |
| Hybrid k=2 + `qwen3-reranker:8b` | 56,51% | 71,47% | 89,75% | 99,09% | 0,4797 | 72,20% |

Hybrid: prefetch 50 mỗi nhánh, lấy 30 ứng viên. Reranker chỉ sắp lại top 30 nên Hit@30 không đổi.

### 5.2 Theo nhóm câu hỏi (Hit@10)

| Nhóm | BM25 | Dense qwen3-4b | Hybrid k=2 | Hybrid k=2 + qwen3-reranker-4b |
| --- | ---: | ---: | ---: | ---: |
| `formulary` | 66,46% | 96,68% | 95,84% | 97,38% |
| `brand_product_qa` | 97,68% | 96,60% | 98,48% | 99,04% |
| `chunk_level_retrieval` | 92,00% | 87,10% | 92,80% | 96,20% |
| `patient_natural` | 100,00% | 97,60% | 100,00% | 100,00% |
| `noisy_confuser` | 54,40% | 82,80% | 77,40% | 81,40% |
| `multi_intent` | 68,00% | 100,00% | 99,60% | 96,80% |

### 5.3 Tốc độ reranker (Kaggle T4, mẫu 512 cặp)

| Reranker | Cặp / giây |
| --- | ---: |
| `bge-reranker-v2-m3` | 31,2 |
| `qwen3-reranker:0.6b` | 18,5 |
| `bge-reranker-v2-gemma` | 8,2 |
| `qwen3-reranker:8b` (chia trên 2 GPU) | 1,7 |
| `qwen3-reranker:4b` | chưa đo |

### 5.4 Nhận xét

- **Model mở nhỏ đủ tốt:** `qwen3-embedding:4b` gần bằng bản 8b (Hit@10 95,22% so với 95,26%), vượt
  `text-embedding-3-large` trả phí; tự host hoàn toàn được.
- **Hybrid cho tập ứng viên tốt nhất:** Hit@30 99,09%; BM25 bù dense ở tên biệt dược và chunk cụ thể.
- **RRF k=2 hơn hẳn k=60 mặc định:** Hit@10 95,67% so với 92,44%.
- **Reranker sửa thứ hạng:** MRR 0,7242 → 0,8060, Hit@3 cao nhất (88,51%); đây là cấu hình production.
- **Đánh đổi:** rerank làm Multi-all-hit@10 giảm 87,00% → 77,20%; vòng judge → refine của agent được
  thiết kế để bù, cần đo ở bước end-to-end.
- **Nhóm khó nhất:** `noisy_confuser` (tốt nhất 82,80%).
- **Kết quả 8b không hợp lệ:** file GGUF của `qwen3-reranker:8b` được convert thành classifier, không còn
  tensor `output.weight`. Bản 8b không dùng chung ma trận embedding với lớp đầu ra, nên khi chấm bằng
  logprobs llama.cpp lấy nhầm ma trận embedding làm lớp đầu ra. Điểm vì thế bị nén trong 0,31–0,91 và MRR
  chỉ còn 0,4797. Đây là lỗi cách chạy, không phản ánh chất lượng model.

## 6. Backend agentic RAG

`pharma-agent`: FastAPI, Python 3.12, khoảng 13.500 dòng, ba tầng domain / application / infrastructure.

### 6.1 Luồng agent (LangGraph)

```text
START → guard ─(tấn công)──────► answer[blocked]
          │  ─(ngoài phạm vi)──► answer[redirect]
          ▼
       rephrase ─(chào hỏi/meta)─► answer[no_retrieval]
          ▼
       search → judge ─(đủ evidence)─► answer[grounded]
                ▲        │ (thiếu, còn ngân sách)
                └─ refine┘
                               judge ─(hết ngân sách)─► answer[partial] hoặc answer[abstain]
   node bất kỳ lỗi ─────────────────────────────────► fallback
```

| Node | Việc | Khi lỗi |
| --- | --- | --- |
| guard | Regex (prompt injection, jailbreak, lộ dữ liệu) rồi LLM phân loại | Fail-open: cho qua, ghi log |
| rephrase | Viết lại câu hỏi độc lập từ tóm tắt và 4 lượt gần nhất; nhận diện đối tượng và ý định | Dùng câu hỏi gốc |
| search | Lần đầu tự tìm bằng câu hỏi độc lập; các vòng sau dùng query của refine | Sang judge với evidence hiện có |
| judge | Structured output: `answer` hoặc `search_more` kèm phần còn thiếu | Thử lại 1 lần, rồi trả lời partial |
| refine | Sinh 1–3 query mới từ phần thiếu; chặn query trùng | Trả lời partial |
| answer | Stream câu trả lời có citation `[n]`; loại citation không hợp lệ | Fallback |

Thiết kế cho LLM rẻ:

- Mỗi bước là một lời gọi nhỏ với structured output; routing, ngân sách, chống lặp nằm trong code.
- Giới hạn mỗi lượt: tối đa 3 vòng search, 10 lời gọi LLM, 40.000 token, 24.000 ký tự evidence
  và 90 giây. Với câu hỏi thông thường, hệ thống thường dùng 1 vòng search và khoảng 4–5 lời gọi
  LLM; trường hợp phải dùng đủ 3 vòng cần khoảng 8–9 lời gọi.
- Mỗi role một model; đổi OpenAI cloud / vLLM / Ollama / llama.cpp chỉ bằng `base_url`, `api_key`, `model`.

### 6.2 Retrieval trong production

| Thành phần | Cấu hình |
| --- | --- |
| Embedding | `qwen3-embedding:4b-fp16` qua llama.cpp `/v1/embeddings`, 2.560 chiều |
| Index | Qdrant: dense + BM25 sparse (server tự tính BM25), alias `chunks_current` |
| Fusion | RRF k=2, prefetch 50, 30 ứng viên |
| Rerank | `qwen3-reranker:4b` qua llama.cpp, prompt giống lúc đánh giá, giữ top 8; lỗi thì giữ thứ tự RRF |
| Hydrate | Đọc văn bản từ Postgres theo hydrate policy, hạ cấp `full_section → chunk_window` khi vượt giới hạn |

### 6.3 Thành phần khác đã xong

| Thành phần | Nội dung |
| --- | --- |
| Corpus platform | Postgres là nguồn chính, Qdrant là index dựng lại được; import theo release (publish, rollback, gc, reindex, resume); chunk version bất biến; dùng lại embedding cache từ Kaggle |
| Memory | Tóm tắt hội thoại cuốn chiếu + các lượt gần nhất; checkpoint LangGraph trên Postgres |
| Audit retrieval | Query, cấu hình, release, hit, điểm rerank, chunk được trích dẫn |
| Observability | Mỗi lượt một trace Langfuse (node, LLM, embedding, rerank); feedback thành score |
| API | Stream AI SDK UI Message Stream, lịch sử hội thoại, xem citation, feedback, auth |
| Triển khai | `docker compose` đủ Postgres, Qdrant, llama.cpp (CPU), proxy LLM, backend; `pharma-agent ask "..." --json` |


## 7. Định hướng tiếp theo

- **Bộ test end-to-end:** lấy mẫu phân tầng từ bộ gold, bổ sung hội thoại nhiều lượt, câu không có đáp án,
  ngoài phạm vi và prompt injection.
- **Baseline RAG:** so sánh agentic RAG với RAG một bước khi dùng cùng retriever, model và context.
- **Đánh giá tự động:** dùng RAGAS/LLM-as-judge đo factual correctness, faithfulness, answer relevance và
  citation correctness.
- **Đánh giá chuyên gia thủ công:** nhờ dược sĩ chấm mù một tập con và đo độ tương quan với đánh giá tự động.
- **Ablation study:** lần lượt bỏ judge/refine, reranker, rephrase.
