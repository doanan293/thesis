# Đánh giá retrieval

Guide này dựng lại các số liệu retrieval của luận văn từ dữ liệu trong archive. Lệnh `uv run pharma-lab` chạy trong `pharma-lab/`, lệnh `uv run pharma-agent` chạy trong `backend/`, lệnh `docker compose` chạy ở thư mục gốc repo.

## 1. Dữ liệu

```bash
uv run pharma-lab data pull --kaggle-account acc1
```

Lệnh tải bundle `data/corpus/formulary` (embedding của 5 model), bộ gold `data/evaluation/gold`, candidates và artifact rerank của các run, cache vector query và điểm rerank, rồi kiểm sha256 từng file. `run.json`, manifest và `report.md` của các run đã có trong Git.

Metrics đọc thêm file nhãn `data/evaluation/gold/section_retrieval_eval.judgments.jsonl` (kèm `section_retrieval_eval.judgments.manifest.json` ghi sha256 của bộ gold và phiên bản luật). Với câu hỏi nhóm `formulary` và `multi_intent`, gold chỉ ghi section Dược thư; file nhãn chấp nhận thêm chunk của tờ hướng dẫn biệt dược có đúng các hoạt chất của chuyên luận đó và nằm trong mục trả lời cùng ý hỏi (chỉ định, liều dùng, chống chỉ định, tác dụng phụ, quá liều, thận trọng, dược lý, tương tác, thai kỳ và cho con bú). Hoạt chất đọc từ mục "Thành phần" của trang; thuốc phối hợp chỉ khớp chuyên luận phối hợp cùng thành phần. Mục của chunk lấy theo heading của trang, kể cả dòng tiêu đề mất dấu `#` như "- Thận trọng khi sử dụng". Đổi luật chỉ cần dựng lại file nhãn rồi chạy lại metrics; candidates, cache embedding và điểm rerank giữ nguyên:

```bash
uv run pharma-lab evaluation judgments
uv run pharma-lab metrics --run hybrid-qwen4b-p50-k30-rrf2 --top-k 30 --force
```

Report ghi `judgments_sha256` trong identity, và `pharma-lab metrics compare` chỉ so hai report dùng cùng file nhãn.

## 2. Môi trường riêng cho đánh giá

```bash
POSTGRES_PORT=5434 QDRANT_HTTP_PORT=6335 QDRANT_GRPC_PORT=6336 docker compose -p thesis-eval up -d --wait postgres qdrant
export PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis
export PHARMA_QDRANT__URL=http://localhost:6335
```

Stack `thesis-eval` tách khỏi stack của app (cổng 5433 và 6333). Không cần llama.cpp: vector chunk có sẵn trong bundle, vector query có sẵn trong cache, `pharma-lab retrieve` tắt rerank. Xong việc: `docker compose -p thesis-eval down -v`.

## 3. Chạy một model

Mỗi model đọc và ghi alias Qdrant riêng `eval_<model_slug>` qua `PHARMA_RETRIEVAL__QDRANT_COLLECTION`. `pharma-agent corpus import` tạo alias khi chưa có nhưng không chuyển một alias đã có sang collection của model khác, nên dùng chung một alias cho nhiều model sẽ truy vấn nhầm vector. Import và publish model trước khi retrieve, vì `pharma-lab retrieve` đọc release đang publish.

```bash
export PHARMA_RETRIEVAL__EMBEDDING__MODEL=qwen3-embedding:4b-fp16
export PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=2560
export PHARMA_RETRIEVAL__QDRANT_COLLECTION=eval_qwen3_embedding_4b_fp16
```

Trong `backend/`:

```bash
uv run pharma-agent corpus import ../pharma-lab/data/corpus/formulary --collection formulary --publish
```

Trong `pharma-lab/`:

```bash
uv run pharma-lab retrieve --run dense-qwen4b-k30 --retriever dense --candidate-k 30 --force
uv run pharma-lab retrieve --run bm25-qwen4b-k30 --retriever bm25 --candidate-k 30 --force
uv run pharma-lab retrieve --run hybrid-qwen4b-p50-k30-rrf60 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 60 --force
uv run pharma-lab metrics --run hybrid-qwen4b-p50-k30-rrf60 --top-k 30 --force
```

`--force` thay run tải từ archive, vì stack mới có release id khác. Mỗi lần retrieve 10.000 query mất khoảng 10–20 phút.

| Model | Chiều vector | `PHARMA_RETRIEVAL__QDRANT_COLLECTION` | Run |
| --- | ---: | --- | --- |
| `embeddinggemma:300m` | 768 | `eval_embeddinggemma_300m` | `dense-gemma300m-k30` |
| `bge-m3:567m-fp16` | 1024 | `eval_bge_m3_567m_fp16` | `dense-bge-m3-k30` |
| `qwen3-embedding:0.6b-fp16` | 1024 | `eval_qwen3_embedding_0_6b_fp16` | `dense-qwen06b-k30` |
| `qwen3-embedding:8b-fp16` | 4096 | `eval_qwen3_embedding_8b_fp16` | `dense-qwen8b-k30` |
| `qwen3-embedding:4b-fp16` | 2560 | `eval_qwen3_embedding_4b_fp16` | `dense-qwen4b-k30`, `bm25-qwen4b-k30`, `hybrid-qwen4b-p50-k30-rrf2`, `hybrid-qwen4b-p50-k30-rrf60` |

## 4. Rerank

```bash
uv run pharma-lab rerank --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --dry-run
uv run pharma-lab rerank --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16
uv run pharma-lab metrics --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

`--dry-run` in `missing_pairs=N`. Candidates của `hybrid-qwen4b-p50-k30-rrf2` trong archive là đúng các cặp query–chunk đã có điểm trong `data/cache/rerank_scores/`, nên `N` bằng 0 và rerank chỉ đọc cache, không khởi động model.

Mọi reranker được gọi qua `/v1/rerank`. Kết quả `bge-reranker-v2-gemma:f16` ở mục 7 là kết quả cuối cùng: model đã bỏ khỏi catalog, report của nó vẫn nằm ở `reports/rerank/bge_reranker_v2_gemma_f16/`.

Cấu hình `llama-reranker` cho backend trên CPU được chọn bằng benchmark đầu-cuối trên máy production (chạy trong tmux: mỗi mức nạp lại model 4B trên CPU):

```bash
uv run pharma-lab rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```

Lệnh lưu `data/cache/local_profiles/rerank/<model>.json` và in các biến `LLAMA_RERANKER_*` cho `.env` ở gốc repo. Trên Kaggle, benchmark chạy tự động trước lần chấm đầu tiên khi chưa có profile khớp (xem [CLI reference](cli-reference.md)).

Qdrant tìm dense bằng HNSW (gần đúng), nên retrieve lại trên index dựng mới có thể chọn khác vài chunk có điểm sát nhau ở cuối top 30. Lần dựng lại tháng 9/2026 lệch 4.678 trên 300.000 cặp, gần hết ở hạng 21–30. Khi đó `N` lớn hơn 0 và cần chạy reranker (CPU hoặc Kaggle) cho các cặp còn thiếu.

## 5. Đo instruction trên tập con

Qwen3-Reranker đọc instruction từ template `rerank` trong file GGUF. Template gốc dùng `<Instruct>: Given a web search query, retrieve relevant passages that answer the query`. Phép đo so bản gốc với bản chỉ đổi dòng đó thành `<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query`.

Quy tắc quyết định được chốt trước khi đo: bootstrap theo cặp trên 1.000 câu hỏi của tập con, 10.000 lần lấy mẫu lại, seed 0, cho hiệu MRR@30 (bản tiếng Việt − bản gốc). Cận dưới khoảng tin cậy 95% lớn hơn 0 thì cả ba model `qwen3-reranker` dùng instruction tiếng Việt; ngược lại giữ template gốc.

### Tập con

Dùng stack của mục 2 và biến môi trường của mục 3, rồi trong `pharma-lab/`:

```bash
uv run pharma-lab retrieve --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --sample 1000 --sample-seed 0
```

Mẫu gồm 10% mỗi `eval_group`: 500 `formulary`, 250 `leaflet`, 100 `chunk_risk`, 50 `patient_natural`, 50 `noisy_confuser`, 50 `multi_intent`, tổng 30.000 cặp.

### File GGUF tiếng Việt y tế

Chạy ở thư mục gốc repo. `gguf` 0.19.0 chỉ chạy tạm qua `uv run --no-project --with`, không phải dependency của project. `gguf-new-metadata --chat-template` thay mọi khoá `tokenizer.chat_template*`, nên bước đầu ghi danh sách JSON gồm template `default` chép từ file gốc và template `rerank` đã đổi instruction.

```bash
SIZE=0.6b
mkdir -p pharma-lab/data/work/vimed
uv run --no-project --with gguf==0.19.0 python - \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  pharma-lab/data/work/vimed/qwen3-reranker-${SIZE}-f16.chat-templates.json <<'EOF'
import json
import sys

from gguf import GGUFReader

ORIGINAL = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve '
    "relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}"
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
OLD_LINE = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
NEW_LINE = "<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query\n"

source, output = sys.argv[1], sys.argv[2]
fields = GGUFReader(source).fields
if fields["tokenizer.chat_template.rerank"].contents() != ORIGINAL:
    raise SystemExit(f"STOP: {source} does not carry the original rerank template")
templates = [
    {"name": "default", "template": fields["tokenizer.chat_template"].contents()},
    {"name": "rerank", "template": ORIGINAL.replace(OLD_LINE, NEW_LINE)},
]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(templates, handle, ensure_ascii=False)
print(f"wrote {output}")
EOF
uv run --no-project --with gguf==0.19.0 gguf-new-metadata \
  --chat-template "$(cat pharma-lab/data/work/vimed/qwen3-reranker-${SIZE}-f16.chat-templates.json)" \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16-vimed.gguf
sha256sum ai-models/gguf/qwen3-reranker-${SIZE}-f16-vimed.gguf
```

Lệnh cho kết quả tất định. So với file gốc, mọi tensor giống từng byte và chỉ khoá `tokenizer.chat_template.rerank` khác.

| File | Kích thước (byte) | sha256 |
| --- | ---: | --- |
| `qwen3-reranker-0.6b-f16-vimed.gguf` | 1.197.634.336 | `fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851` |
| `qwen3-reranker-4b-f16-vimed.gguf` | 8.049.922.912 | `4b428e981efc9a209c674a0af9f1ec527b5570896e0bcc431382c16f1e839c7a` |
| `qwen3-reranker-8b-f16-vimed.gguf` | 15.141.207.776 | `c6516333e32d4f1d8aad8325d5d5bdb8165f9778e892eed4ca8b0eaf3339a810` |

Instruction tiếng Việt thắng (mục Kết quả), nên `qwen3-reranker:0.6b-fp16`, `qwen3-reranker:4b-fp16` và `qwen3-reranker:8b-fp16` trỏ tới ba file trên (lệnh trên với `SIZE=4b` và `SIZE=8b`); model thử nghiệm `qwen3-reranker:0.6b-fp16-vimed` đã bỏ. File gốc đã xoá; khi cần dựng lại, tải từ nguồn sau rồi kiểm sha256 trước khi chạy lệnh:

| File gốc | Nguồn | sha256 |
| --- | --- | --- |
| `qwen3-reranker-0.6b-f16.gguf` | `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp` | `fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa` |
| `qwen3-reranker-4b-f16.gguf` | `Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp` | `c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117` |
| `qwen3-reranker-8b-f16.gguf` | `sinjab/Qwen3-Reranker-8B-F16-GGUF` | `a53322f7936010458424a12f0f6d22291547e42fa85c16dd4730244d659cea96` |

### Chấm và so sánh

```bash
uv run pharma-lab rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend kaggle --model qwen3-reranker:0.6b-fp16 --kaggle-account auto
uv run pharma-lab rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend kaggle --model qwen3-reranker:0.6b-fp16-vimed --kaggle-account auto
uv run pharma-lab metrics --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --top-k 30
uv run pharma-lab metrics compare --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --baseline qwen3-reranker:0.6b-fp16 --candidate qwen3-reranker:0.6b-fp16-vimed --metric mrr --top-k 30 --resamples 10000 --seed 0
```

### Kết quả

| Biến thể | Hit@10 | MRR@30 |
| --- | ---: | ---: |
| Template gốc (`qwen3-reranker:0.6b-fp16`) | 95,60% | 0,7633 |
| Tiếng Việt y tế (`qwen3-reranker:0.6b-fp16-vimed`) | 96,20% | 0,8078 |

Hiệu MRR@30 (tiếng Việt − gốc) là 0,0445, khoảng tin cậy 95% [0,0306; 0,0585] (1.000 câu, 10.000 lần lấy mẫu lại, seed 0). Quyết định: dùng instruction tiếng Việt cho cả ba model.

## 6. Run `dense-text-embedding-3-large-k30`

Run này dùng embedding API trả phí nên không chạy lại. Candidates được nhập từ lần chạy gốc (`origin: imported` trong `run.json`) và chỉ tính lại metrics.

## 7. Kết quả tham chiếu

| Run / biến thể | Hit@10 | MRR |
| --- | ---: | ---: |
| `bm25-qwen4b-k30` | 81,33% | 0,5755 |
| `dense-gemma300m-k30` | 84,00% | 0,5910 |
| `dense-bge-m3-k30` | 91,62% | 0,7394 |
| `dense-qwen06b-k30` | 92,17% | 0,7075 |
| `dense-qwen4b-k30` | 95,39% | 0,7813 |
| `dense-qwen8b-k30` | 95,36% | 0,7997 |
| `dense-text-embedding-3-large-k30` | 94,85% | 0,7506 |
| `hybrid-qwen4b-p50-k30-rrf60` | 93,74% | 0,6976 |
| `hybrid-qwen4b-p50-k30-rrf2` | 95,89% | 0,7331 |
| rrf2 + `bge-reranker-v2-gemma:f16` (logprob, không chạy lại) | 95,30% | 0,7570 |
| rrf2 + `bge-reranker-v2-m3:f16` | 95,81% | 0,8065 |
| rrf2 + `qwen3-reranker:0.6b-fp16` | 96,75% | 0,7905 |
| rrf2 + `qwen3-reranker:4b-fp16` | 97,52% | 0,8280 |
| rrf2 + `qwen3-reranker:8b-fp16` (bản logprob cũ; đang chấm lại qua `/v1/rerank`) | 89,75% | 0,4797 |

Số trước khi chuyển layout và bảng so sánh nằm trong commit "rebuild the evaluation runs with the new CLI". Lần dựng lại chấp nhận Hit@10 thấp hơn tối đa 1 điểm phần trăm và MRR thấp hơn tối đa 0,01.
