# Local + Kaggle workflow

Build, import, retrieve và metrics chạy ở local; Kaggle chạy embedding (bundle và query) và reranker.

## 1. Kiểm tra môi trường

Khai báo profile Kaggle trong `seed-pipeline/.env`:

```dotenv
KAGGLE_ACCOUNT_DEFAULT=acc1
KAGGLE_SHARED_OWNER=account_goc
KAGGLE_ACC1_USERNAME=account_goc
KAGGLE_ACC1_API_TOKEN=token_acc1
KAGGLE_ACC2_USERNAME=account_phu_2
KAGGLE_ACC2_API_TOKEN=token_acc2
```

```bash
uv sync --directory ..    # ở repo root: một .venv chung cho uv workspace
uv run seed doctor --backend kaggle --kaggle-account acc1
docker compose up -d postgres qdrant llama-embedding   # tìm compose.yaml ở thư mục cha
```

Production stage ở profile mode tự tìm checkpoint tương thích trong mọi profile `accN` và mirror sang owner đích trước khi submit kernel.

## 2. Build, validate và export bundle

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/corpus/formulary --force
```

## 3. Embed bundle trên Kaggle

```bash
uv run seed bundle embed --bundle data/corpus/formulary --backend kaggle \
  --model qwen3-embedding:4b-fp16 --kaggle-account acc1 --dry-run
uv run seed bundle embed --bundle data/corpus/formulary --backend kaggle \
  --model qwen3-embedding:4b-fp16 --kaggle-account acc1
```

Lệnh stream log kernel, tải artifact về và merge vào `data/cache/text_embeddings/qwen3_embedding_4b_fp16.jsonl`. Exit code 3 nghĩa là kernel chưa xong hoặc hết budget: chạy lại đúng lệnh để resume từ checkpoint. Khi cache đủ, lệnh ghi `embeddings/qwen3_embedding_4b_fp16.jsonl` vào bundle. `Ctrl-C` chỉ dừng theo dõi; kernel vẫn chạy và lần chạy sau tự attach.

## 4. Import vào backend

```bash
cd ../backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish
uv run pharma-agent corpus releases --collection formulary
cd ../seed-pipeline
```

## 5. Evaluation dataset và retrieval

```bash
uv run seed evaluation build --bundle data/corpus/formulary
uv run seed embed queries --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
uv run seed retrieve --run backend-bm25-k30 --retriever bm25 --candidate-k 30
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid \
  --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

Thêm `--limit 50` và tên run `...-smoke50` để chạy thử trước.

## 6. Rerank trên Kaggle và metrics

Một model có thể cần nhiều phiên GPU 6 giờ, nên mỗi model chạy trong một cửa sổ tmux riêng; tắt terminal hay mất kết nối không làm dừng lệnh:

```bash
tmux new-session -s rerank -n qwen3-reranker-4b
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account auto --dry-run
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account auto
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Với `--kaggle-account auto`, trước mỗi phiên lệnh:

- chạy `kaggle quota -v` bằng thông tin của từng profile `accN` trong `.env` và đọc dòng `GPU`;
- bỏ tài khoản đang bị lệnh khác khoá (`data/work/locks/kaggle-accounts/<accN>.lock`), chọn tài khoản còn nhiều giờ GPU nhất, bằng nhau thì chọn `accN` nhỏ hơn;
- đặt ngân sách phiên = min(`--budget-seconds`, giờ còn lại − 0,5 giờ) và chỉ nộp kernel khi ngân sách ít nhất 1 giờ.

Sau mỗi phiên, điểm (kể cả khi kernel dừng giữa chừng) được gộp vào `data/cache/rerank_scores/<model>.jsonl` trước, rồi mới publish checkpoint. Trước phiên kế tiếp, nguồn có nhiều cặp nhất trong cache ở máy và checkpoint của mọi tài khoản được publish làm checkpoint của tài khoản sắp chạy, nên đổi tài khoản không mất điểm. Lệnh chạy tới khi đủ cặp hoặc hết quota; `--max-runs N` giới hạn số phiên. `--kaggle-account accN` chỉ dùng một tài khoản, với cùng quy tắc quota. Dataset model và input chưa có luôn được tạo bằng profile của `KAGGLE_SHARED_OWNER` (acc1), vì vậy `.env` phải có profile đó.

Nhiều model chạy song song, mỗi model một cửa sổ tmux (`tmux new-window -n qwen3-reranker-8b`); khoá tài khoản bảo đảm mỗi tài khoản chỉ chạy một phiên GPU.

Khi không tài khoản nào còn đủ 1 giờ, lệnh dừng với exit code 3, `stop=quota-exhausted`, và in bảng quota:

```text
account  username                  remaining    total  refresh_at                locked
acc1     account_goc                   0.40h   30.00h  2026-09-19T00:00:00       no
acc2     account_phu_2                 0.20h   30.00h  2026-09-19T00:00:00       yes
```

`refresh_at` là lúc Kaggle làm mới quota của tài khoản; `locked=yes` là tài khoản đang chạy phiên của lệnh khác. Chạy lại đúng lệnh sau thời điểm đó để chấm tiếp.

Mỗi lệnh `seed rerank` ghi thêm vào `data/work/logs/rerank/<model>.log` (ví dụ `qwen3_reranker_4b_fp16.log`), có timestamp: tài khoản được chọn, quota trước và sau phiên, kernel, số cặp, lỗi và lý do dừng. Log nằm ngoài `/tmp` nên còn sau khi máy khởi động lại. Theo dõi từ cửa sổ khác:

```bash
tail -f data/work/logs/rerank/qwen3_reranker_4b_fp16.log
```

Kaggle không truy cập Postgres/Qdrant và không tính metrics; rerank chỉ nhận candidate JSONL cùng manifest.

## Troubleshooting

- Credential/owner lỗi: sửa `.env`, chạy lại `seed doctor --backend kaggle`.
- `checksum mismatch` hoặc `no mounted file found`: dataset input chưa `READY` hoặc lệch; chạy lại để reconciler publish đúng version.
- `Expected hits from exactly one published release`: chưa import/publish release, hoặc `retrieval.collections` trỏ sai collection.
- `pinned to release`: release hiện hành đổi giữa chừng; dùng tên run mới.
- `Query embedding cache ... is missing`: chạy lại `seed embed queries --backend kaggle` với đúng file evaluation và model embedding của backend.
- `--top-k` lớn hơn `candidate-k`: retrieve lại với candidate depth đủ lớn.
- `stop=quota-exhausted`: xem bảng quota và chạy lại sau `refresh_at`.
- `stop=no-progress`: phiên vừa rồi không thêm cặp nào (kernel lỗi không có output hoặc llama-server không lên sau 3 lần khởi động lại); đọc `data/work/logs/rerank/<model>.log` và `server-*.log` trong output kernel.
- Máy hoặc WSL khởi động lại khi kernel đang chạy: mở lại tmux và chạy lại đúng lệnh; lệnh nối vào kernel đang chạy trên tài khoản đó thay vì nộp kernel mới.
- Kernel đã kết thúc nhưng điểm chưa về máy và lệnh không còn nối được vào nó (ví dụ cấu hình runtime đã đổi nên job identity khác): `uv run seed rerank --run RUN --backend kaggle --model MODEL --recover-kernel OWNER/SLUG`, rồi chạy lại lệnh rerank thường để chấm phần còn thiếu.

Chi tiết lệnh: [CLI reference](cli-reference.md). Chính sách bàn giao: [Downstream](downstream.md).
