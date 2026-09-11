# pharma-agent backend

Backend AI agent tra cứu thuốc trên corpus Dược thư Quốc gia. Thiết kế:
`docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md`.

## Chạy lần đầu

```bash
cd backend
uv sync
cp .env.example .env            # điền PHARMA_LLM__DEFAULT__API_KEY
```

Cần Qdrant (collection alias `thesis_chunks_qwen3_embedding_4b_fp16` do corpus-pipeline publish),
llama.cpp embedding (cổng 11434) và reranker (cổng 11435) từ `docker-compose.yml` ở repo root.
LLM, embedding và reranker đổi được sang OpenAI cloud hoặc server tự host chỉ bằng
`base_url`, `api_key`, `model` trong `.env`.

```bash
uv run pharma-agent check                      # kiểm tra kết nối
uv run pharma-agent ask "Paracetamol người lớn uống bao nhiêu?"
uv run pharma-agent ask "..." --json           # kèm trace đầy đủ
```

## Kiểm tra

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check
uv run pytest -q                 # unit
uv run pytest -q -m integration  # cần Docker (Qdrant thật)
```

## Bố cục

- `src/pharma_agent/domain`: Python thuần, không framework. Aggregate `AgentRun`, retrieval,
  skill, guardrail, prompt.
- `src/pharma_agent/application`: vòng lặp agentic RAG trên LangGraph và progress events.
- `src/pharma_agent/infrastructure`: OpenAI, Qdrant, llama.cpp, settings, composition root.
- `skills/`: SKILL.md hệ thống (`## Tìm kiếm` cho bước judge/refine, `## Trả lời` cho bước answer).

Quy tắc phụ thuộc được kiểm tra bởi `tests/architecture/test_layering.py`.
