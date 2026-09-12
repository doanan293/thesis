# Thiết kế backend AI agent dược (pharma-agent)

Ngày: 2026-09-11. Trạng thái: đã duyệt qua brainstorming, chờ implementation plan.

## 1. Mục tiêu

Xây backend cho hệ thống AI agent hỏi đáp về thuốc dựa trên corpus Dược thư Quốc gia
Việt Nam và snapshot An Khang mà `corpus-pipeline` đã tạo. Backend là **một service
FastAPI duy nhất** trong `backend/`, gồm: agent workflow, retrieval trên Qdrant, hội
thoại và memory trên Postgres, skills, auth, streaming SSE.

Bài học lấy từ DocMind (`sun-internal/doc-mind/services/agent`): budget có
reservation, gate "không trả lời khi chưa có evidence", nén evidence cũ, skill
resolution metadata-first, progress phases là public contract, citation sanitizer
khi stream. Những gì không lặp lại: framework lọt vào domain, domain anemic, control
flow bằng string, state graph trùng lặp, guardrail fail-closed, tracing tự viết,
LiteLLM.

## 2. Ràng buộc đã chốt

| Chủ đề | Quyết định |
| --- | --- |
| Người dùng | Cả người dân và dược sĩ/nhân viên y tế; agent tự nhận diện `audience` và đổi giọng |
| Giọng trả lời | Trả lời thẳng, chính xác nhất có thể, **không** disclaimer "không thay thế bác sĩ" |
| Orchestration | LangGraph 1.2 (ưu tiên thư viện có cộng đồng, không tự viết state machine) |
| LLM | OpenAI SDK gọi thẳng, **Chat Completions API** (không Responses) để đổi được sang server self-host; mỗi role một endpoint |
| Model mặc định | `gpt-5-nano` cho guard/rephrase/chọn skill/tóm tắt; `gpt-5-mini` cho judge/refine/answer. Ràng buộc: chỉ dùng model rẻ, mọi bước LLM phải chạy ổn trên model yếu |
| Embedding | `qwen3-embedding:4b-fp16` qua llama.cpp `/v1/embeddings`, dimension 2560, phải khớp collection |
| Retrieval | Qdrant hybrid dense + BM25 sparse, RRF, rerank `qwen3-reranker:4b-fp16` (protocol completion logprobs), cấu hình được |
| Skills | Skill hệ thống trong repo + người dùng upload SKILL.md |
| Guardrail | Regex + LLM nhỏ, fail-open |
| Memory | Rolling summary + N lượt gần nhất |
| API | SSE + endpoint non-stream |
| Auth | fastapi-users: email/password + Google OAuth, JWT bearer |
| Observability | Langfuse |
| Vòng lặp agent | **Agentic RAG** dạng retrieve → judge → refine với structured output, không function calling. Gọi đúng tên trong thesis là agentic RAG workflow, không phải ReAct |

## 3. Stack

Python 3.12, uv, FastAPI, LangGraph 1.2 + langgraph-checkpoint-postgres, openai 3.x,
qdrant-client 1.19 (extra `fastembed` cho BM25 sparse), SQLAlchemy 2 async + asyncpg +
Alembic, fastapi-users (extra `sqlalchemy`, `oauth`), httpx, pydantic-settings,
sse-starlette, langfuse 4.x, pytest + pytest-asyncio + respx + testcontainers, ruff,
pyrefly. Chuẩn lint/format giống `corpus-pipeline`.

## 4. Phân tầng và bố cục

Quy tắc phụ thuộc: `api → application → domain`; `infrastructure → domain`;
`main.py` là composition root duy nhất biết adapter cụ thể. Domain không import
langgraph, openai, qdrant_client, sqlalchemy, fastapi, langfuse. Có test AST cưỡng
chế quy tắc này.

LangGraph nằm trong `application/` vì graph là điều phối use case. Node là hàm mỏng:
đọc `AgentRun` từ state, gọi domain, ghi lại state. State graph là pydantic model chứa
object của domain, không có `BaseMessage`. Prompt là hàm thuần trong `domain/`.

```text
backend/
  pyproject.toml
  alembic/                          # migrations
  skills/                           # SKILL.md hệ thống, versioned trong repo
    drug-monograph/SKILL.md
    drug-interaction/SKILL.md
    dosing-by-population/SKILL.md
    brand-to-generic/SKILL.md
    plain-language/SKILL.md
  src/pharma_agent/
    domain/
      shared/        # ids, Clock port, DomainError base, enums dùng chung
      llm/           # ChatMessage, LlmUsage, LlmRole, port LlmPort
      guardrail/     # Verdict, regex groups, prompts, port GuardrailClassifier
      retrieval/     # Query, Hit, Chunk, Evidence, EvidenceSet, RetrievalService,
                     # ports Retriever/Reranker/Hydrator/RetrievalAuditRepository
      agent/         # AgentRun aggregate, BudgetLimits/BudgetUsage, ActionLog,
                     # AnswerPlan, RunStatus, prompts (rephrase/judge/refine/answer),
                     # structured output schemas
      conversation/  # Conversation aggregate, Message, Citation, port ConversationRepository
      skill/         # Skill aggregate, SKILL.md parser, SkillResolver, port SkillRepository
    application/
      chat/          # ChatTurnUseCase, graph builder, nodes, state model, progress emitter
      conversation/  # list/get/rename/delete
      skill/         # upload/list/enable/disable/delete, sync skill hệ thống
      memory/        # SummarizeConversationUseCase (chạy nền)
      progress.py    # public phase/event contract
    infrastructure/
      llm/openai/            # OpenAiLlmAdapter (chat.completions.parse, stream), bọc langfuse.openai
      retrieval/qdrant/      # QdrantHybridRetriever, QdrantHydrator, embedder qua OpenAI SDK
      retrieval/llama_cpp/   # LlamaCppCompletionReranker, NativeRerankClient
      persistence/postgres/  # SQLAlchemy models, repositories, UnitOfWork
      langgraph/             # AsyncPostgresSaver setup, cleanup checkpoint
      auth/                  # fastapi-users: User model, UserManager, JWT, Google OAuth
      observability/         # Langfuse client, logging JSON
      settings.py            # pydantic-settings
    api/
      routers/  chat.py conversations.py skills.py auth.py health.py
      schemas/  sse.py deps.py
    main.py                  # lifespan + composition root
  tests/
    architecture/ domain/ application/ infrastructure/ api/
```

## 5. Domain model

### 5.1 `AgentRun` (aggregate root, một lượt hỏi đáp)

Trường: `run_id`, `conversation_id`, `user_id`, `original_query`, `standalone_query`,
`audience` (`general_public | professional | unknown`), `language` (`vi | en | other`),
`intent` (`pharma_question | smalltalk | meta`), `limits: BudgetLimits`,
`usage: BudgetUsage`, `evidence: EvidenceSet`, `actions: ActionLog`,
`skills: tuple[SelectedSkill, ...]`, `plan: AnswerPlan | None`, `status: RunStatus`,
`error_code: ErrorCode | None`, `guard_verdict: Verdict | None`, `started_at`.

Method và invariant:

- `allowed_actions() -> frozenset[Action]`: máy trạng thái của vòng lặp.
  - Sau `rephrase` với intent `smalltalk|meta`: chỉ `ANSWER` (mode `no_retrieval`).
  - Sau `rephrase` với `pharma_question`: chỉ `SEARCH`.
  - Sau `SEARCH`: chỉ `JUDGE`.
  - Sau `JUDGE = answer`: chỉ `ANSWER`.
  - Sau `JUDGE = search_more`: `REFINE` nếu `search_rounds < max_search_rounds`
    và còn ngân sách call, ngược lại `ANSWER` (status `partial` nếu có evidence,
    `abstained` nếu không).
  - Sau `REFINE`: `SEARCH` với query mới; nếu mọi query mới trùng query đã dùng thì
    `ANSWER` (partial/abstained).
- `record_search(queries, hits, error=None)`: `EvidenceSet.merge`, đánh dấu evidence
  cũ `superseded`, tăng `search_rounds`, ghi action. Search lỗi ghi action kèm lỗi,
  không raise.
- `record_judge(decision)`, `record_refine(queries)`: kiểm tra query trùng
  (chuẩn hóa lower/strip), ghi action.
- `charge(usage: LlmUsage)`: cộng call và token; vượt `max_llm_calls` hay
  `max_tokens` thì raise `BudgetExhausted`.
- `can_afford(step: OptionalStep)`: reservation. `REPHRASE` và `RESOLVE_SKILLS` chỉ
  được chạy nếu `llm_calls + 3 <= max_llm_calls` (chừa judge + answer + 1 dự phòng).
- `submit_plan(AnswerPlan)`: nếu `mode == grounded` mà không có evidence không
  superseded thì raise `EvidenceRequired`.
- `abstain(reason)`, `block(verdict)`, `redirect(verdict)`, `fail(ErrorCode)`,
  `timeout()`: chuyển status, chỉ cho phép từ trạng thái chưa kết thúc.
- `to_trace() -> dict`: dữ liệu lưu kèm message (actions, usage, skills, status).

### 5.2 Value objects của agent

- `BudgetLimits(max_search_rounds=3, max_llm_calls=10, max_tokens=40_000, max_evidence_chars=24_000, deadline_seconds=90)`.
- `BudgetUsage(llm_calls, prompt_tokens, completion_tokens, search_rounds)` bất biến,
  `charge` trả bản mới.
- `ActionLog`: tuple `Action(kind, at, reason, outcome, payload)` append-only.
- `AnswerPlan(mode, partial: bool)` với `mode ∈ {grounded, no_retrieval, abstain, blocked, redirect}`;
  `partial = True` khi answer bị ép vì hết budget, judge lỗi hoặc refine trùng.
- `RunStatus`: `running, completed, partial, abstained, blocked, redirected, error, timeout`.
- `ErrorCode` enum: `GUARDRAIL_LLM_FAILED, REPHRASE_FAILED, SKILL_RESOLUTION_FAILED,
  SEARCH_FAILED, RERANK_FAILED, JUDGE_FAILED, REFINE_FAILED, ANSWER_FAILED,
  BUDGET_EXHAUSTED, DEADLINE_EXCEEDED, PERSIST_FAILED, EVIDENCE_REQUIRED`.

### 5.3 Retrieval

- `Query(text, origin: initial | refined)`.
- `Hit(chunk_id, section_id, chunk_index, hydrate_strategy, source, title, section,
  start_page, end_page, context_header, chunk_text, embedding_text, content_type,
  table_id, colloquial_mapping, term_annotations, fusion_score, rerank_score,
  matched_queries)`: ánh xạ 1-1 payload Qdrant theo `qdrant_payload_contract`.
- `Chunk(chunk_id, section_id, chunk_index, text, content_type, table_id)` cho hydrate.
- `Evidence(ref, hit, chunks: tuple[Chunk, ...], applied_strategy, superseded)`; `ref`
  là `E1..En` ổn định trong một run.
- `EvidenceSet`: `merge(hits)` gộp theo `chunk_id` giữ `rerank_score` cao hơn và hợp
  `matched_queries`; `supersede_all()`; `active()`; `pack(max_chars)` trả evidence
  theo thứ tự rerank, hạ cấp strategy `full_section → chunk_window → search_only`
  khi không đủ chỗ thay vì cắt giữa chừng; `summary_view()` và `context_view()`.
- `RetrievalService` (domain service thuần, chạy trên port): `search(queries) ->
  SearchResult`: gọi `Retriever.search_many`, dedupe, `Reranker.rerank(query gốc,
  hits, top_n)`, `Hydrator.hydrate(hits)` theo strategy, trả hits đã đầy đủ.
  Rerank lỗi thì giữ thứ tự RRF và ghi `RERANK_FAILED` vào kết quả (không fail cả search).
- Port: `Retriever.search_many(queries, top_k) -> list[list[Hit]]`,
  `Reranker.rerank(query, hits, top_n) -> list[Hit]`,
  `Hydrator.hydrate(hit, strategy) -> list[Chunk]`,
  `RetrievalAuditRepository.record(run_id, message_id, queries, config, hits, cited_ids)`.

### 5.4 Conversation

- `Conversation(conversation_id, user_id, title, summary, summary_covers_message_id,
  created_at, updated_at)`; `messages` nạp riêng có phân trang.
- `Message(message_id, conversation_id, role, content, status, citations,
  phases, usage, run_id, created_at)`.
- `Citation(index, chunk_id, section_id, title, section, start_page, end_page, table_id)`.
- `context_for_rephrase(summary, recent_turns, max_chars)`: hàm thuần, loại lượt có
  status `blocked | error | timeout`, cắt theo ký tự từ cũ nhất.
- Port `ConversationRepository`: `create`, `get_for_user`, `list_for_user`,
  `rename`, `delete`, `append_turn(user_msg, assistant_msg)` (một transaction),
  `recent_turns(conversation_id, n)`, `update_summary`.

### 5.5 Skill

Skill tuân theo **Agent Skills specification** (https://agentskills.io/specification).
Việc parse và validate dùng thư viện tham chiếu `skills-ref` (CLI `agentskills validate`).
Backend chặt hơn một điểm: `name` chỉ gồm a-z, 0-9 và gạch nối (không nhận chữ Unicode),
để skill dùng được ở mọi client Agent Skills. Mọi thứ backend nhận, `agentskills validate`
cũng nhận.

- Frontmatter: `name` bắt buộc (1-64 ký tự, chữ thường, số và gạch nối, không bắt đầu,
  kết thúc hay lặp gạch nối, **trùng tên thư mục**), `description` bắt buộc (1-1024 ký
  tự, nói skill làm gì và khi nào dùng). Các trường tùy chọn `license`, `compatibility`
  (tối đa 500 ký tự), `metadata` (map chuỗi-chuỗi), `allowed-tools` được chấp nhận.
- Body markdown là hướng dẫn tự do, không có mục bắt buộc. Tiêu đề hiển thị lấy từ
  heading `#` đầu tiên, không có thì dùng `name`.
- `Skill(name, owner_user_id | None, description, instructions, content, version,
  enabled)`; `content` là nguyên văn SKILL.md, `version` là sha256 rút gọn của nó.
- `name` là định danh: duy nhất trong phạm vi một owner (skill hệ thống có owner None).
  Người dùng không được đặt trùng tên skill hệ thống.
- `resolve_selected(catalog_metadata, selected_names, max_selected=3)`: hàm thuần loại
  name không có trong catalog, giữ thứ tự, cắt ở 3.
- Port `SkillCatalog`: `list_catalog(user_id, limit)` (skill hệ thống bật, rồi skill bật
  của user, trùng tên thì chỉ giữ skill hệ thống), `get_by_names(user_id, names)`.
  `SkillRepository` thêm `replace_system`, `create`, `list_system`, `list_for_user`,
  `set_enabled`, `delete_owned`.

### 5.6 Guardrail

- `Verdict(passed, in_scope, label, source: regex | llm | skipped, reason)`.
- Regex groups thuần: `PromptInjection`, `Jailbreak`, `DataLeak`, `Bypass`, mỗi
  group có `LABEL`, pattern đã compile, `check(text_lower) -> tuple[str, ...] | None`.
- Port `GuardrailClassifier.classify(query) -> LlmGuardVerdict(is_attack, in_scope, reason)`.
- `in_scope` = câu hỏi liên quan thuốc, sức khỏe, y tế, hoặc chào hỏi/meta về
  trợ lý. Khi không chắc thì `in_scope = true`.

### 5.7 LLM port

- `ChatMessage(role: system | user | assistant, content)`, `LlmUsage(prompt_tokens, completion_tokens)`.
- `LlmRole`: `guardrail, rephrase, skill_selector, judge, refine, answer, summarizer`.
- `LlmPort.structured(role, messages, schema: type[T]) -> tuple[T, LlmUsage]`.
- `LlmPort.stream(role, messages) -> AsyncIterator[str | LlmUsage]` (yield text, cuối
  cùng yield usage).
- Schema structured output (pydantic, `extra=forbid`, enum ngắn):
  - `LlmGuardVerdict(is_attack: bool, in_scope: bool, reason: str)`
  - `RephraseResult(standalone_query: str, audience, language, intent)`
  - `SkillSelection(skill_names: list[str])`
  - `JudgeDecision(decision: answer | search_more, gaps: list[str], reason: str)`
  - `RefineResult(queries: list[str])` (1-3 phần tử, validator cắt về 3)
  - `ConversationSummary(summary: str)`

## 6. Workflow

### 6.1 Graph

```text
START → guard ─(attack)───────────► answer[blocked] ──► END
          │  ─(out_of_scope)───────► answer[redirect] ─► END
          ▼ (pass)
       rephrase ─(smalltalk|meta)──► answer[no_retrieval] ─► END
          │ (pharma_question)
          ▼
   resolve_skills → search → judge ─(answer)──────► answer[grounded] ─► END
                      ▲        │ (search_more, còn budget)
                      │        ▼
                      └───── refine
                               judge ─(hết budget)─► answer[grounded, partial] hoặc answer[abstain]
   bất kỳ node nào raise ──────────────────────────► fallback ─► END
```

Một `AgentRun` cho mỗi lượt. State graph `ChatTurnState(run: AgentRun,
context: ConversationContext, pending_queries: list[Query], last_judge:
JudgeDecision | None)`. Checkpointer `AsyncPostgresSaver`, `thread_id = run_id`,
graph compile một lần lúc khởi động; callback progress và token truyền qua
`RunnableConfig["configurable"]`, không đóng gói trong node.

Routing là hàm thuần đọc `run.allowed_actions()` và `run.status`, không kiểm tra
"key có trong dict".

### 6.2 Node

| Node | LLM role | Việc | Fail policy |
| --- | --- | --- | --- |
| guard | guardrail (nano) | regex trước; hit → `block`. LLM: `is_attack` → `block`; `not in_scope` → `redirect` | LLM lỗi → pass, ghi `GUARDRAIL_LLM_FAILED` vào action log |
| rephrase | rephrase (nano) | input: summary + tối đa 4 lượt gần nhất (≤ 4.000 ký tự) + câu hỏi. Output `RephraseResult` | lỗi hoặc không đủ budget → `standalone_query = original`, audience `unknown`, intent `pharma_question` |
| resolve_skills | skill_selector (nano) | catalog metadata (id + description) → `SkillSelection` → `SkillResolver` → nạp body | lỗi, catalog rỗng, catalog > 30, không đủ budget → không skill |
| search | không | lần đầu: `[standalone_query]`; các lần sau: `pending_queries` từ refine. Gọi `RetrievalService.search`, `run.record_search`, ghi audit | lỗi → ghi action lỗi, sang judge với evidence hiện có |
| judge | judge (mini) | input: câu hỏi, audience, body hướng dẫn của skill đã chọn, `EvidenceSet.summary_view()` (E1..En, title/section/trang, snippet ≤ 300 ký tự, term hints). Output `JudgeDecision` | retry 1; lỗi → `record_judge(answer)` (partial) |
| refine | refine (mini) | input: câu hỏi, `gaps`, query đã dùng, term hints (`colloquial_mapping.aliases/product_names`, `term_annotations.vi/en`). Output `RefineResult` | lỗi hoặc toàn query trùng → `ANSWER` partial |
| answer | answer (mini, stream) | prompt theo mode (xem 6.3); citation `[n]`; sanitizer; `run.submit_plan` trước khi stream | lỗi giữa stream → fallback |
| fallback | không | text deterministic theo `error_code`/`timeout`, status `error` hoặc `timeout` | không thể lỗi |

Cả graph bọc `asyncio.timeout(limits.deadline_seconds)`; timeout → `run.timeout()`
→ fallback. Mọi LLM call đi qua `run.charge(usage)`.

### 6.3 Answer node

Prompt system gồm: vai trò (trợ lý dược, trả lời bằng ngôn ngữ của người hỏi), quy
tắc theo `audience` (người dân: ngắn, dễ hiểu, giải thích thuật ngữ; chuyên môn:
đầy đủ, đúng thuật ngữ, ghi liều/đơn vị chính xác), quy tắc citation (chỉ trích dẫn
`[n]` có trong danh sách, mọi khẳng định về liều/chống chỉ định/tương tác phải có
`[n]`), quy tắc không có evidence (nói rõ không tìm thấy trong Dược thư, gợi ý
cách hỏi lại, không đoán), body hướng dẫn của skill đã chọn. **Không có** disclaimer y tế.

Mode:

- `grounded`: có `context_view()` (mỗi nguồn `[n]` + header title/section/trang +
  text đã hydrate; bảng đứng đầu section nếu hit có `table_id`).
- `grounded` với status `partial`: thêm dòng hệ thống "evidence có thể chưa đủ, nêu
  rõ phần chưa tìm thấy".
- `no_retrieval`: chào hỏi/meta, không evidence, không citation.
- `abstain`: không evidence; giải thích không tìm thấy, gợi ý hỏi lại.
- `blocked`: từ chối ngắn, không nhắc chi tiết kỹ thuật của guardrail.
- `redirect`: nói rõ trợ lý chỉ hỗ trợ về thuốc và sức khỏe.

Citation sanitizer: state machine trên delta stream; giữ lại buffer khi gặp `[`
chưa đóng; chỉ cho qua `[n]` với `n` hợp lệ, xóa `[n]` không hợp lệ; cuối stream
xả buffer. Citation object tạo từ evidence `n` đã dùng, lưu cùng message.

### 6.4 Chi phí mỗi lượt

guard 1 + rephrase 1 + skills 1 (nano) + judge 1-3 + refine 0-2 + answer 1 (mini).
Tổng 5-9 call, input đa số ngắn (< 3.000 token trừ answer ≈ 8.000 token). Ước tính
dưới 0,01 USD mỗi lượt với giá hiện tại của `gpt-5-nano` và `gpt-5-mini`.

## 7. Retrieval

### 7.1 Cấu hình mặc định (theo run tốt nhất `hybrid-qwen4b-p50-k30-rrf2` + rerank qwen3-4b)

| Setting | Mặc định |
| --- | --- |
| `retrieval.collection_alias` | `thesis_chunks_qwen3_embedding_4b_fp16` |
| `retrieval.embedding.model` / `dimension` | `qwen3-embedding:4b-fp16` / 2560 |
| `retrieval.embedding.base_url` | llama.cpp embedding server (`/v1/embeddings`, gọi bằng OpenAI SDK, không prefix) |
| `retrieval.mode` | `hybrid` (`dense` cũng hỗ trợ) |
| `retrieval.prefetch_k` / `rrf_k` / `candidate_k` | 50 / 2 / 30 |
| `retrieval.rerank.protocol` | `completion_logprobs` (`native_rerank`, `none` cũng hỗ trợ) |
| `retrieval.rerank.model` | `qwen3-reranker:4b-fp16` |
| `retrieval.rerank.top_n` | 8 |
| `retrieval.hydrate.window` | 1 chunk mỗi bên cho `chunk_window` |
| `retrieval.max_concurrent_searches` | 3 |

Khởi động: adapter Qdrant đọc `get_collection(alias)` và từ chối chạy nếu vector
size khác `dimension` cấu hình. Tên vector: dense `dense_vector` theo
`DENSE_VECTOR_NAME` của pipeline, sparse `bm25_sparse_vector`, BM25 qua
`models.Document(text, model="Qdrant/bm25")` giống pipeline.

### 7.2 Luồng search

1. Embed batch mọi query một lần.
2. Mỗi query: `query_points` với `Prefetch(dense, limit=prefetch_k)` +
   `Prefetch(Document bm25, limit=prefetch_k)`, `RrfQuery(k=rrf_k)`, `limit=candidate_k`,
   `with_payload=True`. Semaphore `max_concurrent_searches`.
3. Dedupe theo `chunk_id`.
4. Rerank theo `standalone_query` trên `embedding_text`, protocol:
   - `completion_logprobs`: prompt `qwen3_yes_no_v1` (system "Judge whether the
     Document meets the requirements...", `<Instruct>` = "Given a Vietnamese medical
     retrieval query, retrieve relevant passages that answer the query", `<Query>`,
     `<Document>`, assistant `<think>\n\n</think>\n\n`), gọi llama.cpp `/completion`
     với `n_predict=1, n_probs=2, temperature=1.0, samplers=["temperature"],
     post_sampling_probs=true, logit_bias +100 cho token "yes"/"no"`, điểm =
     p(yes) / (p(yes) + p(no)). Template và scoring copy nguyên từ
     `corpus_pipeline.runtime.model_profiles`, có test so sánh chuỗi prompt để không
     lệch nhau.
   - `native_rerank`: POST `/v1/rerank` `{model, query, documents}` → `results[].relevance_score`.
   - `none`: giữ thứ tự RRF, `rerank_score = None`.
5. Giữ `top_n`, hydrate theo `hydrate_strategy` (scroll filter `section_id`, sort
   `chunk_index`), pack theo `max_evidence_chars`.

### 7.3 Audit

Mỗi lần search ghi `retrieval_runs(run_id, conversation_id, message_id, corpus_version =
collection thật sau alias, retriever_config, query_text, query_embedding_model)` và
`retrieval_hits(rank, chunk_id, section_id, table_id, qdrant_score, rerank_score,
hydrate_strategy, cited, snippet)`. `cited` cập nhật sau khi answer xong.

## 8. Persistence, memory, skills, auth, API

### 8.1 Bảng Postgres (Alembic, schema `public`)

`user`, `oauth_account` (fastapi-users), `conversations`, `messages`,
`retrieval_runs`, `retrieval_hits`, `feedback`, `skills`, và bảng checkpoint của
LangGraph (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, tạo bởi
`AsyncPostgresSaver.setup()` trong migration riêng). Corpus text không bao giờ vào
Postgres.

### 8.2 Lưu một lượt

`ChatTurnUseCase`: tạo conversation nếu thiếu (title = `standalone_query` cắt 80 ký
tự, đặt sau khi rephrase xong) → chạy graph, stream event → `append_turn` một
transaction (user message, assistant message với citations/phases/usage/run_id/
status, audit retrieval, `updated_at`) → nếu số lượt kể từ `summary_covers_message_id`
≥ 2 thì `BackgroundTasks.add_task(summarize)`. Persist lỗi: log, gửi event `error`
code `PERSIST_FAILED` sau `done`.

`SummarizeConversationUseCase`: summarizer (nano) nhận summary cũ + các lượt mới
(loại blocked/error/timeout) → `ConversationSummary` ≤ 1.500 ký tự → `update_summary`.

Checkpoint cleanup: task lúc khởi động xóa checkpoint có `thread_id` của run cũ hơn
7 ngày.

### 8.3 Skills

Khởi động: đọc `backend/skills/<name>/SKILL.md`, validate theo spec (tên thư mục phải trùng
`name`), rồi `replace_system` để Postgres chứa đúng bộ skill hệ thống trong repo. Upload:
multipart một file `SKILL.md` tối đa 64 KB (lớn hơn trả 413), validate cùng parser (sai
trả 422 kèm thông điệp của validator), trùng tên skill của chính user hoặc skill hệ thống
trả 409. Catalog khi resolve: `list_catalog(user_id, limit=31)`; hơn 30 thì bỏ qua skill
với `SKILL_RESOLUTION_FAILED`. Nội dung body của skill được chọn đưa vào prompt judge,
refine và answer.

### 8.4 Auth

fastapi-users với SQLAlchemy async: `User(id: UUID, email, hashed_password,
is_active, is_superuser, is_verified, display_name)`, `OAuthAccount`. Backend JWT
bearer (lifetime 7 ngày). Routes dưới `/api/v1/auth`: `register`, `jwt/login`,
`jwt/logout`, `google/authorize`, `google/callback`, `users/me`. Không xác minh
email (không có SMTP); `is_verified` không dùng để chặn. Mọi tài nguyên scope theo
`user_id`; truy cập tài nguyên của user khác trả 404.

### 8.5 API (`/api/v1`)

| Method | Path | Body / Query | Trả về |
| --- | --- | --- | --- |
| POST | `/chat/stream` | `{conversation_id?: str, message: str}` | SSE |
| POST | `/chat` | như trên | `ChatTurnResult` JSON |
| GET | `/conversations` | `?limit&cursor` | danh sách |
| GET | `/conversations/{id}` | | conversation |
| GET | `/conversations/{id}/messages` | `?limit&before` | messages kèm citations, phases, status |
| PATCH | `/conversations/{id}` | `{title}` | |
| DELETE | `/conversations/{id}` | | 204 |
| POST | `/messages/{id}/feedback` | `{rating: up \| down, note?}` | 201, ghi Langfuse score |
| GET | `/skills` | | skill hệ thống + của user |
| POST | `/skills` | multipart `file` | skill |
| PATCH | `/skills/{id}` | `{enabled}` | chỉ skill của user |
| DELETE | `/skills/{id}` | | chỉ skill của user |
| GET | `/health` | | trạng thái Postgres, Qdrant, embed, rerank, LLM configured |

SSE (sse-starlette, `X-Accel-Buffering: no`), mỗi event có `event` và `data` JSON:

| event | data |
| --- | --- |
| `phase` | `{phase: guarding \| understanding \| selecting_skills \| searching \| reading \| answering \| done, round?: int}` |
| `skills_selected` | `{skills: [{id, name}]}` |
| `evidence` | `{items: [{index, title, section, start_page, end_page, table_id?}]}` trước khi stream token |
| `token` | `{text}` |
| `citations` | `{items: [Citation]}` |
| `done` | `{message_id, conversation_id, status, usage: {llm_calls, prompt_tokens, completion_tokens, search_rounds}}` |
| `error` | `{code, message}` |

`ChatTurnResult` (non-stream) = `{conversation_id, message_id, status, content,
citations, phases, usage}`. Không có OpenAI key thì `/chat*` trả 503, phần còn
lại vẫn chạy.

## 9. LLM adapter và settings

`OpenAiLlmAdapter`: một `AsyncOpenAI` (bọc `langfuse.openai`) cho mỗi endpoint khác
nhau; `structured` dùng `chat.completions.parse(response_format=schema)`; `stream`
dùng `chat.completions.create(stream=True, stream_options={"include_usage": True})`.
Retry transient dùng `max_retries` của SDK. Chat Completions được chọn vì llama.cpp,
vLLM, Ollama đều hỗ trợ; đổi server chỉ là đổi `base_url`, `api_key`, `model`.

Settings (pydantic-settings, prefix `PHARMA_`, đọc `.env`):

```text
llm.default: {base_url, api_key, model}
llm.roles.{guardrail,rephrase,skill_selector,judge,refine,answer,summarizer}: {base_url?, api_key?, model?}  # thiếu thì lấy default
   mặc định: guardrail/rephrase/skill_selector/summarizer = gpt-5-nano; judge/refine/answer = gpt-5-mini
retrieval.*                     # bảng 7.1
budget.*                        # 5.2
postgres.dsn
qdrant.url, qdrant.api_key?
auth.jwt_secret, auth.google_client_id, auth.google_client_secret, auth.frontend_url
langfuse.public_key, langfuse.secret_key, langfuse.host
```

Validate lúc khởi động; thiếu `llm.default.api_key` thì agent tắt, API khác vẫn
chạy.

## 10. Lỗi và status

| Nơi lỗi | Chính sách | Status message |
| --- | --- | --- |
| Guardrail LLM | fail-open | không đổi |
| Rephrase, resolve_skills | fail-open | không đổi |
| Embed, Qdrant | retry SDK, rồi search lỗi → judge với evidence hiện có; không evidence → abstain | `partial` / `abstained` |
| Rerank | giữ thứ tự RRF | không đổi |
| Judge/refine | retry 1, rồi ép answer | `partial` |
| Answer | fallback deterministic | `error` |
| Deadline 90s | fallback | `timeout` |
| Budget | ép answer với evidence hiện có | `partial` / `abstained` |
| Persist | log, event `error` sau `done` | không đổi, message có thể không được lưu |

Fallback text (tiếng Việt, ngắn): "Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi này.
Bạn thử gửi lại sau ít phút." hoặc bản timeout tương ứng. Không LLM.

## 11. Observability

Langfuse 4.x: trace mỗi lượt (`session_id = conversation_id`, `user_id`,
`metadata.run_id`), span cho mỗi node và mỗi search qua `@observe`, generation tự
động từ wrapper `langfuse.openai` (LLM và embed), `CallbackHandler` truyền vào
config LangGraph để thấy graph. Feedback → `score` trên trace của message. Log
JSON (structlog hoặc logging chuẩn với formatter JSON) có `run_id`,
`conversation_id`, `user_id`. Langfuse Cloud mặc định; self-host chỉ đổi `host`.

## 12. Testing

- `tests/architecture`: AST test cấm domain import framework; api không import domain.
- `tests/domain`: `AgentRun` (allowed_actions theo từng trạng thái, chống lặp,
  reservation, `EvidenceRequired`, budget), `EvidenceSet` (merge, supersede, pack và
  hạ cấp strategy), parser SKILL.md, `SkillResolver`, regex guardrail, citation
  sanitizer (marker bị cắt giữa chunk, `[n]` không hợp lệ), `context_for_rephrase`.
  Thuần, không mock.
- `tests/application`: compile graph thật với `MemorySaver` và fake port (fake LLM
  trả structured theo kịch bản, fake retriever trả hits có sẵn), kiểm tra routing:
  smalltalk, blocked, redirect, grounded 1 vòng, refine 2 vòng, partial khi hết
  budget, abstain khi search lỗi, timeout, persist lỗi.
- `tests/infrastructure`: Qdrant adapter và Postgres repository trên container thật
  qua testcontainers (mark `integration`, skip nếu không có Docker); OpenAI adapter
  và llama.cpp reranker qua `respx`; test prompt reranker bằng chuỗi kỳ vọng copy từ
  pipeline.
- `tests/api`: `TestClient`, app.state giả, kiểm tra SSE frame theo thứ tự, mã lỗi
  401/404/413/503.
- Lệnh: `uv run ruff check`, `uv run ruff format --check`, `uv run pyrefly check`,
  `uv run pytest -q` (integration: `uv run pytest -q -m integration`).

## 13. Ngoài scope

Eval end-to-end chất lượng câu trả lời (spec riêng, dùng dataset 10k của
corpus-pipeline và Langfuse datasets), rate limiting, xác minh email, deploy
production, frontend, ingest corpus (thuộc corpus-pipeline).

## 14. Nhật ký quyết định

1. LangGraph thay vì state machine tự viết: ưu tiên thư viện có cộng đồng.
2. Agentic RAG judge/refine với structured output thay vì ReAct tool calling: model
   rẻ; mỗi call một việc nhỏ. Đường nâng cấp: gộp judge + refine thành một bước
   function calling 2 tool (`search`, `finish`) nếu sau này dùng model mạnh hơn.
3. Lần retrieve đầu tự động, hydrate tự động: giảm quyết định cho model yếu.
4. Chat Completions thay vì Responses API: tương thích self-host.
5. Reranker mặc định qwen3-4b vì eval tốt nhất, chấp nhận chậm hơn trên CPU;
   `native_rerank` giữ làm lựa chọn nhanh.
6. fastapi-users thay vì better-auth-server (mới 1 tuần tuổi) hay Authlib tự viết.
7. Langfuse thay vì tracing tự viết.
8. Không disclaimer y tế trong mọi prompt và template.
