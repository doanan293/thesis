# Native Rerank Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Leave `native_rerank` (`POST /v1/rerank`) as the only reranker protocol in seed-pipeline and backend, batch many documents into one llama.cpp pass on Kaggle and on the local CPU server, choose each server configuration with an end-to-end benchmark, and keep checkpoints reusable when the runtime configuration changes.

**Architecture:** Contract first: delete every completion-logprobs path while freezing the native contract digest that existing score caches and variant identities hash (Tasks 1–4). Then the serving model: a reranker `RuntimeCandidate` is one full llama-server level (`-np`, one size for `-c`/`-b`/`-ub`, documents per request, derived concurrency, optional CPU threads), `build_server_command` and `compose.yaml` both emit `--reranking --kv-unified -np N -c UB -b UB -ub UB` (Tasks 5, 9). The job reuse identity ignores the runtime profile (Task 6), workers send one request per query group (Tasks 7, 10), and one benchmark core in `runtime/benchmarking.py` measures candidates, checks scores against the first valid level and recommends by throughput (Kaggle) or p95 (CPU) (Tasks 8, 11).

**Tech Stack:** Python 3.12, uv workspace, Typer, pytest (`asyncio_mode = "auto"`, warnings are errors), ruff, pyrefly, llama.cpp server (Kaggle runtime `b9637`, compose image `ghcr.io/ggml-org/llama.cpp:server-b10920`), Docker Compose, httpx/respx (backend tests), FastAPI + pydantic-settings (backend).

**Spec:** seed-pipeline/docs/superpowers/specs/2026-09-15-native-rerank-optimisation-design.md (sections 4.1–4.6)

## Global Constraints

- llama.cpp flags must exist in both `b9637` (Kaggle) and `b10920` (compose): `--reranking`, `--kv-unified`, `-np`, `-c`, `-b`, `-ub`; env `LLAMA_ARG_RERANKING`, `LLAMA_ARG_KV_UNIFIED`, `LLAMA_ARG_N_PARALLEL`, `LLAMA_ARG_CTX_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`, `LLAMA_ARG_THREADS`. Never use `--kv-unified-per-slot` / `LLAMA_ARG_KV_UNIFIED_PER_SLOT` for a reranker (it exists only in `b10920`).
- Reranker server rule: `-c` = `-b` = `-ub` (not multiplied by slots); `-ub` ≥ 2048 (longest rerank prompt is 1,664 tokens); at most 256 slots. In `RuntimeCandidate`: `server_slots` = `-np`; `context_per_slot` = `logical_batch_size` = `physical_batch_size` = `-ub`; `request_batch_size` = documents per request; Kaggle `concurrency = ceil(server_slots / request_batch_size) + 1` per server.
- Search spaces: Kaggle 0.6b np 64, ub {8192, 16384, 32768}; 4b np 32, ub {8192, 16384}; 8b np 16, ub {4096, 8192}; `bge-reranker-v2-m3:f16` np 64, ub {8192, 16384, 32768}; 30 documents per request. Local CPU: np 16, ub {4096, 8192, 16384}, threads {8, 12}, 15 documents per request, concurrency 1.
- Benchmark: Kaggle 32 query groups × 30 documents per level, CPU 6 groups × 15 documents; one untimed warm-up group per level; server restarted per level; `max_abs_score_delta` > `1e-3` against the first valid level → `invalid` / `score_mismatch`; Kaggle picks the highest pairs/s, CPU the lowest p95 then the highest pairs/s; no valid level → stop and print the server log tails.
- f16 GGUF files only; no quantized files.
- `native_rerank_contract().sha256` stays `95b81f733a6695906ec4b9c0a30ab9588dc1f43bd9e64e45800101c87ee0eb48` (every `bge-reranker-v2-m3:f16` score and its variant `199c8f6b…` hash it).
- Base: branch `native-rerank-serving`, created from `dev` after `dev` was fast-forwarded to `main` at `18f5050` ("feat(seed): refresh corpus pipeline and native reranking"), which holds the spec and the native-default edits this plan builds on. Feature branches in this repo always start from `dev` level with `main`. That commit also carries another session's corpus work (`seed-pipeline/src/seed_pipeline/corpus/processing/clean_markdown_corpus.py`, `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py`, `seed-pipeline/pyproject.toml`, `uv.lock`, `seed-pipeline/data/corpus/*/manifest.json`, `seed-pipeline/data/sources/vietnamese_valid_syllables.json`, `seed-pipeline/tests/corpus/`, `seed-pipeline/tests/orchestration/`, `report/`). This plan never edits, stages, restores or formats those paths; if any of them shows uncommitted changes, another session is working there — leave them alone. Test failures confined to `tests/corpus/` or `tests/orchestration/` are not this plan's to fix: note them in the task report.
- Stage explicit paths only (`git add <path> ...`, `git rm <path>`); never `git add -A`, `git add .`, `git commit -a`. Review `git diff --cached --name-status` before every commit.
- Fix lint and type findings in code. No `# noqa`, `# type: ignore`, `# pyrefly: ignore`, ruff ignores or pyrefly error suppressions.
- pytest runs with `filterwarnings = ["error"]`; a warning is a failure.
- Test folders have no `__init__.py`; tests never write into the real `seed-pipeline/data/` (use `tmp_path` and parameters for roots).
- Kaggle worker modules (`integrations/kaggle/workers/*`) and `runtime/runtime_profiles.py`, `runtime/benchmarking.py` must not import `seed_pipeline.config.paths` or `pharma_agent`.
- Commit messages follow repo style (`feat(seed): …`, `refactor(backend): …`, `test(seed): …`) and end with the line `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. pre-commit runs on commit; on failure fix, re-stage the same explicit paths, commit again; never `--no-verify`.
- **Seed gate** (in `seed-pipeline/`): `uv run ruff check --fix src tests && uv run ruff format src tests && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check && uv run pytest -q`. The snippets in this plan are close to formatter output but import order is left to `ruff check --fix`. The fixers may touch only files this task changed; if they change anything else, leave that change unstaged and report it.
- **Backend gate** (in `backend/`): `uv run ruff check --fix src tests && uv run ruff format src tests && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check && uv run pytest -q`.
- Long-running commands (Docker integration test, anything over ~5 minutes) run inside tmux, one window per job.
- Plan B (quota-aware multi-account sessions, worker server restart, persistent logs, checkpoint seeding) and Plan C (`seed retrieve --sample`, instruction A/B, rollout) run after this plan and build on the interfaces listed in each task.

## File Structure

| Area | Files | Responsibility after this plan |
| --- | --- | --- |
| Contract | `seed-pipeline/src/seed_pipeline/runtime/model_profiles.py`, `runtime/catalog.py`, `evaluation/rerank_contract.py` | `RerankContract` names only `native_rerank`; frozen canonical payload; catalog has four native rerankers |
| Client | `runtime/client.py`, `evaluation/rerankers.py` | `LlamaCppClient.rerank_native` and embeddings only; `LlamaCppReranker` sends one native request |
| Server policy | `runtime/server_policy.py` | Cache flags only (`--cache-ram`, `--no-cache-idle-slots`) |
| Candidates | `runtime/runtime_profiles.py` | `RuntimeCandidate.threads`, `rerank_concurrency`, `reranker_candidate` |
| Kaggle command | `integrations/kaggle/workers/runtime.py` | `--reranking --kv-unified`, `-c` = `-ub` for rerankers |
| Reuse identity | `integrations/kaggle/models.py`, `integrations/kaggle/artifacts.py` | `reuse_payload_of` drops `input_sha256` and `runtime_parameters.runtime_profile` |
| Kaggle rerank | `integrations/kaggle/workers/rerank.py`, `workers/telemetry.py` | One `/v1/rerank` request per query group; no prompt-cache telemetry |
| Benchmark core | `runtime/benchmarking.py` | `BenchmarkMeasurement.candidate`, `RerankGroup`, sampling, score check, `recommend(objective=…)` |
| Kaggle benchmark | `integrations/kaggle/workers/benchmark.py`, `integrations/kaggle/stages.py`, `integrations/kaggle/auto_profile.py` | Full-candidate levels, per-level logs, profile from the recommendation |
| Compose | `compose.yaml`, `runtime/compose.py`, `.env.example`, `backend/.env.example`, `README.md` | Batched unified-KV reranker; manager passes a candidate's env; image tag reader |
| Local | `config/paths.py`, `evaluation/rerank_service.py`, `evaluation/local_rerank_benchmark.py` (new), `cli/commands/rerank.py` | Per-query local scoring, `seed rerank --backend local --benchmark`, `data/cache/local_profiles/rerank/<model>.json` |
| Run registry | `evaluation/run_workspace.py`, `data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json` | `unregister_rerank_variant`; Gemma variant unregistered, its report kept |
| Backend | `backend/src/pharma_agent/infrastructure/retrieval/llama_cpp_reranker.py`, `backend/src/pharma_agent/infrastructure/settings.py`, backend spec | `NativeReranker`, `NoopReranker`; protocol `native_rerank | none` |
| Docs | `seed-pipeline/docs/guides/{cli-reference,evaluation,workflow-local-only,downstream}.md`, `seed-pipeline/README.md`, `seed-pipeline/data/README.md` | Native-only rerank, Kaggle and CPU benchmarks, `cache/local_profiles/` |

New tests: `tests/evaluation/test_tracked_runs.py`, `tests/integrations/kaggle/test_job_identity.py`, `tests/evaluation/test_local_rerank_benchmark.py`, `tests/runtime/test_compose_reranker_integration.py` (all under `seed-pipeline/`).

---

### Task 1: Check the branch and a green baseline

The native-default edits (the three Qwen3 rerankers on `native_rerank`, the classifier 4b file, the backend default, compose/env/README/spec wording and the adjusted tests) were committed as `18f5050` on `main`; `dev` was fast-forwarded to it and branch `native-rerank-serving` was created from `dev`, together with a commit that adds Plans A, B and C. That base still exercises completion_logprobs through `qwen3_rerank_contract()` and `bge-reranker-v2-gemma:f16`; Tasks 2–4 rewrite it. This task only confirms the base and a green baseline.

**Files:** none changed.

**Interfaces:**
- Consumes: nothing.
- Produces: confirmed branch `native-rerank-serving` whose history contains `18f5050`, where catalog entries `qwen3-reranker:{0.6b,4b,8b}-fp16` have `reranker_protocol == "native_rerank"`, `qwen3-reranker:4b-fp16` has byte size `8_049_922_912` and sha256 `c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117`, and backend `RerankSettings.protocol` defaults to `"native_rerank"`.

- [ ] **Step 1: Check the branch and a clean tree**

Run (repo root):

```bash
git branch --show-current
git merge-base --is-ancestor 18f5050 HEAD && echo "base ok"
git merge-base --is-ancestor main dev && echo "dev contains main"
git log --oneline -3
git grep -n '= "native_rerank"' HEAD -- backend/src/pharma_agent/infrastructure/settings.py
git grep -n 'c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117' HEAD -- seed-pipeline/src/seed_pipeline/runtime/catalog.py
git status --short
```

Expected: `native-rerank-serving`, `base ok`, `dev contains main`, a log whose latest commit adds the three plans (`docs(seed): plan native rerank serving, quota sessions and rollout`), the protocol line `protocol: Literal["completion_logprobs", "native_rerank", "none"] = "native_rerank"`, one catalog match, and an empty `git status`. If the branch is missing, create it the repo's way (`git switch dev && git merge --ff-only main && git switch -c native-rerank-serving dev`) and ask the author, because the plan commit would then be missing too. Any other change: stop and report.

- [ ] **Step 2: Run both gates for a baseline**

Run the Seed gate and the Backend gate (Global Constraints). Expected: PASS. Record any failure with its test id; failures outside `tests/corpus/` and `tests/orchestration/` block the plan and must be reported before Task 2.

---

### Task 2: Remove completion_logprobs from seed-pipeline

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/runtime/model_profiles.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/catalog.py:6-14` (imports), `:136-199` (`_reranker`), `:256-308` (`RERANKER_MODELS`)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/client.py:10`, `:36-46`, `:66`, `:189-441`
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerankers.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerank_contract.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/server_policy.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py:37`, `:134`, `:246-378`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/benchmark.py:185`, `:214-256`, `:303`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/telemetry.py:25-34`, `:256-283`, `:315-331`, `:341-358`
- Modify: `seed-pipeline/src/seed_pipeline/runtime/benchmarking.py:40`, `:92-129`, `:181`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py:64-68`
- Modify: `seed-pipeline/src/seed_pipeline/runtime/compose.py:90-93`
- Test: `seed-pipeline/tests/runtime/test_model_profiles.py`, `seed-pipeline/tests/evaluation/test_rerank_contract.py`, `seed-pipeline/tests/evaluation/test_rerankers.py`, `seed-pipeline/tests/runtime/test_client.py`, `seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py`, `seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py`, `seed-pipeline/tests/integrations/kaggle/test_worker_telemetry.py`, `seed-pipeline/tests/runtime/test_benchmarking.py`, `seed-pipeline/tests/integrations/kaggle/test_auto_profile.py`, `seed-pipeline/tests/runtime/test_compose.py`, `seed-pipeline/tests/evaluation/conftest.py`, `seed-pipeline/tests/test_removed_contracts.py`

**Interfaces:**
- Consumes: Task 1 catalog.
- Produces:
  - `seed_pipeline.runtime.model_profiles.NATIVE_RERANK_PROTOCOL: str = "native_rerank"`
  - `seed_pipeline.runtime.model_profiles.RerankContract(protocol: str = "native_rerank")` with `canonical_payload() -> dict[str, object]` and `sha256: str`
  - `seed_pipeline.runtime.model_profiles.native_rerank_contract() -> RerankContract` (sha256 `95b81f73…eb48`, unchanged)
  - `seed_pipeline.runtime.server_policy.InferenceCachePolicy(host_cache_ram_mib: int, cache_idle_slots: bool, workload_locality: str)`
  - `RuntimeTelemetry.record_operation(server_index, item_count, input_characters, latency_seconds, status, retries) -> None` (no prompt-token arguments)
  - Removed: `CompletionScoring`, `qwen3_rerank_contract`, `bge_gemma_rerank_contract`, `build_qwen3_yes_no_prompt`, `build_bge_gemma_yes_no_prompt`, `build_qwen_rerank_prompt`, `prompt_contract_hash`, `CompletionRerankResult`, `CompletionPromptTiming`, `LlamaCppClient.completion_payload`, `LlamaCppClient.rerank_completion*`, `compare_cache_arms`, `CacheComparison`, catalog model `bge-reranker-v2-gemma:f16`.

- [ ] **Step 1: Write the failing guard tests**

Append to `seed-pipeline/tests/test_removed_contracts.py`:

```python
@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("seed_pipeline.runtime.model_profiles", "CompletionScoring"),
        ("seed_pipeline.runtime.model_profiles", "qwen3_rerank_contract"),
        ("seed_pipeline.runtime.model_profiles", "bge_gemma_rerank_contract"),
        ("seed_pipeline.runtime.model_profiles", "build_qwen3_yes_no_prompt"),
        ("seed_pipeline.runtime.model_profiles", "build_bge_gemma_yes_no_prompt"),
        ("seed_pipeline.evaluation.rerankers", "build_qwen_rerank_prompt"),
        ("seed_pipeline.evaluation.rerank_contract", "prompt_contract_hash"),
        ("seed_pipeline.runtime.client", "CompletionRerankResult"),
        ("seed_pipeline.runtime.client", "CompletionPromptTiming"),
        ("seed_pipeline.runtime.benchmarking", "compare_cache_arms"),
        ("seed_pipeline.runtime.benchmarking", "CacheComparison"),
    ],
)
def test_completion_logprobs_code_is_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "name",
    [
        "completion_payload",
        "rerank_completion",
        "rerank_completions_async",
        "rerank_completion_results_async",
    ],
)
def test_llama_cpp_client_has_no_completion_scoring(name: str) -> None:
    from seed_pipeline.runtime.client import LlamaCppClient

    assert not hasattr(LlamaCppClient, name)
```

Replace the whole of `seed-pipeline/tests/evaluation/test_rerank_contract.py` with:

```python
import pytest

from seed_pipeline.runtime.catalog import MODEL_CATALOG, RERANKER_MODELS, require_model
from seed_pipeline.runtime.model_profiles import native_rerank_contract

# Every bge-reranker-v2-m3 score in data/cache/rerank_scores was sealed with this digest.
SCORED_NATIVE_CONTRACT_SHA256 = (
    "95b81f733a6695906ec4b9c0a30ab9588dc1f43bd9e64e45800101c87ee0eb48"
)


@pytest.mark.parametrize("model", sorted(RERANKER_MODELS))
def test_every_reranker_is_called_through_v1_rerank(model: str) -> None:
    spec = require_model(model)

    assert spec.reranker_protocol == "native_rerank"
    assert spec.rerank_contract == native_rerank_contract()


def test_native_contract_digest_matches_existing_scores() -> None:
    assert native_rerank_contract().sha256 == SCORED_NATIVE_CONTRACT_SHA256


def test_completion_reranker_is_not_in_the_catalog() -> None:
    assert "bge-reranker-v2-gemma:f16" not in MODEL_CATALOG
```

- [ ] **Step 2: Run the guard tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/test_removed_contracts.py tests/evaluation/test_rerank_contract.py`
Expected: FAIL — `hasattr` assertions for the completion names and `bge-reranker-v2-gemma:f16` still in `MODEL_CATALOG`.

- [ ] **Step 3: Rewrite `model_profiles.py`**

Replace the whole of `seed-pipeline/src/seed_pipeline/runtime/model_profiles.py` with:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


NATIVE_RERANK_PROTOCOL = "native_rerank"


@dataclass(frozen=True)
class RerankContract:
    """How a reranker is called: POST /v1/rerank with the template inside the GGUF.

    The model sha256 already covers that template, so the contract names only the
    protocol.
    """

    protocol: str = NATIVE_RERANK_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != NATIVE_RERANK_PROTOCOL:
            raise ValueError(f"unsupported rerank protocol: {self.protocol}")

    def canonical_payload(self) -> dict[str, object]:
        # Score caches and rerank variant identities hash this payload, so it keeps the
        # shape it had when contracts also described completion prompts.
        return {
            "protocol": self.protocol,
            "template_id": None,
            "template_version": None,
            "instruction": "",
            "scoring": None,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class RerankRuntimeProfile:
    server_slots_per_gpu: int
    concurrency_per_gpu: int
    context_per_slot: int
    logical_batch_size: int
    physical_batch_size: int
    benchmark_concurrency: tuple[int, ...]

    def __post_init__(self) -> None:
        values = (
            self.server_slots_per_gpu,
            self.concurrency_per_gpu,
            self.context_per_slot,
            self.logical_batch_size,
            self.physical_batch_size,
        )
        if any(value < 1 for value in values):
            raise ValueError("rerank runtime values must be positive")
        if self.concurrency_per_gpu > self.server_slots_per_gpu:
            raise ValueError("rerank concurrency cannot exceed server slots")
        if not self.benchmark_concurrency or any(
            value < 1 for value in self.benchmark_concurrency
        ):
            raise ValueError("rerank benchmark concurrency must be positive")


@dataclass(frozen=True)
class EmbeddingWorkloadProfile:
    production_batch_size: int
    production_concurrency: int
    benchmark_batch_sizes: tuple[int, ...]
    benchmark_concurrency: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.production_batch_size < 1 or self.production_concurrency < 1:
            raise ValueError("embedding production values must be positive")
        if not self.benchmark_batch_sizes or any(
            value < 1 for value in self.benchmark_batch_sizes
        ):
            raise ValueError("embedding benchmark batch sizes must be positive")
        if not self.benchmark_concurrency or any(
            value < 1 for value in self.benchmark_concurrency
        ):
            raise ValueError("embedding benchmark concurrency must be positive")


@dataclass(frozen=True)
class EmbeddingRuntimeProfile:
    query: EmbeddingWorkloadProfile
    corpus: EmbeddingWorkloadProfile
    context_per_slot: int
    logical_batch_size: int
    physical_batch_size: int

    def __post_init__(self) -> None:
        if (
            min(
                self.context_per_slot, self.logical_batch_size, self.physical_batch_size
            )
            < 1
        ):
            raise ValueError("embedding runtime values must be positive")


def native_rerank_contract() -> RerankContract:
    return RerankContract()
```

(`RerankRuntimeProfile` stays until Task 5 removes it.)

- [ ] **Step 4: Drop the protocol parameter and the Gemma entry from the catalog**

In `seed-pipeline/src/seed_pipeline/runtime/catalog.py` replace the import block (lines 6–14) with:

```python
from seed_pipeline.runtime.model_profiles import (
    NATIVE_RERANK_PROTOCOL,
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
    RerankRuntimeProfile,
    native_rerank_contract,
)
```

Replace `def _reranker(...)` up to the start of its `runtime = RerankRuntimeProfile(` line (lines 136–157) with:

```python
def _reranker(
    name: str,
    filename: str,
    size: int,
    sha256: str,
    topology: ModelTopology,
    parallel: int = 1,
    batch: int = 1,
    context_per_slot: int = 4096,
    logical_batch_size: int = 4096,
    physical_batch_size: int = 2048,
) -> ModelSpec:
```

and in the returned `ModelSpec(...)` replace `reranker_protocol=protocol,` and `rerank_contract=contract,` with:

```python
reranker_protocol = (NATIVE_RERANK_PROTOCOL,)
rerank_contract = (native_rerank_contract(),)
```

`RerankContract` stays imported because `ModelSpec.rerank_contract: RerankContract | None` uses it. In `RERANKER_MODELS` delete the positional `"native_rerank",` argument from the four remaining entries and delete the whole `"bge-reranker-v2-gemma:f16": _reranker(...)` entry (lines 297–307).

- [ ] **Step 5: Delete completion scoring from the client**

In `seed-pipeline/src/seed_pipeline/runtime/client.py`:
- delete line 10 `from seed_pipeline.runtime.model_profiles import RerankContract`;
- delete the dataclasses `CompletionPromptTiming` and `CompletionRerankResult` (lines 36–46) and the now unused `from dataclasses import dataclass` import;
- delete line 66 `self._token_ids: dict[tuple[str, str], int] = {}`;
- delete everything from `def _single_token_id` (line 189) through the end of `rerank_completion_results_async` (line 441), so `def _request` directly follows `rerank_native`.

- [ ] **Step 6: Make the local reranker native-only**

Replace the whole of `seed-pipeline/src/seed_pipeline/evaluation/rerankers.py` with:

```python
import time
from typing import Protocol

import requests

from seed_pipeline.evaluation.candidate_text import candidate_document_text
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.runtime.catalog import ModelSpec
from seed_pipeline.runtime.client import LlamaCppClient

DEFAULT_RERANK_MAX_RETRIES = 3
DEFAULT_RERANK_RETRY_SLEEP_SECONDS = 5.0
RERANK_TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]: ...


class CachedReranker:
    accepts_query_row = True

    def __init__(self, cache: RerankScoreCache, reranker: str):
        self.cache = cache
        self.reranker = reranker

    def rerank(
        self, query_row: dict, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        scored = []
        for candidate in candidates:
            score = self.cache.require_score(self.reranker, query_row, candidate)
            scored.append(
                (score, candidate.rank, candidate.resolved_chunk_id, candidate)
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            item[3].with_rerank_score(item[0], rank)
            for rank, item in enumerate(scored, start=1)
        ]


def _exception_chain(exc: BaseException):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_transient(exc: BaseException) -> bool:
    for current in _exception_chain(exc):
        if isinstance(current, requests.exceptions.HTTPError):
            response = current.response
            return (
                response is None
                or response.status_code in RERANK_TRANSIENT_HTTP_STATUS_CODES
            )
        if isinstance(
            current,
            ConnectionError | TimeoutError | requests.exceptions.RequestException,
        ):
            return True
    return False


class LlamaCppReranker:
    """Scores every candidate of a query in one llama.cpp /v1/rerank request."""

    def __init__(
        self,
        spec: ModelSpec,
        client: LlamaCppClient,
        max_retries: int = DEFAULT_RERANK_MAX_RETRIES,
        retry_sleep_seconds: float = DEFAULT_RERANK_RETRY_SLEEP_SECONDS,
        retry_sleep=time.sleep,
    ) -> None:
        self.spec = spec
        self.client = client
        self.max_retries = max_retries
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.retry_sleep = retry_sleep

    def _call(self, operation):
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise
                self.retry_sleep(self.retry_sleep_seconds)
        raise AssertionError("unreachable")

    def rerank(
        self, query: str, candidates: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        documents = [
            candidate.document_text or candidate_document_text(candidate.payload)
            for candidate in candidates
        ]
        scores = self._call(
            lambda: self.client.rerank_native(query, documents, self.spec.name)
        )
        scored = [
            (score, original_index, candidate)
            for original_index, (score, candidate) in enumerate(
                zip(scores, candidates, strict=True)
            )
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            candidate.with_rerank_score(rerank_score=score, rank=rank)
            for rank, (score, _original_index, candidate) in enumerate(scored, start=1)
        ]
```

Replace the whole of `seed-pipeline/src/seed_pipeline/evaluation/rerank_contract.py` with:

```python
from __future__ import annotations

import hashlib


def document_hash(document_text: str) -> str:
    return hashlib.sha256(document_text.encode("utf-8")).hexdigest()
```

- [ ] **Step 7: Reduce the server cache policy to server flags**

Replace the whole of `seed-pipeline/src/seed_pipeline/runtime/server_policy.py` with:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from seed_pipeline.runtime.catalog import ModelKind, ModelSpec


@dataclass(frozen=True)
class InferenceCachePolicy:
    host_cache_ram_mib: int
    cache_idle_slots: bool
    workload_locality: str

    def __post_init__(self) -> None:
        if self.host_cache_ram_mib < 0:
            raise ValueError("host_cache_ram_mib must be non-negative")
        if not self.workload_locality.strip():
            raise ValueError("workload_locality must not be empty")

    def arguments(self) -> tuple[str, ...]:
        return (
            "--cache-ram",
            str(self.host_cache_ram_mib),
            "--cache-idle-slots" if self.cache_idle_slots else "--no-cache-idle-slots",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "host_cache_ram_mib": self.host_cache_ram_mib,
            "cache_idle_slots": self.cache_idle_slots,
            "workload_locality": self.workload_locality,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def inference_cache_policy(spec: ModelSpec) -> InferenceCachePolicy:
    if spec.kind is ModelKind.RERANKER:
        return InferenceCachePolicy(0, False, "query-group-request-v1")
    return InferenceCachePolicy(0, False, "batch-independent-v1")
```

- [ ] **Step 8: Keep only the native branch in the Kaggle rerank worker**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py`:
- replace line 37 `from seed_pipeline.runtime.server_policy import inference_cache_policy` with `from seed_pipeline.runtime.model_profiles import NATIVE_RERANK_PROTOCOL`;
- replace line 134 `protocol = str(config.get("protocol", "native_rerank"))` with:

```python
    protocol = str(config.get("protocol", NATIVE_RERANK_PROTOCOL))
    if protocol != NATIVE_RERANK_PROTOCOL:
        raise ValueError(f"unsupported rerank protocol: {protocol}")
```

- replace the block from `termination = None` (line 246) through the end of the `except Exception as error:` clause (line 378) with:

```python
termination = None
if score_pair is None and missing:
    server_config = dict(config)
    runtime_overrides = server_config.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("runtime_overrides is required")
    server_config["runtime_overrides"] = runtime_overrides
    try:
        if client_factory is None:
            from seed_pipeline.runtime.client import LlamaCppClient

            client_factory = LlamaCppClient
        with server_manager(server_config, telemetry=telemetry) as servers:
            clients = [client_factory(server.base_url) for server in servers]
            resources = list(enumerate(clients))
            per_client = max(1, int(runtime_overrides["concurrency"]))
            grouped: dict[tuple[str, str], list[Pair]] = {}
            for pair in missing:
                query, _text = details[_pair_key(pair)]
                grouped.setdefault((pair["query_id"], query), []).append(pair)
            groups = [
                (query_id, query, group) for (query_id, query), group in grouped.items()
            ]

            async def native_operation(resource, _index, item):
                server_index, client = resource
                query_id, query, group = item
                started_operation = clock()
                scores = await asyncio.to_thread(
                    client.rerank_native,
                    query,
                    [details[_pair_key(pair)][1] for pair in group],
                    model,
                )
                if len(scores) != len(group):
                    raise ValueError("reranker returned an unexpected score count")
                telemetry.record_operation(
                    server_index,
                    len(group),
                    sum(
                        len(query) + len(details[_pair_key(pair)][1]) for pair in group
                    ),
                    max(0.0, clock() - started_operation),
                    "ok",
                    0,
                )
                return (query_id, group, scores)

            async def run_native():
                return await stream_map_ordered(
                    groups,
                    resources,
                    per_client,
                    native_operation,
                    deadline,
                    on_completed=lambda batch: commit(
                        [
                            _score_record(pair, score)
                            for _index, value in batch
                            for pair, score in zip(value[1], value[2], strict=True)
                        ]
                    ),
                )

            scheduled = asyncio.run(run_native())
            if scheduled.stopped_early:
                emit(f"budget exhausted pairs={len(existing)}/{len(pairs)}")
    except Exception as error:
        termination = recoverable_termination(error)
        if termination is None:
            raise
```

- [ ] **Step 9: Drop the completion arm from the benchmark worker and benchmark core**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/benchmark.py`:
- delete the parameter line `cache_prompt: bool | None = None,` of `_measure_level` (line 185) and the keyword `cache_prompt=cache_prompt,` in its returned `BenchmarkMeasurement` (line 303);
- replace the rerank branch (lines 214–256, from `if "rerank" in str(config.get("stage", "")):` to the first `return None`) with:

```python
            if "rerank" in str(config.get("stage", "")):
                items = samples

                async def operation(resource, _index, item):
                    server_index, client = resource
                    started = time.monotonic()
                    query, document = item
                    await asyncio.to_thread(
                        client.rerank_native,
                        query,
                        [document],
                        str(config["model"]),
                    )
                    if recording:
                        telemetry.record_operation(
                            server_index,
                            1,
                            len(query) + len(document),
                            time.monotonic() - started,
                            "ok",
                            0,
                        )
                    return None
```

In `seed-pipeline/src/seed_pipeline/runtime/benchmarking.py`: delete the field `cache_prompt: bool | None = None` (line 40), the whole `CacheComparison` dataclass and `compare_cache_arms` function (lines 92–129), and the condition line `and item.cache_prompt is not False` inside `recommend` (line 181).

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py` delete lines 64–68 (the `comparison = …` read and its `if comparison is not None …: raise RuntimeError("benchmark cache comparison did not pass acceptance gates")`).

In `seed-pipeline/src/seed_pipeline/runtime/compose.py` delete lines 90–93 (the `if role == "reranker":` block that sets `LLAMA_RERANKER_RERANKING`); compose.yaml already defaults `LLAMA_ARG_RERANKING` to `true`.

- [ ] **Step 10: Remove prompt-token telemetry**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/telemetry.py`:
- delete the fields `prompt_tokens_cached` and `prompt_tokens_evaluated` of `OperationSample`;
- replace `record_operation` with:

```python
    def record_operation(
        self,
        server_index: int,
        item_count: int,
        input_characters: int,
        latency_seconds: float,
        status: str,
        retries: int,
    ) -> None:
        if item_count < 0 or input_characters < 0 or latency_seconds < 0 or retries < 0:
            raise ValueError("telemetry operation values must be non-negative")
        self.operations.append(
            OperationSample(
                server_index,
                item_count,
                input_characters,
                float(latency_seconds),
                status,
                retries,
            )
        )
```

- in `summary()` delete the `cached_values`, `evaluated_values`, `cached_total`, `evaluated_total`, `timing_total` locals and the three keys `"prompt_tokens_cached"`, `"prompt_tokens_evaluated"`, `"prompt_cache_hit_ratio"` of `"operations"`.

- [ ] **Step 11: Update the remaining tests**

Replace the whole of `seed-pipeline/tests/runtime/test_model_profiles.py` with:

```python
from dataclasses import replace

import pytest

from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.model_profiles import (
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
    RerankRuntimeProfile,
)


def test_rerank_contract_rejects_other_protocols():
    with pytest.raises(ValueError, match="unsupported rerank protocol"):
        RerankContract(protocol="completion_logprobs")


def test_runtime_changes_do_not_change_rerank_contract_hash():
    spec = require_model("qwen3-reranker:0.6b-fp16")
    assert spec.rerank_runtime is not None
    assert spec.rerank_contract is not None
    tuned = replace(
        spec, rerank_runtime=replace(spec.rerank_runtime, concurrency_per_gpu=2)
    )

    assert tuned.rerank_runtime != spec.rerank_runtime
    assert tuned.rerank_contract is not None
    assert tuned.rerank_contract.sha256 == spec.rerank_contract.sha256


def test_embedding_profile_separates_query_and_corpus_workloads():
    profile = require_model("qwen3-embedding:4b-fp16").embedding_runtime

    assert isinstance(profile, EmbeddingRuntimeProfile)
    assert isinstance(profile.query, EmbeddingWorkloadProfile)
    assert isinstance(profile.corpus, EmbeddingWorkloadProfile)
    assert profile.query is not profile.corpus
    assert profile.query.production_batch_size > 0
    assert profile.corpus.production_batch_size > 0


def test_embedding_model_exposes_separate_runtime_search_spaces():
    search_space = require_model("qwen3-embedding:4b-fp16").embedding_search_space

    assert search_space is not None
    assert search_space.query != search_space.corpus
    assert all(
        candidate.request_batch_size > 0 for candidate in search_space.query.candidates
    )


def test_reranker_exposes_runtime_search_space():
    spec = require_model("qwen3-reranker:0.6b-fp16")

    assert spec.rerank_search_space is not None
    assert len(spec.rerank_search_space.candidates) >= 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RerankRuntimeProfile(
            server_slots_per_gpu=1,
            concurrency_per_gpu=0,
            context_per_slot=4096,
            logical_batch_size=4096,
            physical_batch_size=2048,
            benchmark_concurrency=(1,),
        ),
        lambda: RerankRuntimeProfile(
            server_slots_per_gpu=1,
            concurrency_per_gpu=2,
            context_per_slot=4096,
            logical_batch_size=4096,
            physical_batch_size=2048,
            benchmark_concurrency=(1,),
        ),
    ],
)
def test_rerank_runtime_rejects_invalid_concurrency(factory):
    with pytest.raises(ValueError):
        factory()
```

Replace the whole of `seed-pipeline/tests/evaluation/test_rerankers.py` with:

```python
from unittest.mock import create_autospec

from seed_pipeline.evaluation.rerankers import LlamaCppReranker
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.client import LlamaCppClient


def _candidate(chunk_id: str, text: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        score=0.5,
        rank=rank,
        source="test",
        payload={"chunk_text": text},
    )


def test_local_reranker_scores_every_candidate_in_one_native_request():
    spec = require_model("qwen3-reranker:4b-fp16")
    client = create_autospec(LlamaCppClient, instance=True)
    client.rerank_native.return_value = [0.2, 0.9]
    reranker = LlamaCppReranker(spec, client)

    result = reranker.rerank(
        "thuốc gì",
        [_candidate("a", "tài liệu a", 1), _candidate("b", "tài liệu b", 2)],
    )

    client.rerank_native.assert_called_once_with(
        "thuốc gì", ["tài liệu a", "tài liệu b"], spec.name
    )
    assert [(item.chunk_id, item.rerank_score, item.rank) for item in result] == [
        ("b", 0.9, 1),
        ("a", 0.2, 2),
    ]
```

Replace the whole of `seed-pipeline/tests/runtime/test_client.py` with:

```python
import pytest

from seed_pipeline.runtime.client import LlamaCppClient, LlamaCppResponseError


class JsonResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> object:
        return self.payload


def test_rerank_native_returns_scores_in_document_order():
    seen = []

    def post(url, json, timeout):
        seen.append((url, json))
        return JsonResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.2},
                    {"index": 0, "relevance_score": 0.9},
                ]
            }
        )

    client = LlamaCppClient("http://server", post=post)

    assert client.rerank_native("q", ["a", "b"], "model") == [0.9, 0.2]
    assert seen == [
        (
            "http://server/v1/rerank",
            {"model": "model", "query": "q", "documents": ["a", "b"]},
        )
    ]


def test_rerank_native_rejects_duplicate_indices():
    client = LlamaCppClient(
        "http://server",
        post=lambda *_args, **_kwargs: JsonResponse(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.4},
                ]
            }
        ),
    )

    with pytest.raises(LlamaCppResponseError, match="duplicate"):
        client.rerank_native("q", ["a", "b"], "model")


def test_sync_request_retries_retryable_http_status():
    attempts = 0

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = "busy"

        def json(self):
            return {"ok": True}

    def post(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return Response(503 if attempts == 1 else 200)

    client = LlamaCppClient(
        "http://server", post=post, max_attempts=2, retry_delay_seconds=0
    )

    assert client._request("/health", {}) == {"ok": True}
    assert attempts == 2
```

In `seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py`:
- replace the client import block with `from seed_pipeline.runtime.client import LlamaCppRequestError, LlamaCppResponseError`;
- in `_config` set `"model": "qwen3-reranker:0.6b-fp16"` and `"protocol": "native_rerank"`;
- delete the class `_CompletionCacheAwareClient` and the test `test_completion_worker_enables_slot_prompt_cache_and_records_timings`;
- append:

```python
def test_rerank_worker_rejects_non_native_protocol(tmp_path):
    config = _config(tmp_path, query_count=1, candidates_per_query=1)
    config["protocol"] = "completion_logprobs"

    with pytest.raises(ValueError, match="unsupported rerank protocol"):
        run_rerank_worker(config, score_pair=lambda _q, _d, _m: 0.5)
```

In `seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py` delete `test_completion_policy_preserves_slot_prompt_reuse` and replace the four policy tests `test_native_rerank_policy_has_no_completion_request_setting`, `test_embedding_policy_has_no_completion_request_setting`, `test_stateless_policy_rejects_negative_cache_limit`, `test_server_policy_fingerprint_changes_with_performance_behavior` with:

```python
@pytest.mark.parametrize("model", ("bge-reranker-v2-m3:f16", "qwen3-reranker:4b-fp16"))
def test_reranker_policy_is_stateless_per_query_group(model):
    policy = inference_cache_policy(require_model(model))

    assert policy.host_cache_ram_mib == 0
    assert policy.cache_idle_slots is False
    assert policy.workload_locality == "query-group-request-v1"
    assert policy.arguments() == ("--cache-ram", "0", "--no-cache-idle-slots")


def test_embedding_policy_is_batch_independent():
    policy = inference_cache_policy(require_model("qwen3-embedding:4b-fp16"))

    assert policy.workload_locality == "batch-independent-v1"


def test_stateless_policy_rejects_negative_cache_limit():
    from seed_pipeline.runtime.server_policy import InferenceCachePolicy

    with pytest.raises(ValueError, match="cache_ram_mib"):
        InferenceCachePolicy(-1, False, "test")


def test_server_policy_fingerprint_changes_with_performance_behavior():
    from seed_pipeline.runtime.server_policy import InferenceCachePolicy

    stateless = InferenceCachePolicy(0, False, "query-group-request-v1")
    cached = InferenceCachePolicy(8192, True, "query-group-request-v1")

    assert stateless.to_dict() == {
        "schema_version": 2,
        "host_cache_ram_mib": 0,
        "cache_idle_slots": False,
        "workload_locality": "query-group-request-v1",
    }
    assert stateless.sha256 != cached.sha256
```

In `seed-pipeline/tests/integrations/kaggle/test_worker_telemetry.py`:
- change `telemetry.record_operation(0, 1, 321, 1.2, "success", 0, 120, 30)` to `telemetry.record_operation(0, 1, 321, 1.2, "success", 0)`;
- replace the three `prompt_tokens_cached`, `prompt_tokens_evaluated`, `prompt_cache_hit_ratio` assertions with `assert "prompt_cache_hit_ratio" not in report["operations"]`;
- delete `test_embedding_summary_has_no_prompt_cache_ratio`.

In `seed-pipeline/tests/runtime/test_benchmarking.py` delete `compare_cache_arms` from the import list and delete `test_recommendation_excludes_disabled_cache_control_arm` and `test_cache_comparison_applies_score_and_performance_gates`.

In `seed-pipeline/tests/integrations/kaggle/test_auto_profile.py` change `InferenceCachePolicy(8192, True, True, 0.1, "query-adjacent-v1")` to `InferenceCachePolicy(8192, True, "query-adjacent-v1")`.

In `seed-pipeline/tests/runtime/test_compose.py` replace the parametrized test `test_reranker_service_serves_rerank_endpoint_only_for_native_protocol` with:

```python
@pytest.mark.parametrize("model", ["bge-reranker-v2-m3:f16", "qwen3-reranker:4b-fp16"])
def test_reranker_service_leaves_reranking_to_compose(
    tmp_path: Path, model: str
) -> None:
    artifact = tmp_path / "reranker.gguf"
    artifact.write_bytes(b"gguf")
    spec = replace(
        require_model(model),
        canonical_filename=artifact.name,
        byte_size=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )
    runner = RecordingRunner()
    manager = LlamaCppComposeManager(
        tmp_path / "compose.yaml",
        runner=runner,
        client_factory=HealthyClient,
        environment={},
    )

    endpoint = manager.ensure("reranker", spec, tmp_path)

    command, environment = runner.calls[0]
    assert command[-1] == "llama-reranker"
    assert environment is not None
    assert environment["LLAMA_RERANKER_MODEL"] == "reranker.gguf"
    assert "LLAMA_RERANKER_RERANKING" not in environment
    assert endpoint == "http://127.0.0.1:11435"
```

In `seed-pipeline/tests/evaluation/conftest.py` delete `from seed_pipeline.evaluation.rerank_contract import prompt_contract_hash` and replace `contract = prompt_contract_hash(protocol=spec.reranker_protocol or "")` with:

```python
    assert spec.rerank_contract is not None
    contract = spec.rerank_contract.sha256
```

- [ ] **Step 12: Run the focused tests**

Run (in `seed-pipeline/`): `uv run pytest -q tests/test_removed_contracts.py tests/evaluation tests/runtime tests/integrations/kaggle`
Expected: PASS.

- [ ] **Step 13: Confirm no completion code remains**

Run (in `seed-pipeline/`):

```bash
grep -rnE 'completion_logprobs|CompletionScoring|qwen3_rerank_contract|bge_gemma|yes_no_prompt|build_qwen_rerank_prompt|rerank_completion|completion_payload|CompletionRerankResult|CompletionPromptTiming|compare_cache_arms|CacheComparison|cache_comparison|completion_cache_prompt|slot_prompt_similarity|prompt_tokens_|prompt_contract_hash|bge-reranker-v2-gemma' src
```

Expected: no output. The same grep over `tests` may only match `tests/test_removed_contracts.py`, `tests/evaluation/test_rerank_contract.py`, `tests/runtime/test_model_profiles.py` and `tests/integrations/kaggle/test_worker_rerank.py` (rejection tests).

- [ ] **Step 14: Run the Seed gate**

Expected: PASS. (`seed metrics` on the real run stays broken for the Gemma variant until Task 3; no unit test reads it yet.)

- [ ] **Step 15: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/runtime/model_profiles.py seed-pipeline/src/seed_pipeline/runtime/catalog.py seed-pipeline/src/seed_pipeline/runtime/client.py seed-pipeline/src/seed_pipeline/evaluation/rerankers.py seed-pipeline/src/seed_pipeline/evaluation/rerank_contract.py seed-pipeline/src/seed_pipeline/runtime/server_policy.py seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/benchmark.py seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/telemetry.py seed-pipeline/src/seed_pipeline/runtime/benchmarking.py seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py seed-pipeline/src/seed_pipeline/runtime/compose.py seed-pipeline/tests/runtime/test_model_profiles.py seed-pipeline/tests/evaluation/test_rerank_contract.py seed-pipeline/tests/evaluation/test_rerankers.py seed-pipeline/tests/runtime/test_client.py seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py seed-pipeline/tests/integrations/kaggle/test_worker_telemetry.py seed-pipeline/tests/runtime/test_benchmarking.py seed-pipeline/tests/integrations/kaggle/test_auto_profile.py seed-pipeline/tests/runtime/test_compose.py seed-pipeline/tests/evaluation/conftest.py seed-pipeline/tests/test_removed_contracts.py
git diff --cached --name-status
git commit -m "refactor(seed): remove completion_logprobs reranking" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Unregister the Gemma variant and delete its data

The Gemma report stays as the final result; its variant, rerank artifact, score cache and Kaggle profile go.

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py:217-229` (add a method after `register_rerank_variant`)
- Modify (data, tracked): `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json`
- Delete (tracked): `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/bge_reranker_v2_gemma_f16/manifest.json`
- Delete (ignored): `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/bge_reranker_v2_gemma_f16/rerank_scores.jsonl`, `seed-pipeline/data/cache/rerank_scores/bge_reranker_v2_gemma_f16.jsonl`, `seed-pipeline/data/cache/kaggle_profiles/rerank/bge_reranker_v2_gemma_f16.json`
- Keep: `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/bge_reranker_v2_gemma_f16/`
- Test: `seed-pipeline/tests/evaluation/test_run_workspace.py`, `seed-pipeline/tests/evaluation/test_tracked_runs.py` (new), `seed-pipeline/tests/evaluation/test_metrics_artifacts.py`

**Interfaces:**
- Consumes: Task 2 catalog without `bge-reranker-v2-gemma:f16`.
- Produces: `RunWorkspace.unregister_rerank_variant(model_slug: str) -> RerankVariantRecord` (raises `RunConflictError` for an unknown slug).

- [ ] **Step 1: Write the failing tests**

Append to `seed-pipeline/tests/evaluation/test_run_workspace.py`:

```python
def test_unregister_rerank_variant_keeps_the_other_variants(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    kept = RerankVariantRecord("model-a", "a" * 64, "rerank/model_a")
    removed = RerankVariantRecord("model-b", "b" * 64, "rerank/model_b")
    workspace.register_rerank_variant("model_a", kept)
    workspace.register_rerank_variant("model_b", removed)

    assert workspace.unregister_rerank_variant("model_b") == removed
    assert workspace.variant_records() == {"model_a": kept}
    with pytest.raises(RunConflictError, match="model_b"):
        workspace.unregister_rerank_variant("model_b")
```

Create `seed-pipeline/tests/evaluation/test_tracked_runs.py`:

```python
from pathlib import Path

import pytest

from seed_pipeline.config.paths import RUNS_DIR
from seed_pipeline.evaluation.run_workspace import load_run_record
from seed_pipeline.runtime.catalog import RERANKER_MODELS

TRACKED_RUN_RECORDS = sorted(RUNS_DIR.glob("*/run.json"))


@pytest.mark.parametrize(
    "record_path", TRACKED_RUN_RECORDS, ids=lambda path: path.parent.name
)
def test_tracked_runs_register_only_catalog_rerankers(record_path: Path) -> None:
    record = load_run_record(record_path)

    assert {variant.model for variant in record.rerank_variants.values()} <= set(
        RERANKER_MODELS
    )
```

Append to `seed-pipeline/tests/evaluation/test_metrics_artifacts.py` (it already imports `run_metrics` and `MetricsRequest`; add `from pathlib import Path` only if the file does not import it):

```python
def test_metrics_leave_reports_of_unregistered_rerankers_alone(complete_run: Path):
    report = (
        complete_run
        / "reports"
        / "rerank"
        / "bge_reranker_v2_gemma_f16"
        / "top1-window3"
        / "report.md"
    )
    report.parent.mkdir(parents=True)
    report.write_text("final Gemma numbers\n", encoding="utf-8")

    result = run_metrics(MetricsRequest(complete_run, top_k=1))

    assert result.reranked == ()
    assert report.read_text(encoding="utf-8") == "final Gemma numbers\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation/test_run_workspace.py tests/evaluation/test_tracked_runs.py tests/evaluation/test_metrics_artifacts.py`
Expected: FAIL — `AttributeError: 'RunWorkspace' object has no attribute 'unregister_rerank_variant'`, and `test_tracked_runs_register_only_catalog_rerankers[hybrid-qwen4b-p50-k30-rrf2]` fails because `bge-reranker-v2-gemma:f16` is registered. The metrics sentinel test passes already (it guards the behaviour this task relies on).

- [ ] **Step 3: Add the registry method**

In `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py` insert after `register_rerank_variant`:

```python
def unregister_rerank_variant(self, model_slug: str) -> RerankVariantRecord:
    current = self.read_record()
    if model_slug not in current.rerank_variants:
        raise RunConflictError(f"Run {self.root} has no rerank variant {model_slug}")
    variants = dict(current.rerank_variants)
    removed = variants.pop(model_slug)
    self.write_record(
        RunRecord(current.identity, current.origin, current.candidates_dir, variants)
    )
    return removed
```

- [ ] **Step 4: Unregister Gemma from the tracked run with the new method**

Run (in `seed-pipeline/`):

```bash
uv run python - <<'EOF'
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.run_workspace import RunWorkspace, load_run_record

root = run_dir("hybrid-qwen4b-p50-k30-rrf2")
record = load_run_record(root / "run.json")
workspace = RunWorkspace(root, record.identity, record.origin)
print(workspace.unregister_rerank_variant("bge_reranker_v2_gemma_f16"))
EOF
git diff -- data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json
```

Expected: the diff removes only the five-line `"bge_reranker_v2_gemma_f16": {…}` block.

- [ ] **Step 5: Delete the Gemma artifact, score cache and profile**

Run (repo root):

```bash
git rm seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/bge_reranker_v2_gemma_f16/manifest.json
rm seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/bge_reranker_v2_gemma_f16/rerank_scores.jsonl
rmdir seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/bge_reranker_v2_gemma_f16
rm seed-pipeline/data/cache/rerank_scores/bge_reranker_v2_gemma_f16.jsonl
rm seed-pipeline/data/cache/kaggle_profiles/rerank/bge_reranker_v2_gemma_f16.json
ls seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/bge_reranker_v2_gemma_f16/top30-window3
```

Expected: the last command lists `manifest.json`, `metrics.jsonl`, `report.md`.

- [ ] **Step 6: Check the untouched bge-m3 variant still validates**

Run (in `seed-pipeline/`):

```bash
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --model bge-reranker-v2-m3:f16 --top-k 30
git status --short data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/reports
```

Expected: `status=complete`; `git status` prints nothing (the baseline and bge-m3 reports are reused, the Gemma report is not touched). Do not run `seed metrics` without `--model` on this run: the three Qwen3 variants were scored with completion_logprobs and fail the native contract check until Plan C re-scores them.

- [ ] **Step 7: Run the Seed gate**

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json seed-pipeline/tests/evaluation/test_run_workspace.py seed-pipeline/tests/evaluation/test_tracked_runs.py seed-pipeline/tests/evaluation/test_metrics_artifacts.py
git diff --cached --name-status
git commit -m "chore(seed-data): unregister the bge-reranker-v2-gemma variant and keep its report" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Expected staged set: the five paths above plus `D seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/bge_reranker_v2_gemma_f16/manifest.json` from Step 5.

---

### Task 4: Remove completion_logprobs from the backend

**Files:**
- Modify: `backend/src/pharma_agent/infrastructure/settings.py:100` (`RerankSettings.protocol`)
- Modify: `backend/src/pharma_agent/infrastructure/retrieval/llama_cpp_reranker.py` (whole file)
- Modify: `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md:88`, `:323`, `:348-359`, `:514-516`, `:536-540`
- Modify: `compose.yaml:87-88`, `.env.example:18-21`, `README.md:51-54`
- Test: `backend/tests/infrastructure/test_llama_cpp_reranker.py` (whole file), `backend/tests/infrastructure/test_settings.py`, `backend/tests/infrastructure/test_langfuse_tracing.py:130`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `RerankSettings.protocol: Literal["native_rerank", "none"] = "native_rerank"`; `build_reranker(settings: RerankSettings) -> Reranker` returns `NoopReranker` or `NativeReranker`; `LlamaCppCompletionReranker` and `build_qwen3_yes_no_prompt` no longer exist.

- [ ] **Step 1: Write the failing settings test**

In `backend/tests/infrastructure/test_settings.py` add `from pydantic import ValidationError` to the imports, change `monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "completion_logprobs")` to `monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "none")` and `assert settings.retrieval.rerank.protocol == "completion_logprobs"` to `assert settings.retrieval.rerank.protocol == "none"`, and append:

```python
def test_rerank_protocol_rejects_completion_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHARMA_LLM__DEFAULT__API_KEY", "sk-test")
    monkeypatch.setenv("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "completion_logprobs")

    with pytest.raises(ValidationError, match="protocol"):
        Settings(_env_file=None)
```

- [ ] **Step 2: Run it to verify it fails**

Run (in `backend/`): `uv run pytest -q tests/infrastructure/test_settings.py`
Expected: FAIL — `DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>`.

- [ ] **Step 3: Narrow the setting and rewrite the reranker module**

In `backend/src/pharma_agent/infrastructure/settings.py` replace line 100 with:

```python
    protocol: Literal["native_rerank", "none"] = "native_rerank"
```

Replace the whole of `backend/src/pharma_agent/infrastructure/retrieval/llama_cpp_reranker.py` with:

```python
"""Rerankers over an OpenAI-style `/v1/rerank` endpoint (llama.cpp, TEI, vLLM).

The rerank template ships inside the GGUF, so the agent and the seed-pipeline evaluation
score candidates with the same computation without copying prompts."""

import asyncio
from collections.abc import Sequence

import httpx

from pharma_agent.domain.retrieval.models import Hit
from pharma_agent.domain.retrieval.ports import Reranker, RetrievalError
from pharma_agent.infrastructure.settings import RerankSettings


def _document_text(hit: Hit) -> str:
    return hit.embedding_text or f"{hit.context_header}\n\n{hit.chunk_text}".strip()


def _sorted_top(
    hits: Sequence[Hit], scores: Sequence[float | None], top_n: int
) -> list[Hit]:
    scored = [
        h.model_copy(update={"rerank_score": s})
        for h, s in zip(hits, scores, strict=True)
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:top_n]


class NativeReranker:
    """`POST /v1/rerank` with every candidate of a search round in one request."""

    def __init__(
        self, http: httpx.AsyncClient, *, model: str, max_concurrent: int = 2
    ) -> None:
        self._http = http
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": [_document_text(h) for h in hits],
            "top_n": len(hits),
        }
        try:
            async with self._semaphore:
                response = await self._http.post("/v1/rerank", json=body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RetrievalError(f"llama.cpp /v1/rerank failed: {exc}") from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RetrievalError("rerank response has no results")
        scores: list[float | None] = [None] * len(hits)
        for item in results:
            index, score = item.get("index"), item.get("relevance_score")
            if (
                isinstance(index, int)
                and 0 <= index < len(hits)
                and isinstance(score, int | float)
            ):
                scores[index] = float(score)
        return _sorted_top(hits, scores, top_n)


class NoopReranker:
    async def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        return sorted(hits, key=lambda h: h.fusion_score, reverse=True)[:top_n]


def build_reranker(settings: RerankSettings) -> Reranker:
    if settings.protocol == "none":
        return NoopReranker()
    headers = (
        {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else None
    )
    http = httpx.AsyncClient(
        base_url=settings.base_url, timeout=settings.timeout_seconds, headers=headers
    )
    return NativeReranker(
        http, model=settings.model, max_concurrent=settings.max_concurrent
    )
```

- [ ] **Step 4: Rewrite the reranker tests and the tracing test**

Replace the whole of `backend/tests/infrastructure/test_llama_cpp_reranker.py` with:

```python
import json

import httpx
import pytest
import respx

from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.infrastructure.retrieval.llama_cpp_reranker import (
    NativeReranker,
    NoopReranker,
    build_reranker,
)
from pharma_agent.infrastructure.settings import RerankSettings
from tests.domain.factories import chunk_uuid, make_hit

BASE = "http://rerank"


@respx.mock(base_url=BASE)
async def test_native_reranker_scores_every_candidate_in_one_request(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post("/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.7},
                    {"index": 0, "relevance_score": 0.3},
                ]
            },
        )
    )
    async with httpx.AsyncClient(base_url=BASE) as http:
        hits = await NativeReranker(http, model="qwen3-reranker:4b-fp16").rerank(
            "Q", [make_hit("a"), make_hit("b")], top_n=1
        )

    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b")]
    assert hits[0].rerank_score == 0.7
    assert len(route.calls) == 1
    body = json.loads(route.calls[0].request.content)
    assert (body["model"], body["query"], body["top_n"]) == (
        "qwen3-reranker:4b-fp16",
        "Q",
        2,
    )
    assert len(body["documents"]) == 2
    # Scores embedding_text, like the evaluation.
    assert body["documents"][0].startswith("Paracetamol > Liều dùng")


@respx.mock(base_url=BASE)
async def test_native_reranker_maps_http_errors(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/rerank").mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient(base_url=BASE) as http:
        reranker = NativeReranker(http, model="m")
        with pytest.raises(RetrievalError):
            await reranker.rerank("Q", [make_hit("a")], top_n=1)


async def test_noop_reranker_keeps_fusion_order() -> None:
    hits = await NoopReranker().rerank(
        "Q", [make_hit("a", fusion=0.1), make_hit("b", fusion=0.9)], top_n=1
    )
    assert [h.chunk_version_id for h in hits] == [chunk_uuid("b")]
    assert hits[0].rerank_score is None


def test_build_reranker_switches_on_protocol() -> None:
    assert isinstance(build_reranker(RerankSettings(protocol="none")), NoopReranker)
    assert isinstance(
        build_reranker(RerankSettings(protocol="native_rerank")), NativeReranker
    )


@respx.mock(base_url=BASE)
async def test_build_reranker_sends_the_api_key_only_when_configured(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post("/v1/rerank").mock(
        return_value=httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
        )
    )
    for api_key in ("rerank-secret", None):
        reranker = build_reranker(
            RerankSettings(protocol="native_rerank", base_url=BASE, api_key=api_key)
        )
        assert isinstance(reranker, NativeReranker)
        await reranker.rerank("Q", [make_hit("a")], top_n=1)
        await reranker.aclose()
    assert route.calls[0].request.headers["Authorization"] == "Bearer rerank-secret"
    assert "Authorization" not in route.calls[1].request.headers
```

In `backend/tests/infrastructure/test_langfuse_tracing.py` line 130 change `protocol="completion_logprobs"` to `protocol="native_rerank"`.

- [ ] **Step 5: Update the stack config and docs**

In `compose.yaml` replace lines 87–88 with:

```yaml
      # /v1/rerank needs a classifier GGUF (tensor cls.output.weight).
      LLAMA_ARG_RERANKING: "true"
```

In `.env.example` replace lines 18–21 (from `# LLAMA_RERANKER_PROTOCOL=native_rerank` through `# LLAMA_RERANKER_RERANKING=true`) with:

```text
# The reranker GGUF must be a classifier conversion (tensor cls.output.weight) so llama-server
# serves /v1/rerank. LLAMA_RERANKER_PROTOCOL=none keeps the RRF order without reranking.
# LLAMA_RERANKER_PROTOCOL=native_rerank
```

In `README.md` replace the sentence that starts `Reranker mặc định chạy` and ends with `` `LLAMA_RERANKER_RERANKING=false`. `` (lines 51–54) with:

```text
  `/health` báo `CORPUS_NOT_READY` nếu metadata collection không khớp. Reranker chỉ chạy qua
  `/v1/rerank`, nên file GGUF phải là bản convert classifier (có tensor `cls.output.weight`);
  đặt `LLAMA_RERANKER_PROTOCOL=none` để bỏ rerank.
```

(keep the preceding `` `ai-models/gguf`, rồi `docker compose up -d`. Embedding phải là model đã dùng khi import corpus; `` line as is).

In `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md`:
- line 88: `      retrieval/llama_cpp/   # NativeReranker, NoopReranker`
- line 323: `` | `retrieval.rerank.protocol` | `native_rerank` (`none` tắt rerank) | ``
- replace step 4 of §7.2 (lines 348–359, from `4. Rerank theo` through the `` - `none`: `` bullet) with:

```text
4. Rerank theo `standalone_query` trên `embedding_text`, protocol:
   - `native_rerank`: POST `/v1/rerank` `{model, query, documents, top_n}` →
     `results[].relevance_score`, một request chứa mọi ứng viên của vòng search. Template
     `rerank` nằm trong file GGUF bản convert classifier (có `cls.output.weight`), nên backend
     và seed-pipeline chấm cùng một phép tính mà không chép prompt.
   - `none`: giữ thứ tự RRF, `rerank_score = None`.
```

- lines 514–516: replace `` và llama.cpp reranker qua `respx`; test prompt reranker bằng chuỗi kỳ vọng copy từ `` and the following `  pipeline.` with `` và llama.cpp reranker (`/v1/rerank`) qua `respx`. ``
- decision 5 (lines 536–539, from `5. Reranker mặc định` through `classifier head.`): replace with:

```text
5. Reranker mặc định qwen3-4b vì eval tốt nhất, chấp nhận chậm hơn trên CPU. Chỉ gọi qua
   `native_rerank` (`/v1/rerank`, GGUF convert classifier): llama.cpp, vLLM, TEI và các nhà
   host đều có endpoint rerank, và phép softmax trên logit "yes"/"no" qua `cls.output.weight`
   đúng với model card. `completion_logprobs` đã bỏ: nó chấm sai bản 8b (llama.cpp lấy nhầm
   `token_embd.weight` làm lớp đầu ra) và buộc chép prompt giữa hai project.
```

- [ ] **Step 6: Run the tests and confirm nothing mentions the removed protocol**

Run (in `backend/`): `uv run pytest -q tests/infrastructure/test_settings.py tests/infrastructure/test_llama_cpp_reranker.py tests/infrastructure/test_langfuse_tracing.py tests/infrastructure/test_composition.py`
Expected: PASS.

Run (repo root): `grep -rn 'completion_logprobs\|LlamaCppCompletionReranker\|LLAMA_RERANKER_RERANKING' backend/src backend/tests backend/.env.example compose.yaml .env.example README.md`
Expected: only the `test_rerank_protocol_rejects_completion_logprobs` lines in `backend/tests/infrastructure/test_settings.py`.

- [ ] **Step 7: Run the Backend gate**

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/src/pharma_agent/infrastructure/settings.py backend/src/pharma_agent/infrastructure/retrieval/llama_cpp_reranker.py backend/tests/infrastructure/test_llama_cpp_reranker.py backend/tests/infrastructure/test_settings.py backend/tests/infrastructure/test_langfuse_tracing.py backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md compose.yaml .env.example README.md
git diff --cached --name-status
git commit -m "refactor(backend): drop the completion_logprobs reranker" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Batched reranker levels, search spaces and server command

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/runtime/runtime_profiles.py:1-86` (imports, `RuntimeCandidate`, new helpers)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/catalog.py` (imports, `ModelSpec`, constants, `_reranker`, `RERANKER_MODELS`)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/model_profiles.py` (delete `RerankRuntimeProfile`)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/benchmarking.py:10-13`, `:167-170` (delete `rerank_levels`)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/runtime.py:328-368` (`build_server_command`)
- Test: `seed-pipeline/tests/runtime/test_model_profiles.py`, `seed-pipeline/tests/runtime/test_runtime_profiles.py`, `seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py`, `seed-pipeline/tests/test_removed_contracts.py`

**Interfaces:**
- Consumes: Task 2 `NATIVE_RERANK_PROTOCOL`, `native_rerank_contract()`.
- Produces:
  - `RuntimeCandidate(server_slots, concurrency, request_batch_size, context_per_slot, logical_batch_size, physical_batch_size, threads: int | None = None)`; `to_dict()` includes `"threads"` only when set, so existing embedding search-space digests do not change.
  - `seed_pipeline.runtime.runtime_profiles.MIN_RERANK_UBATCH = 2048`
  - `seed_pipeline.runtime.runtime_profiles.rerank_concurrency(server_slots: int, request_batch_size: int) -> int` (`ceil(server_slots / request_batch_size) + 1`)
  - `seed_pipeline.runtime.runtime_profiles.reranker_candidate(*, server_slots: int, ubatch: int, request_batch_size: int, concurrency: int, threads: int | None = None) -> RuntimeCandidate`
  - `seed_pipeline.runtime.catalog.KAGGLE_RERANK_REQUEST_BATCH_SIZE = 30`, `LOCAL_RERANK_REQUEST_BATCH_SIZE = 15`, `LOCAL_RERANK_SEARCH_SPACE: RuntimeSearchSpace`
  - `seed_pipeline.runtime.catalog._reranker(name: str, filename: str, size: int, sha256: str, topology: ModelTopology, *, server_slots: int, ubatch_sizes: tuple[int, ...]) -> ModelSpec` (Plan C adds its experimental model with this call)
  - `ModelSpec.local_rerank_search_space: RuntimeSearchSpace | None`; `ModelSpec.rerank_runtime` removed.
  - `build_server_command(...)` keeps its signature; for a reranker it emits `-np S -c UB -b UB -ub UB … --reranking --kv-unified` and raises `ValueError` unless `context_per_slot == logical_batch_size == physical_batch_size`.

- [ ] **Step 1: Write the failing tests**

Replace the whole of `seed-pipeline/tests/runtime/test_model_profiles.py` with:

```python
import pytest

from seed_pipeline.runtime.catalog import RERANKER_MODELS, require_model
from seed_pipeline.runtime.model_profiles import (
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
)


def test_rerank_contract_rejects_other_protocols():
    with pytest.raises(ValueError, match="unsupported rerank protocol"):
        RerankContract(protocol="completion_logprobs")


def test_embedding_profile_separates_query_and_corpus_workloads():
    profile = require_model("qwen3-embedding:4b-fp16").embedding_runtime

    assert isinstance(profile, EmbeddingRuntimeProfile)
    assert isinstance(profile.query, EmbeddingWorkloadProfile)
    assert isinstance(profile.corpus, EmbeddingWorkloadProfile)
    assert profile.query is not profile.corpus
    assert profile.query.production_batch_size > 0
    assert profile.corpus.production_batch_size > 0


def test_embedding_model_exposes_separate_runtime_search_spaces():
    search_space = require_model("qwen3-embedding:4b-fp16").embedding_search_space

    assert search_space is not None
    assert search_space.query != search_space.corpus
    assert all(
        candidate.request_batch_size > 0 for candidate in search_space.query.candidates
    )


@pytest.mark.parametrize(
    ("model", "server_slots", "ubatch_sizes", "concurrency"),
    [
        ("qwen3-reranker:0.6b-fp16", 64, (8192, 16384, 32768), 4),
        ("qwen3-reranker:4b-fp16", 32, (8192, 16384), 3),
        ("qwen3-reranker:8b-fp16", 16, (4096, 8192), 2),
        ("bge-reranker-v2-m3:f16", 64, (8192, 16384, 32768), 4),
    ],
)
def test_kaggle_rerank_search_space_sweeps_the_ubatch(
    model: str, server_slots: int, ubatch_sizes: tuple[int, ...], concurrency: int
):
    space = require_model(model).rerank_search_space

    assert space is not None
    assert tuple(item.physical_batch_size for item in space.candidates) == ubatch_sizes
    for item in space.candidates:
        assert (
            item.server_slots,
            item.request_batch_size,
            item.concurrency,
            item.threads,
        ) == (server_slots, 30, concurrency, None)
        assert item.context_per_slot == item.logical_batch_size
        assert item.logical_batch_size == item.physical_batch_size


@pytest.mark.parametrize("model", sorted(RERANKER_MODELS))
def test_local_rerank_search_space_sweeps_ubatch_and_threads(model: str):
    space = require_model(model).local_rerank_search_space

    assert space is not None
    assert [(item.physical_batch_size, item.threads) for item in space.candidates] == [
        (4096, 8),
        (4096, 12),
        (8192, 8),
        (8192, 12),
        (16384, 8),
        (16384, 12),
    ]
    for item in space.candidates:
        assert (item.server_slots, item.request_batch_size, item.concurrency) == (
            16,
            15,
            1,
        )
        assert item.context_per_slot == item.logical_batch_size
        assert item.logical_batch_size == item.physical_batch_size
```

In `seed-pipeline/tests/runtime/test_runtime_profiles.py` add `from dataclasses import replace` to the imports, add `rerank_concurrency` and `reranker_candidate` to the `seed_pipeline.runtime.runtime_profiles` import list, and append:

```python
def test_candidate_threads_are_optional_in_the_payload():
    assert "threads" not in candidate().to_dict()
    local = replace(candidate(), threads=12)

    assert local.to_dict()["threads"] == 12
    assert RuntimeCandidate.from_dict(local.to_dict()) == local
    assert RuntimeCandidate.from_dict(candidate().to_dict()) == candidate()


@pytest.mark.parametrize(
    ("server_slots", "request_batch_size", "expected"),
    [(64, 30, 4), (32, 30, 3), (16, 30, 2), (16, 15, 3)],
)
def test_rerank_concurrency_keeps_every_slot_busy(
    server_slots: int, request_batch_size: int, expected: int
):
    assert rerank_concurrency(server_slots, request_batch_size) == expected


def test_reranker_candidate_uses_one_size_for_context_batch_and_ubatch():
    level = reranker_candidate(
        server_slots=32, ubatch=16384, request_batch_size=30, concurrency=3
    )

    assert (
        level.context_per_slot,
        level.logical_batch_size,
        level.physical_batch_size,
    ) == (16384, 16384, 16384)


def test_reranker_candidate_rejects_ubatch_below_the_longest_prompt():
    with pytest.raises(ValueError, match="2048"):
        reranker_candidate(
            server_slots=4, ubatch=1024, request_batch_size=30, concurrency=1
        )
```

In `seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py` replace `test_build_server_command_accepts_benchmark_runtime_overrides` with:

```python
RERANK_LEVEL = {
    "server_slots": 64,
    "concurrency": 4,
    "request_batch_size": 30,
    "context_per_slot": 16384,
    "logical_batch_size": 16384,
    "physical_batch_size": 16384,
}


def test_embedding_command_sizes_context_per_slot():
    command = build_server_command(
        binary="llama-server",
        model="model.gguf",
        port=11434,
        visible_devices="0",
        spec=require_model("bge-m3:567m-fp16"),
        runtime_overrides={
            "server_slots": 8,
            "context_per_slot": 4096,
            "logical_batch_size": 8192,
            "physical_batch_size": 4096,
        },
    )

    assert command[command.index("-np") + 1] == "8"
    assert command[command.index("-c") + 1] == "32768"
    assert command[command.index("-b") + 1] == "8192"
    assert command[command.index("-ub") + 1] == "4096"
    assert "--embedding" in command
    assert "--reranking" not in command
    assert "--kv-unified" not in command


@pytest.mark.parametrize(
    "model",
    ("qwen3-reranker:0.6b-fp16", "qwen3-reranker:8b-fp16", "bge-reranker-v2-m3:f16"),
)
def test_reranker_command_batches_documents_in_one_unified_kv_pool(model):
    command = build_server_command(
        binary="llama-server",
        model="model.gguf",
        port=11434,
        visible_devices="0",
        spec=require_model(model),
        runtime_overrides=RERANK_LEVEL,
    )

    assert "--reranking" in command
    assert "--kv-unified" in command
    assert command[command.index("-np") + 1] == "64"
    for flag in ("-c", "-b", "-ub"):
        assert command[command.index(flag) + 1] == "16384"


def test_sharded_reranker_splits_one_server_over_two_gpus():
    command = build_server_command(
        binary="llama-server",
        model="model.gguf",
        port=11434,
        visible_devices="0,1",
        spec=require_model("qwen3-reranker:8b-fp16"),
        runtime_overrides=RERANK_LEVEL,
    )

    assert command[command.index("--tensor-split") + 1] == "1,1"


def test_reranker_command_rejects_split_context_and_ubatch():
    with pytest.raises(ValueError, match="one size"):
        build_server_command(
            binary="llama-server",
            model="model.gguf",
            port=11434,
            visible_devices="0",
            spec=require_model("qwen3-reranker:0.6b-fp16"),
            runtime_overrides={**RERANK_LEVEL, "physical_batch_size": 8192},
        )
```

In the same file, set `"physical_batch_size": 4096` (was `2048`) in the `runtime_overrides` of `test_build_server_command_enforces_stateless_prompt_cache` and in the four `managed_model_servers` tests (the dicts starting at lines 300, 361, 429 and 487), so every reranker level uses one size.

Append to the `("module", "name")` list of `test_completion_logprobs_code_is_gone` in `seed-pipeline/tests/test_removed_contracts.py`:

```python
(("seed_pipeline.runtime.model_profiles", "RerankRuntimeProfile"),)
(("seed_pipeline.runtime.benchmarking", "rerank_levels"),)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_model_profiles.py tests/runtime/test_runtime_profiles.py tests/integrations/kaggle/test_worker_runtime.py tests/test_removed_contracts.py`
Expected: FAIL — `ImportError: cannot import name 'rerank_concurrency'`, search-space assertions (0.6b has `server_slots == 4`), `--kv-unified` missing, `-c` equal to `65536`.

- [ ] **Step 3: Add threads and the reranker helpers to `runtime_profiles.py`**

Add `import math` to the imports of `seed-pipeline/src/seed_pipeline/runtime/runtime_profiles.py` and replace the whole `RuntimeCandidate` class (lines 38–86) with:

```python
@dataclass(frozen=True)
class RuntimeCandidate:
    server_slots: int
    concurrency: int
    request_batch_size: int
    context_per_slot: int
    logical_batch_size: int
    physical_batch_size: int
    # CPU threads of a local llama-server; Kaggle GPU levels leave it unset.
    threads: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.server_slots,
            self.concurrency,
            self.request_batch_size,
            self.context_per_slot,
            self.logical_batch_size,
            self.physical_batch_size,
        )
        if any(value < 1 for value in values):
            raise ValueError("runtime candidate values must be positive")
        if self.threads is not None and self.threads < 1:
            raise ValueError("runtime candidate threads must be positive")

    def to_dict(self) -> dict[str, int]:
        payload = {
            "server_slots": self.server_slots,
            "concurrency": self.concurrency,
            "request_batch_size": self.request_batch_size,
            "context_per_slot": self.context_per_slot,
            "logical_batch_size": self.logical_batch_size,
            "physical_batch_size": self.physical_batch_size,
        }
        if self.threads is not None:
            payload["threads"] = self.threads
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeCandidate:
        names = (
            "server_slots",
            "concurrency",
            "request_batch_size",
            "context_per_slot",
            "logical_batch_size",
            "physical_batch_size",
        )
        missing = [name for name in names if name not in payload]
        if missing:
            raise ValueError(f"runtime candidate is missing: {', '.join(missing)}")
        try:
            values = {name: _to_int(payload[name]) for name in names}
            raw_threads = payload.get("threads")
            threads = None if raw_threads is None else _to_int(raw_threads)
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime candidate values must be integers") from exc
        return cls(**values, threads=threads)


MIN_RERANK_UBATCH = 2048


def rerank_concurrency(server_slots: int, request_batch_size: int) -> int:
    """Concurrent requests per server that keep every reranker slot busy."""
    if server_slots < 1 or request_batch_size < 1:
        raise ValueError("server_slots and request_batch_size must be positive")
    return math.ceil(server_slots / request_batch_size) + 1


def reranker_candidate(
    *,
    server_slots: int,
    ubatch: int,
    request_batch_size: int,
    concurrency: int,
    threads: int | None = None,
) -> RuntimeCandidate:
    """One reranker server level: -np server_slots and -c = -b = -ub = ubatch.

    Rank pooling computes each document in a single pass, so a document must fit in one
    ubatch; the longest rerank prompt is 1,664 tokens.
    """
    if ubatch < MIN_RERANK_UBATCH:
        raise ValueError(f"reranker ubatch must be at least {MIN_RERANK_UBATCH} tokens")
    return RuntimeCandidate(
        server_slots=server_slots,
        concurrency=concurrency,
        request_batch_size=request_batch_size,
        context_per_slot=ubatch,
        logical_batch_size=ubatch,
        physical_batch_size=ubatch,
        threads=threads,
    )
```

- [ ] **Step 4: Rebuild the reranker catalog entries**

In `seed-pipeline/src/seed_pipeline/runtime/catalog.py` replace both import blocks with:

```python
from seed_pipeline.runtime.model_profiles import (
    NATIVE_RERANK_PROTOCOL,
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
    native_rerank_contract,
)
from seed_pipeline.runtime.runtime_profiles import (
    EmbeddingRuntimeSearchSpaces,
    RuntimeCandidate,
    RuntimeSearchSpace,
    rerank_concurrency,
    reranker_candidate,
)
```

In `ModelSpec` delete the field `rerank_runtime: RerankRuntimeProfile | None = None` and add after `rerank_search_space`:

```python
    local_rerank_search_space: RuntimeSearchSpace | None = None
```

Insert above `class ModelSpec`:

```python
KAGGLE_RERANK_REQUEST_BATCH_SIZE = 30
LOCAL_RERANK_REQUEST_BATCH_SIZE = 15
# Local CPU llama-reranker levels. The backend sends one request with at most 15
# candidates at a time, so a level has a single client request in flight.
LOCAL_RERANK_SEARCH_SPACE = RuntimeSearchSpace(
    tuple(
        reranker_candidate(
            server_slots=16,
            ubatch=ubatch,
            request_batch_size=LOCAL_RERANK_REQUEST_BATCH_SIZE,
            concurrency=1,
            threads=threads,
        )
        for ubatch in (4096, 8192, 16384)
        for threads in (8, 12)
    )
)
```

Replace the whole `_reranker` function with:

```python
def _reranker(
    name: str,
    filename: str,
    size: int,
    sha256: str,
    topology: ModelTopology,
    *,
    server_slots: int,
    ubatch_sizes: tuple[int, ...],
) -> ModelSpec:
    concurrency = rerank_concurrency(server_slots, KAGGLE_RERANK_REQUEST_BATCH_SIZE)
    search_space = RuntimeSearchSpace(
        tuple(
            reranker_candidate(
                server_slots=server_slots,
                ubatch=ubatch,
                request_batch_size=KAGGLE_RERANK_REQUEST_BATCH_SIZE,
                concurrency=concurrency,
            )
            for ubatch in ubatch_sizes
        )
    )
    smallest_ubatch = min(ubatch_sizes)
    return ModelSpec(
        name=name,
        kind=ModelKind.RERANKER,
        canonical_filename=filename,
        byte_size=size,
        sha256=sha256,
        topology=topology,
        kaggle_parallel=server_slots,
        kaggle_request_batch_size=KAGGLE_RERANK_REQUEST_BATCH_SIZE,
        kaggle_context_per_slot=smallest_ubatch,
        kaggle_logical_batch_size=smallest_ubatch,
        kaggle_physical_batch_size=smallest_ubatch,
        reranker_protocol=NATIVE_RERANK_PROTOCOL,
        rerank_contract=native_rerank_contract(),
        rerank_search_space=search_space,
        local_rerank_search_space=LOCAL_RERANK_SEARCH_SPACE,
    )
```

Replace `RERANKER_MODELS` with:

```python
RERANKER_MODELS = {
    "qwen3-reranker:0.6b-fp16": _reranker(
        "qwen3-reranker:0.6b-fp16",
        "qwen3-reranker-0.6b-f16.gguf",
        1_197_634_304,
        "fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa",
        ModelTopology.REPLICATED_2X1,
        server_slots=64,
        ubatch_sizes=(8192, 16384, 32768),
    ),
    "qwen3-reranker:4b-fp16": _reranker(
        "qwen3-reranker:4b-fp16",
        "qwen3-reranker-4b-f16.gguf",
        8_049_922_912,
        "c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117",
        ModelTopology.REPLICATED_2X1,
        server_slots=32,
        ubatch_sizes=(8192, 16384),
    ),
    "qwen3-reranker:8b-fp16": _reranker(
        "qwen3-reranker:8b-fp16",
        "qwen3-reranker-8b-f16.gguf",
        15_141_207_744,
        "a53322f7936010458424a12f0f6d22291547e42fa85c16dd4730244d659cea96",
        ModelTopology.SHARDED_1X2,
        server_slots=16,
        ubatch_sizes=(4096, 8192),
    ),
    # XLM-R cross-encoder (568M, 8,192 positions) in the 0.6b size class; its inputs carry
    # no chat template, so it takes the 0.6b levels.
    "bge-reranker-v2-m3:f16": _reranker(
        "bge-reranker-v2-m3:f16",
        "bge-reranker-v2-m3-f16.gguf",
        1_159_774_912,
        "3c2de408d2c0a85a9472dc09f9d5a22c9b73743c6343952c15053299c777c298",
        ModelTopology.REPLICATED_2X1,
        server_slots=64,
        ubatch_sizes=(8192, 16384, 32768),
    ),
}
```

In `seed-pipeline/src/seed_pipeline/runtime/model_profiles.py` delete the class `RerankRuntimeProfile`. In `seed-pipeline/src/seed_pipeline/runtime/benchmarking.py` change the model-profile import to `from seed_pipeline.runtime.model_profiles import EmbeddingWorkloadProfile` and delete `def rerank_levels`.

- [ ] **Step 5: Emit the batched reranker command**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/runtime.py` replace the body of `build_server_command` after the `if runtime_overrides is None:` guard (lines 339–368) with:

```python
overrides = dict(runtime_overrides)
server_slots = overrides["server_slots"]
if spec.kind is ModelKind.RERANKER:
    ubatch = overrides["physical_batch_size"]
    if not (overrides["context_per_slot"] == overrides["logical_batch_size"] == ubatch):
        raise ValueError(
            "reranker runtime must use one size for context, batch and ubatch"
        )
    # Rank pooling frees a sequence's KV right after its single pass, so the unified
    # pool only holds the tokens computed together: -c equals -ub, not slots x -ub.
    context = ubatch
else:
    context = overrides["context_per_slot"] * server_slots
command = [
    str(binary),
    "--model",
    str(model),
    "--port",
    str(port),
    "--host",
    "127.0.0.1",
    "--offline",
    "--no-webui",
    "--n-gpu-layers",
    "99",
    "-np",
    str(server_slots),
    "-c",
    str(context),
    "-b",
    str(overrides["logical_batch_size"]),
    "-ub",
    str(overrides["physical_batch_size"]),
    *inference_cache_policy(spec).arguments(),
]
if spec.kind is ModelKind.EMBEDDING:
    command.append("--embedding")
else:
    command.extend(["--reranking", "--kv-unified"])
if spec.topology is ModelTopology.SHARDED_1X2:
    command.extend(["--tensor-split", "1,1"])
return command
```

- [ ] **Step 6: Run the tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime tests/integrations/kaggle tests/test_removed_contracts.py tests/evaluation`
Expected: PASS. `grep -rn 'rerank_runtime\|RerankRuntimeProfile\|rerank_levels' src tests` prints only the `test_removed_contracts.py` lines.

- [ ] **Step 7: Run the Seed gate**

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/runtime/runtime_profiles.py seed-pipeline/src/seed_pipeline/runtime/catalog.py seed-pipeline/src/seed_pipeline/runtime/model_profiles.py seed-pipeline/src/seed_pipeline/runtime/benchmarking.py seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/runtime.py seed-pipeline/tests/runtime/test_model_profiles.py seed-pipeline/tests/runtime/test_runtime_profiles.py seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py seed-pipeline/tests/test_removed_contracts.py
git diff --cached --name-status
git commit -m "feat(seed): serve rerankers from one unified KV pool per physical batch" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Reuse checkpoints across runtime profiles

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/models.py:186-194` (add `reuse_payload_of`, use it in `JobIdentity.reuse_payload`)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/artifacts.py:13-17` (import), `:172-180` (derived reuse hash)
- Test: `seed-pipeline/tests/integrations/kaggle/test_job_identity.py` (new)

**Interfaces:**
- Consumes: Task 5 rerank search space (two candidates for `qwen3-reranker:0.6b-fp16`).
- Produces:
  - `seed_pipeline.integrations.kaggle.models.reuse_payload_of(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]` — drops `input_sha256` and `runtime_parameters.runtime_profile`.
  - `JobIdentity.reuse_payload` / `JobIdentity.reuse_sha256` use it; `JobIdentity.sha256` still covers the runtime profile (one kernel per configuration).
  - `load_cloud_artifact` derives a manifest's reuse hash with the same function.

- [ ] **Step 1: Write the failing test**

Create `seed-pipeline/tests/integrations/kaggle/test_job_identity.py`:

```python
from collections.abc import Mapping
from pathlib import Path

from seed_pipeline.integrations.kaggle.artifacts import load_cloud_artifact
from seed_pipeline.integrations.kaggle.models import (
    JobIdentity,
    StageName,
    reuse_payload_of,
)
from seed_pipeline.integrations.kaggle.workers.runtime import artifact_from_output
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

MODEL = "qwen3-reranker:0.6b-fp16"


def _levels() -> tuple[RuntimeCandidate, RuntimeCandidate]:
    space = require_model(MODEL).rerank_search_space
    assert space is not None
    return space.candidates[0], space.candidates[1]


def _identity(
    profile: RuntimeCandidate,
    *,
    contract: str = "c" * 64,
    input_sha256: str = "b" * 64,
) -> JobIdentity:
    return JobIdentity.create(
        stage=StageName.RERANK,
        contract_version=3,
        model=MODEL,
        model_sha256="a" * 64,
        input_sha256=input_sha256,
        runtime_parameters={
            "protocol": "native_rerank",
            "request_contract_sha256": contract,
            "runtime_profile": profile.to_dict(),
        },
    )


def test_runtime_profile_changes_the_job_but_not_the_reuse_identity():
    first, second = _levels()
    one, other = _identity(first), _identity(second)

    assert one.sha256 != other.sha256
    assert one.reuse_sha256 == other.reuse_sha256
    parameters = one.reuse_payload["runtime_parameters"]
    assert isinstance(parameters, Mapping)
    assert "runtime_profile" not in parameters
    assert "input_sha256" not in one.reuse_payload


def test_reuse_identity_still_tracks_the_scoring_contract():
    first, _second = _levels()
    base = _identity(first)

    assert _identity(first, contract="d" * 64).reuse_sha256 != base.reuse_sha256
    assert _identity(first, input_sha256="e" * 64).reuse_sha256 == base.reuse_sha256


def test_reuse_payload_without_runtime_parameters_only_drops_the_input():
    assert reuse_payload_of({"stage": "rerank", "input_sha256": "b" * 64}) == {
        "stage": "rerank"
    }


def test_artifact_from_another_runtime_profile_is_reusable(tmp_path: Path):
    first, second = _levels()
    data = tmp_path / "rerank_scores.jsonl"
    data.write_text('{"score": 0.5}\n', encoding="utf-8")
    artifact = artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=_identity(first),
        total=1,
        complete=1,
    )

    loaded = load_cloud_artifact(
        data, artifact.manifest_path, _identity(second), allow_reuse=True
    )

    assert loaded.strict_identity_match is False
    assert loaded.completion.complete == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run (in `seed-pipeline/`): `uv run pytest -q tests/integrations/kaggle/test_job_identity.py`
Expected: FAIL — `ImportError: cannot import name 'reuse_payload_of'`.

- [ ] **Step 3: Implement the shared reuse payload**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/models.py` insert above `@dataclass(frozen=True) class JobIdentity`:

```python
def reuse_payload_of(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Identity fields that decide whether checkpointed records can be reused.

    Inputs are matched record by record and a runtime profile changes only speed, so
    neither belongs to the reuse identity.
    """
    reuse: dict[str, JSONValue] = {
        key: value for key, value in payload.items() if key != "input_sha256"
    }
    runtime_parameters = reuse.get("runtime_parameters")
    if isinstance(runtime_parameters, Mapping):
        reuse["runtime_parameters"] = {
            key: value
            for key, value in runtime_parameters.items()
            if key != "runtime_profile"
        }
    return reuse
```

and replace the body of `JobIdentity.reuse_payload` with `return reuse_payload_of(self.payload)`.

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/artifacts.py` add `reuse_payload_of` to the `seed_pipeline.integrations.kaggle.models` import and replace

```python
derived_reuse_sha256 = canonical_sha256(
    {key: value for key, value in identity_payload.items() if key != "input_sha256"}
)
```

with:

```python
        derived_reuse_sha256 = canonical_sha256(reuse_payload_of(identity_payload))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/integrations/kaggle`
Expected: PASS (including `test_load_rejects_manifest_reuse_hash_not_derived_from_identity` in `test_artifacts.py` and the checkpoint name tests in `test_checkpoints.py`).

- [ ] **Step 5: Run the Seed gate**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/models.py seed-pipeline/src/seed_pipeline/integrations/kaggle/artifacts.py seed-pipeline/tests/integrations/kaggle/test_job_identity.py
git diff --cached --name-status
git commit -m "feat(kaggle): reuse checkpoints across runtime profiles" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Score each Kaggle query group in one request

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py` (the `grouped`/`groups` block written in Task 2 Step 8)
- Test: `seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py`

**Interfaces:**
- Consumes: Task 5 `RuntimeCandidate.to_dict()` in `config["runtime_overrides"]` (`concurrency`, `request_batch_size`); Task 2 native-only worker.
- Produces: `run_rerank_worker(config, *, score_pair=None, emit=…, clock=…, server_manager=managed_model_servers, client_factory=None) -> CloudArtifact` unchanged in signature; it sends one `rerank_native` call per query with at most `runtime_overrides["request_batch_size"]` documents and runs `runtime_overrides["concurrency"]` requests per server over a shared queue. Plan B adds server restarts around this block.

- [ ] **Step 1: Write the failing tests**

Append to `seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py` (add `import seed_pipeline.integrations.kaggle.workers.rerank as rerank_worker` and `from tests.integrations.kaggle.factories import rerank_runtime_profile` to the imports):

```python
@contextmanager
def _two_server_manager(_config, **_kwargs):
    yield [
        SimpleNamespace(base_url="http://127.0.0.1:11434"),
        SimpleNamespace(base_url="http://127.0.0.1:11435"),
    ]


class _RecordingClient:
    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def rerank_native(self, query, documents, _model):
        self.calls.append((query, list(documents)))
        return [0.5 for _document in documents]


def test_rerank_worker_sends_one_request_per_query_with_derived_concurrency(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, query_count=3, candidates_per_query=30)
    config["runtime_overrides"] = rerank_runtime_profile(
        "qwen3-reranker:0.6b-fp16"
    ).to_dict()
    client = _RecordingClient()
    captured = {}
    real_scheduler = rerank_worker.stream_map_ordered

    async def recording_scheduler(items, resources, concurrency, *args, **kwargs):
        captured["concurrency"] = concurrency
        captured["servers"] = len(resources)
        return await real_scheduler(items, resources, concurrency, *args, **kwargs)

    monkeypatch.setattr(rerank_worker, "stream_map_ordered", recording_scheduler)

    artifact = run_rerank_worker(
        config,
        server_manager=_two_server_manager,
        client_factory=lambda _url: client,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )

    assert artifact.completion.complete == 90
    assert sorted(query for query, _documents in client.calls) == [
        "query 0",
        "query 1",
        "query 2",
    ]
    assert all(len(documents) == 30 for _query, documents in client.calls)
    # ceil(64 slots / 30 documents) + 1 requests per server keep every slot busy.
    assert captured == {"concurrency": 4, "servers": 2}


def test_rerank_worker_splits_a_query_larger_than_the_request_batch(tmp_path):
    config = _config(tmp_path, query_count=1, candidates_per_query=5)
    config["runtime_overrides"] = {
        **rerank_runtime_profile("qwen3-reranker:0.6b-fp16").to_dict(),
        "request_batch_size": 2,
    }
    client = _RecordingClient()

    artifact = run_rerank_worker(
        config,
        server_manager=_fake_server_manager,
        client_factory=lambda _url: client,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )

    assert artifact.completion.complete == 5
    assert sorted(len(documents) for _query, documents in client.calls) == [1, 2, 2]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/integrations/kaggle/test_worker_rerank.py`
Expected: `test_rerank_worker_splits_a_query_larger_than_the_request_batch` FAILS (`[5] != [1, 2, 2]`); the concurrency test passes already and guards the formula.

- [ ] **Step 3: Split query groups by the request batch size**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py` replace

```python
per_client = max(1, int(runtime_overrides["concurrency"]))
grouped: dict[tuple[str, str], list[Pair]] = {}
for pair in missing:
    query, _text = details[_pair_key(pair)]
    grouped.setdefault((pair["query_id"], query), []).append(pair)
groups = [(query_id, query, group) for (query_id, query), group in grouped.items()]
```

with:

```python
per_client = max(1, int(runtime_overrides["concurrency"]))
request_batch_size = max(1, int(runtime_overrides["request_batch_size"]))
grouped: dict[tuple[str, str], list[Pair]] = {}
for pair in missing:
    query, _text = details[_pair_key(pair)]
    grouped.setdefault((pair["query_id"], query), []).append(pair)
# One /v1/rerank request per query; llama-server packs the documents of
# all busy slots into shared physical batches.
groups = [
    (query_id, query, group[start : start + request_batch_size])
    for (query_id, query), group in grouped.items()
    for start in range(0, len(group), request_batch_size)
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/integrations/kaggle/test_worker_rerank.py`
Expected: PASS, including `test_rerank_worker_seals_partial_artifact_after_server_failure` (partial artifact when the server drops).

- [ ] **Step 5: Run the Seed gate**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py
git diff --cached --name-status
git commit -m "feat(seed): send one /v1/rerank request per query group on Kaggle" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: End-to-end benchmark of full runtime candidates

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/runtime/benchmarking.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/benchmark.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/stages.py:17` (import), `:263-343` (`BenchmarkStage`)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py:1-20` (imports), `:49-118` (`_load_benchmark_selection`), `:194-203` (profile creation)
- Modify: `seed-pipeline/docs/guides/cli-reference.md` (section `## Runtime profiling trên Kaggle`)
- Test: `seed-pipeline/tests/runtime/test_benchmarking.py` (whole file), `seed-pipeline/tests/integrations/kaggle/test_benchmark_worker.py` (whole file), `seed-pipeline/tests/integrations/kaggle/test_stages.py`, `seed-pipeline/tests/integrations/kaggle/test_auto_profile.py`

**Interfaces:**
- Consumes: Task 5 `RuntimeCandidate`, `reranker_candidate`, search spaces; Task 2 `LlamaCppClient.rerank_native`.
- Produces (`seed_pipeline.runtime.benchmarking`):
  - `KAGGLE_RERANK_BENCHMARK_GROUPS = 32`, `LOCAL_RERANK_BENCHMARK_GROUPS = 6`, `SCORE_MISMATCH_TOLERANCE = 1e-3`, `LOG_TAIL_CHARACTERS = 16_000`, `BenchmarkObjective = Literal["throughput", "latency"]`
  - `BenchmarkMeasurement(candidate: RuntimeCandidate, items: int, input_characters: int, elapsed_seconds: float, status: str = "ok", error_category: str | None = None, latency_p50_seconds: float | None = None, latency_p95_seconds: float | None = None, max_abs_score_delta: float | None = None, log_tail: str | None = None)` with `items_per_second`, `characters_per_second`, `to_dict() -> dict[str, object]`, `from_dict(payload: Mapping[str, object])`, `invalid(candidate, error_category, log_tail=None)`
  - `BenchmarkWorkload(stage: str, model: str, sample_count: int, levels: tuple[RuntimeCandidate, ...])`, `BenchmarkReport(workload, measurements, recommendation: RuntimeCandidate | None, schema_version: int = 2)`
  - `RerankGroup(query_id: str, query: str, chunk_ids: tuple[str, ...], documents: tuple[str, ...])` with `characters: int` and `score_keys() -> tuple[str, ...]`
  - `rerank_groups_from_rows(rows: Iterable[Mapping[str, Any]], *, documents_per_group: int) -> list[RerankGroup]`
  - `sample_rerank_groups(groups: Sequence[RerankGroup], count: int) -> tuple[RerankGroup, ...]`
  - `LevelResult(measurement: BenchmarkMeasurement, scores: Mapping[str, float])`
  - `check_score_consistency(results: Sequence[LevelResult], *, tolerance: float = SCORE_MISMATCH_TOLERANCE) -> tuple[BenchmarkMeasurement, ...]`
  - `recommend(measurements: Sequence[BenchmarkMeasurement], *, objective: BenchmarkObjective) -> RuntimeCandidate | None`
  - `percentile(values: Sequence[float], fraction: float) -> float | None`, `describe_levels(measurements: Sequence[BenchmarkMeasurement]) -> str`, `stratified_sample` (unchanged)
  - Removed: `BenchmarkLevel`, `embedding_levels`.
- Produces (Kaggle): `run_benchmark_worker(config: dict, *, measure_level: Callable[[int, RuntimeCandidate], LevelResult] | None = None, clock: Callable[[], float] = time.monotonic) -> CloudArtifact`; benchmark rows `{"identity", "schema_version": 2, "measurement": BenchmarkMeasurement.to_dict()}`; manifest `runtime = {"recommendation": RuntimeCandidate.to_dict() | None, "sample_count": int}`; per-level server logs under `<output_dir>/level-<index>/server-<n>.log`. `BenchmarkStage.contract_version == 2`; worker config and identity carry `benchmark_levels` (candidate dicts), `benchmark_items`, and for rerankers `benchmark_groups`. `ensure_runtime_profile` stores the recommended candidate as `RuntimeProfile.selected` with `sample_count` from the manifest, and raises `RuntimeError` listing every level's status and log tail when no level is valid.

- [ ] **Step 1: Write the failing benchmark-core tests**

Replace the whole of `seed-pipeline/tests/runtime/test_benchmarking.py` with:

```python
import math

import pytest

from seed_pipeline.runtime.benchmarking import (
    BenchmarkMeasurement,
    LevelResult,
    RerankGroup,
    check_score_consistency,
    describe_levels,
    percentile,
    recommend,
    rerank_groups_from_rows,
    sample_rerank_groups,
    stratified_sample,
)
from seed_pipeline.runtime.runtime_profiles import reranker_candidate

LEVELS = tuple(
    reranker_candidate(
        server_slots=64, ubatch=ubatch, request_batch_size=30, concurrency=4
    )
    for ubatch in (8192, 16384, 32768)
)


def _measurement(
    level: int, *, seconds: float = 10.0, p95: float = 1.0
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        LEVELS[level],
        960,
        100_000,
        seconds,
        latency_p50_seconds=p95 / 2,
        latency_p95_seconds=p95,
    )


def _group(query_id: str, document_length: int) -> RerankGroup:
    return RerankGroup(
        query_id,
        "q",
        (f"{query_id}-0", f"{query_id}-1"),
        ("x" * document_length, "x" * document_length),
    )


def test_stratified_sample_is_stable_and_spans_lengths():
    items = [{"id": i, "length": i} for i in range(1, 101)]
    first = stratified_sample(
        items, 20, key=lambda x: x["id"], length=lambda x: x["length"]
    )
    second = stratified_sample(
        list(reversed(items)), 20, key=lambda x: x["id"], length=lambda x: x["length"]
    )
    assert [x["id"] for x in first] == [x["id"] for x in second]
    assert min(x["length"] for x in first) <= 5
    assert max(x["length"] for x in first) >= 95


def test_sample_rerank_groups_takes_one_group_per_character_stratum():
    groups = [_group(f"q{index:03d}", index) for index in range(1, 101)]

    first = sample_rerank_groups(groups, 6)

    assert first == sample_rerank_groups(list(reversed(groups)), 6)
    assert [len(group.documents[0]) for group in first] == [9, 26, 42, 59, 76, 92]


def test_sample_rerank_groups_returns_every_group_when_there_are_few():
    groups = [_group("b", 5), _group("a", 1)]

    assert sample_rerank_groups(groups, 6) == (groups[1], groups[0])


def test_rerank_groups_keep_only_full_query_groups():
    rows = [
        {
            "query_id": "q1",
            "query": "một",
            "candidates": [
                {"chunk_id": f"c{index}", "document_text": f"d{index}"}
                for index in range(3)
            ],
        },
        {
            "query_id": "q2",
            "query": "hai",
            "candidates": [{"chunk_id": "c0", "document_text": "d0"}],
        },
    ]

    groups = rerank_groups_from_rows(rows, documents_per_group=2)

    assert groups == [RerankGroup("q1", "một", ("c0", "c1"), ("d0", "d1"))]
    assert groups[0].score_keys() == ("q1\x1fc0", "q1\x1fc1")
    assert groups[0].characters == 2 * len("một") + 4


def test_throughput_objective_picks_the_most_pairs_per_second():
    measurements = [
        _measurement(0, seconds=12.0),
        _measurement(1, seconds=8.0),
        _measurement(2, seconds=9.0),
    ]

    assert recommend(measurements, objective="throughput") == LEVELS[1]


def test_latency_objective_picks_the_lowest_p95_then_throughput():
    measurements = [
        _measurement(0, seconds=9.0, p95=0.8),
        _measurement(1, seconds=8.0, p95=0.8),
        _measurement(2, seconds=5.0, p95=0.9),
    ]

    assert recommend(measurements, objective="latency") == LEVELS[1]


def test_recommend_skips_invalid_levels_and_needs_one_valid_level():
    invalid = BenchmarkMeasurement.invalid(
        LEVELS[1], "ModelServerExited", "CUDA out of memory"
    )

    assert (
        recommend([_measurement(0, seconds=12.0), invalid], objective="throughput")
        == LEVELS[0]
    )
    assert recommend([invalid], objective="throughput") is None
    assert recommend([invalid], objective="latency") is None


def test_score_check_compares_every_level_with_the_first_valid_level():
    results = [
        LevelResult(BenchmarkMeasurement.invalid(LEVELS[0], "RuntimeError"), {}),
        LevelResult(_measurement(1), {"q\x1fa": 0.9, "q\x1fb": 0.1}),
        LevelResult(_measurement(2), {"q\x1fa": 0.9005, "q\x1fb": 0.1}),
        LevelResult(_measurement(0), {"q\x1fa": 0.902, "q\x1fb": 0.1}),
    ]

    checked = check_score_consistency(results)

    assert [(item.status, item.error_category) for item in checked] == [
        ("invalid", "RuntimeError"),
        ("ok", None),
        ("ok", None),
        ("invalid", "score_mismatch"),
    ]
    assert checked[1].max_abs_score_delta == 0.0
    assert checked[2].max_abs_score_delta == pytest.approx(5e-4)
    assert checked[3].max_abs_score_delta == pytest.approx(2e-3)


def test_score_check_rejects_levels_that_scored_different_pairs():
    checked = check_score_consistency(
        [
            LevelResult(_measurement(0), {"q\x1fa": 0.5}),
            LevelResult(_measurement(1), {"q\x1fb": 0.5}),
        ]
    )

    assert (
        checked[1].status,
        checked[1].error_category,
        checked[1].max_abs_score_delta,
    ) == ("invalid", "score_mismatch", None)


def test_percentile_interpolates_between_ranks():
    assert percentile([], 0.95) is None
    assert percentile([2.0], 0.95) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_measurement_payload_round_trips_the_full_candidate():
    measurement = BenchmarkMeasurement.invalid(
        LEVELS[2], "ModelServerExited", "x" * 20_000
    )

    payload = measurement.to_dict()

    assert payload["candidate"] == LEVELS[2].to_dict()
    assert BenchmarkMeasurement.from_dict(payload) == measurement
    assert measurement.log_tail is not None
    assert len(measurement.log_tail) == 16_000
    assert math.isinf(measurement.items_per_second)


def test_describe_levels_lists_status_error_and_log_tail():
    text = describe_levels(
        [
            BenchmarkMeasurement.invalid(
                LEVELS[0], "ModelServerExited", "CUDA out of memory"
            )
        ]
    )

    assert '"physical_batch_size": 8192' in text
    assert "status=invalid error=ModelServerExited" in text
    assert "CUDA out of memory" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_benchmarking.py`
Expected: FAIL — `ImportError: cannot import name 'LevelResult'`.

- [ ] **Step 3: Rewrite the benchmark core**

Replace the whole of `seed-pipeline/src/seed_pipeline/runtime/benchmarking.py` with:

```python
"""Deterministic, side-effect-free runtime benchmark primitives."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

KAGGLE_RERANK_BENCHMARK_GROUPS = 32
LOCAL_RERANK_BENCHMARK_GROUPS = 6
SCORE_MISMATCH_TOLERANCE = 1e-3
LOG_TAIL_CHARACTERS = 16_000

BenchmarkObjective = Literal["throughput", "latency"]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError("benchmark measurement number is invalid")


def _count(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError("benchmark measurement count is invalid")


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class BenchmarkWorkload:
    stage: str
    model: str
    sample_count: int
    levels: tuple[RuntimeCandidate, ...]


@dataclass(frozen=True)
class BenchmarkMeasurement:
    candidate: RuntimeCandidate
    items: int
    input_characters: int
    elapsed_seconds: float
    status: str = "ok"
    error_category: str | None = None
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None
    max_abs_score_delta: float | None = None
    log_tail: str | None = None

    def __post_init__(self) -> None:
        if self.items < 0 or self.input_characters < 0:
            raise ValueError("benchmark counts must be non-negative")
        for name, value in (
            ("elapsed_seconds", self.elapsed_seconds),
            ("latency_p50_seconds", self.latency_p50_seconds),
            ("latency_p95_seconds", self.latency_p95_seconds),
            ("max_abs_score_delta", self.max_abs_score_delta),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def invalid(
        cls,
        candidate: RuntimeCandidate,
        error_category: str,
        log_tail: str | None = None,
    ) -> BenchmarkMeasurement:
        return cls(
            candidate,
            0,
            0,
            0.0,
            status="invalid",
            error_category=error_category,
            log_tail=log_tail[-LOG_TAIL_CHARACTERS:] if log_tail else None,
        )

    @property
    def items_per_second(self) -> float:
        return self.items / self.elapsed_seconds if self.elapsed_seconds else math.inf

    @property
    def characters_per_second(self) -> float:
        return (
            self.input_characters / self.elapsed_seconds
            if self.elapsed_seconds
            else math.inf
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "items": self.items,
            "input_characters": self.input_characters,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
            "error_category": self.error_category,
            "latency_p50_seconds": self.latency_p50_seconds,
            "latency_p95_seconds": self.latency_p95_seconds,
            "max_abs_score_delta": self.max_abs_score_delta,
            "log_tail": self.log_tail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BenchmarkMeasurement:
        candidate = payload.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("benchmark measurement has no candidate")
        elapsed = _optional_float(payload.get("elapsed_seconds"))
        if elapsed is None:
            raise ValueError("benchmark measurement has no elapsed time")
        return cls(
            RuntimeCandidate.from_dict(candidate),
            items=_count(payload.get("items")),
            input_characters=_count(payload.get("input_characters")),
            elapsed_seconds=elapsed,
            status=str(payload.get("status", "")),
            error_category=_optional_text(payload.get("error_category")),
            latency_p50_seconds=_optional_float(payload.get("latency_p50_seconds")),
            latency_p95_seconds=_optional_float(payload.get("latency_p95_seconds")),
            max_abs_score_delta=_optional_float(payload.get("max_abs_score_delta")),
            log_tail=_optional_text(payload.get("log_tail")),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    workload: BenchmarkWorkload
    measurements: tuple[BenchmarkMeasurement, ...]
    recommendation: RuntimeCandidate | None
    schema_version: int = 2


@dataclass(frozen=True)
class RerankGroup:
    """One /v1/rerank request: a query with its first candidates."""

    query_id: str
    query: str
    chunk_ids: tuple[str, ...]
    documents: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.documents or len(self.chunk_ids) != len(self.documents):
            raise ValueError("rerank group needs one chunk id per document")

    @property
    def characters(self) -> int:
        return sum(len(self.query) + len(document) for document in self.documents)

    def score_keys(self) -> tuple[str, ...]:
        return tuple(f"{self.query_id}\x1f{chunk_id}" for chunk_id in self.chunk_ids)


def rerank_groups_from_rows(
    rows: Iterable[Mapping[str, Any]], *, documents_per_group: int
) -> list[RerankGroup]:
    """Full query groups (exactly `documents_per_group` candidates) from candidate rows."""
    if documents_per_group < 1:
        raise ValueError("documents_per_group must be positive")
    groups: list[RerankGroup] = []
    for row in rows:
        candidates = list(row["candidates"])[:documents_per_group]
        if len(candidates) < documents_per_group:
            continue
        groups.append(
            RerankGroup(
                str(row["query_id"]),
                str(row.get("query", "")),
                tuple(str(item["chunk_id"]) for item in candidates),
                tuple(str(item.get("document_text", "")) for item in candidates),
            )
        )
    return groups


def sample_rerank_groups(
    groups: Sequence[RerankGroup], count: int
) -> tuple[RerankGroup, ...]:
    """The middle group of `count` equal strata ordered by total characters."""
    if count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(groups, key=lambda group: (group.characters, group.query_id))
    if len(ordered) <= count:
        return tuple(ordered)
    return tuple(
        ordered[(2 * index + 1) * len(ordered) // (2 * count)] for index in range(count)
    )


def stratified_sample(items, count: int, key, length) -> tuple:
    if count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(items, key=key)
    if not ordered:
        return ()
    strata: list[list] = [[] for _ in range(8)]
    lengths = [length(item) for item in ordered]
    low, high = min(lengths), max(lengths)
    span = max(high - low + 1, 1)
    for item, item_length in zip(ordered, lengths, strict=True):
        index = min(7, max(0, int((item_length - low) * 8 / span)))
        strata[index].append(item)
    nonempty = [bucket for bucket in strata if bucket]
    target = min(count, len(ordered))
    quotient, remainder = divmod(target, len(nonempty))
    selected = []
    for index, bucket in enumerate(nonempty):
        take = quotient + (index < remainder)
        if take >= len(bucket):
            selected.extend(bucket)
            continue
        for offset in range(take):
            selected.append(bucket[(offset * len(bucket)) // take])
    return tuple(sorted(selected, key=key))


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class LevelResult:
    measurement: BenchmarkMeasurement
    # "<query_id>\x1f<chunk_id>" -> score; empty for embeddings and failed levels.
    scores: Mapping[str, float]


def _max_abs_delta(
    baseline: Mapping[str, float], scores: Mapping[str, float]
) -> float | None:
    if set(baseline) != set(scores):
        return None
    return max((abs(scores[key] - baseline[key]) for key in baseline), default=0.0)


def check_score_consistency(
    results: Sequence[LevelResult], *, tolerance: float = SCORE_MISMATCH_TOLERANCE
) -> tuple[BenchmarkMeasurement, ...]:
    """Mark levels whose scores drift from the first valid level as score_mismatch."""
    baseline: Mapping[str, float] | None = None
    checked: list[BenchmarkMeasurement] = []
    for result in results:
        measurement = result.measurement
        if measurement.status != "ok" or not result.scores:
            checked.append(measurement)
            continue
        if baseline is None:
            baseline = result.scores
            checked.append(replace(measurement, max_abs_score_delta=0.0))
            continue
        delta = _max_abs_delta(baseline, result.scores)
        if delta is None or delta > tolerance:
            checked.append(
                replace(
                    measurement,
                    status="invalid",
                    error_category="score_mismatch",
                    max_abs_score_delta=delta,
                )
            )
        else:
            checked.append(replace(measurement, max_abs_score_delta=delta))
    return tuple(checked)


def recommend(
    measurements: Sequence[BenchmarkMeasurement], *, objective: BenchmarkObjective
) -> RuntimeCandidate | None:
    valid = [
        item
        for item in measurements
        if item.status == "ok"
        and item.items > 0
        and math.isfinite(item.items_per_second)
    ]
    if objective == "throughput":
        if not valid:
            return None
        return max(valid, key=lambda item: item.items_per_second).candidate
    timed = [
        (item.latency_p95_seconds, item)
        for item in valid
        if item.latency_p95_seconds is not None
    ]
    if not timed:
        return None
    return min(timed, key=lambda pair: (pair[0], -pair[1].items_per_second))[
        1
    ].candidate


def describe_levels(measurements: Sequence[BenchmarkMeasurement]) -> str:
    lines: list[str] = []
    for item in measurements:
        lines.append(
            f"- {json.dumps(item.candidate.to_dict(), sort_keys=True)} "
            f"status={item.status} error={item.error_category}"
        )
        if item.log_tail:
            lines.append(item.log_tail)
    return "\n".join(lines)
```

- [ ] **Step 4: Run the core tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_benchmarking.py`
Expected: PASS. (The Kaggle worker, stage and auto-profile still import `BenchmarkLevel`; Steps 5–9 update them before the gate.)

- [ ] **Step 5: Write the failing Kaggle benchmark tests**

Replace the whole of `seed-pipeline/tests/integrations/kaggle/test_benchmark_worker.py` with:

```python
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import seed_pipeline.integrations.kaggle.workers.benchmark as benchmark_worker
from seed_pipeline.integrations.kaggle.artifacts import sha256_file
from seed_pipeline.integrations.kaggle.workers.benchmark import run_benchmark_worker
from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement, LevelResult
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

MODEL = "qwen3-reranker:0.6b-fp16"


def _levels() -> tuple[RuntimeCandidate, ...]:
    space = require_model(MODEL).rerank_search_space
    assert space is not None
    return space.candidates


def _config(tmp_path: Path, **extra: object) -> dict:
    return {
        "output_dir": str(tmp_path / "output"),
        "model": MODEL,
        "stage": "rerank-benchmark",
        "benchmark_items": 960,
        "benchmark_groups": 32,
        "benchmark_levels": [level.to_dict() for level in _levels()],
        "identity": {"stage": "rerank-benchmark"},
        "job_sha256": "a" * 64,
        **extra,
    }


def _ok(
    candidate: RuntimeCandidate, seconds: float, scores: dict[str, float]
) -> LevelResult:
    return LevelResult(
        BenchmarkMeasurement(
            candidate,
            960,
            100_000,
            seconds,
            latency_p50_seconds=0.5,
            latency_p95_seconds=1.0,
        ),
        scores,
    )


def _rows(config: dict) -> list[dict]:
    path = Path(config["output_dir"]) / "benchmark_results.jsonl"
    return [
        json.loads(line)["measurement"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_benchmark_worker_keeps_failed_levels_with_their_server_log(tmp_path):
    config = _config(tmp_path)
    levels = _levels()

    def measure(index, candidate):
        if index == 1:
            log = Path(config["output_dir"]) / "level-1" / "server-0.log"
            log.parent.mkdir(parents=True)
            log.write_text("ggml_cuda: out of memory\n", encoding="utf-8")
            raise RuntimeError("replica(s) 0 exited")
        return _ok(candidate, 10.0 if index == 0 else 8.0, {"q\x1fc": 0.5})

    artifact = run_benchmark_worker(config, measure_level=measure)

    rows = _rows(config)
    assert [row["status"] for row in rows] == ["ok", "invalid", "ok"]
    assert rows[1]["error_category"] == "RuntimeError"
    assert "out of memory" in rows[1]["log_tail"]
    assert [row["candidate"] for row in rows] == [level.to_dict() for level in levels]
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"] == {
        "recommendation": levels[2].to_dict(),
        "sample_count": 960,
    }
    assert (Path(config["output_dir"]) / "benchmark_report.md").is_file()


def test_benchmark_worker_rejects_a_level_whose_scores_drift(tmp_path):
    config = _config(tmp_path)
    levels = _levels()

    def measure(index, candidate):
        drift = 0.01 if index == 2 else 0.0
        return _ok(candidate, 10.0 - index, {"q\x1fc": 0.5 + drift})

    artifact = run_benchmark_worker(config, measure_level=measure)

    rows = _rows(config)
    assert [row["status"] for row in rows] == ["ok", "ok", "invalid"]
    assert rows[2]["error_category"] == "score_mismatch"
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["recommendation"] == levels[1].to_dict()


def test_benchmark_worker_records_no_recommendation_when_every_level_fails(tmp_path):
    config = _config(tmp_path)

    def measure(_index, _candidate):
        raise RuntimeError("server did not start")

    artifact = run_benchmark_worker(config, measure_level=measure)

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["recommendation"] is None
    assert artifact.completion.complete == len(_levels())


def _write_candidates(path: Path, queries: int, documents: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query in range(queries):
            row = {
                "query_id": f"q{query}",
                "query": f"query {query}",
                "candidates": [
                    {
                        "chunk_id": f"c{query}-{document}",
                        "document_text": "x" * (query + document + 1),
                    }
                    for document in range(documents)
                ],
            }
            handle.write(json.dumps(row) + "\n")


def test_rerank_level_restarts_servers_and_sends_warm_up_then_full_groups(
    tmp_path, monkeypatch
):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, queries=40, documents=30)
    config = _config(
        tmp_path,
        input_files={
            "candidates": {
                "path": str(candidates),
                "filename": candidates.name,
                "sha256": sha256_file(candidates),
            }
        },
    )
    started_levels: list[dict] = []
    calls: list[tuple[str, int]] = []

    @contextmanager
    def fake_servers(level_config, *, telemetry=None):
        del telemetry
        started_levels.append(level_config)
        yield [
            SimpleNamespace(base_url="http://127.0.0.1:11434"),
            SimpleNamespace(base_url="http://127.0.0.1:11435"),
        ]

    class FakeClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def rerank_native(self, query, documents, model):
            assert model == MODEL
            calls.append((query, len(documents)))
            return [len(document) / 100 for document in documents]

    monkeypatch.setattr(benchmark_worker, "managed_model_servers", fake_servers)
    monkeypatch.setattr(benchmark_worker, "LlamaCppClient", FakeClient)

    artifact = run_benchmark_worker(config)

    levels = _levels()
    assert [item["runtime_overrides"] for item in started_levels] == [
        level.to_dict() for level in levels
    ]
    assert [Path(item["output_dir"]).name for item in started_levels] == [
        "level-0",
        "level-1",
        "level-2",
    ]
    # One untimed warm-up group, then 32 timed groups of 30 documents, per level.
    assert len(calls) == len(levels) * 33
    assert {size for _query, size in calls} == {30}
    rows = _rows(config)
    assert all(row["status"] == "ok" and row["items"] == 960 for row in rows)
    assert all(row["max_abs_score_delta"] == 0.0 for row in rows)
    assert artifact.completion.complete == 3


def test_benchmark_worker_main_loads_config_and_invokes_worker(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config = {"output_dir": str(tmp_path), "model": "model"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured = {}

    def fake_worker(received):
        captured["config"] = received

    monkeypatch.setenv("KAGGLE_PIPELINE_CONFIG", str(config_path))
    monkeypatch.setattr(benchmark_worker, "run_benchmark_worker", fake_worker)

    assert benchmark_worker.main() == 0
    assert captured["config"] == config
```

In `seed-pipeline/tests/integrations/kaggle/test_stages.py` add `from seed_pipeline.runtime.catalog import require_model` and in `test_rerank_benchmark_isolated_from_production_artifact` replace `assert "benchmark_levels" in job.worker_config` with:

```python
    space = require_model("qwen3-reranker:0.6b-fp16").rerank_search_space
    assert space is not None
    assert job.contract_version == 2
    assert job.worker_config["benchmark_levels"] == [
        candidate.to_dict() for candidate in space.candidates
    ]
    assert job.worker_config["benchmark_groups"] == 32
    assert job.worker_config["benchmark_items"] == 960
```

In `seed-pipeline/tests/integrations/kaggle/test_auto_profile.py` add `import pytest` and `from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement`, replace `_benchmark_result` with:

```python
def _benchmark_result(
    tmp_path: Path, *, valid: bool = True, log_tail: str | None = None
):
    selected = rerank_runtime_profile(MODEL, index=-1)
    measurement = (
        BenchmarkMeasurement(
            selected, 960, 100, 1.0, latency_p50_seconds=0.1, latency_p95_seconds=0.2
        )
        if valid
        else BenchmarkMeasurement.invalid(selected, "ModelServerExited", log_tail)
    )
    data = tmp_path / "benchmark_results.jsonl"
    data.write_text(
        json.dumps({"measurement": measurement.to_dict()}) + "\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "recommendation": selected.to_dict() if valid else None,
                    "sample_count": 960,
                },
                "identity": {"job_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(artifact_path=data, completion=Completion(1, 1, 0))
```

change `assert result.profile.sample_count == 512` to `assert result.profile.sample_count == 960`, and append:

```python
def test_profile_stores_the_recommended_candidate(tmp_path):
    result = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=False,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=lambda **_kwargs: _benchmark_result(tmp_path),
    )

    assert result.profile is not None
    assert result.profile.selected == rerank_runtime_profile(MODEL, index=-1)
    assert result.profile.measurements[0]["candidate"] == (
        result.profile.selected.to_dict()
    )


def test_benchmark_without_a_valid_level_stops_with_the_server_log_tail(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA error: out of memory"):
        ensure_runtime_profile(
            workload="rerank",
            benchmark_stage="rerank-benchmark",
            model=MODEL,
            input_path=tmp_path / "candidates.jsonl",
            gguf_root=tmp_path / "gguf",
            budget_seconds=60,
            dry_run=False,
            force=False,
            profile_root=tmp_path / "profiles",
            runtime_sha256="b" * 64,
            benchmark_runner=lambda **_kwargs: _benchmark_result(
                tmp_path, valid=False, log_tail="CUDA error: out of memory"
            ),
        )

    assert not list((tmp_path / "profiles").rglob("*.json"))
```

- [ ] **Step 6: Run them to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/integrations/kaggle/test_benchmark_worker.py tests/integrations/kaggle/test_stages.py tests/integrations/kaggle/test_auto_profile.py`
Expected: FAIL — `ImportError: cannot import name 'BenchmarkLevel'` from `seed_pipeline.runtime.benchmarking` while collecting.

- [ ] **Step 7: Rewrite the Kaggle benchmark worker**

Replace the whole of `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/benchmark.py` with:

```python
from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from seed_pipeline.integrations.kaggle.models import CloudArtifact
from seed_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
)
from seed_pipeline.integrations.kaggle.workers.scheduling import stream_map_ordered
from seed_pipeline.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from seed_pipeline.runtime.benchmarking import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkWorkload,
    LevelResult,
    RerankGroup,
    check_score_consistency,
    percentile,
    recommend,
    rerank_groups_from_rows,
    sample_rerank_groups,
    stratified_sample,
)
from seed_pipeline.runtime.catalog import ModelKind, ModelSpec, require_model
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

MeasureLevel = Callable[[int, RuntimeCandidate], LevelResult]
Clients = list[tuple[int, LlamaCppClient]]
SERVER_LOG_TAIL_CHARACTERS = 4000


def run_benchmark_worker(
    config: dict,
    *,
    measure_level: MeasureLevel | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CloudArtifact:
    """Measure every runtime level end to end; never creates a production cache."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = identity_from_config(config)
    levels = tuple(
        RuntimeCandidate.from_dict(item) for item in config.get("benchmark_levels", ())
    )
    if not levels:
        raise ValueError("benchmark_levels must not be empty")
    workload = BenchmarkWorkload(
        stage=str(config.get("stage", "benchmark")),
        model=str(config["model"]),
        sample_count=int(config.get("benchmark_items", 0)),
        levels=levels,
    )
    measure = measure_level or _default_measure(config, levels, output_dir, clock)
    results: list[LevelResult] = []
    for index, candidate in enumerate(levels):
        try:
            result = measure(index, candidate)
            if result.measurement.candidate != candidate:
                raise ValueError("benchmark measurement level mismatch")
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            detail = f"{exc}\n{_server_log_tail(output_dir / f'level-{index}')}"
            result = LevelResult(
                BenchmarkMeasurement.invalid(
                    candidate, type(exc).__name__, detail.strip()
                ),
                {},
            )
        results.append(result)
    measurements = check_score_consistency(results)
    report = BenchmarkReport(
        workload, measurements, recommend(measurements, objective="throughput")
    )
    data_path = output_dir / "benchmark_results.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            row = {
                "identity": identity.sha256,
                "schema_version": report.schema_version,
                "measurement": measurement.to_dict(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "benchmark_report.md").write_text(_markdown(report), encoding="utf-8")
    return artifact_from_output(
        data_path,
        artifact_type="runtime_benchmark",
        identity=identity,
        total=len(measurements),
        complete=len(measurements),
        runtime_summary={
            "recommendation": (
                report.recommendation.to_dict() if report.recommendation else None
            ),
            "sample_count": workload.sample_count,
        },
    )


def _default_measure(
    config: dict,
    levels: tuple[RuntimeCandidate, ...],
    output_dir: Path,
    clock: Callable[[], float],
) -> MeasureLevel:
    spec = require_model(str(config["model"]))
    if spec.kind is ModelKind.RERANKER:
        groups = sample_rerank_groups(
            rerank_groups_from_rows(
                _candidate_rows(config),
                documents_per_group=levels[0].request_batch_size,
            ),
            int(config["benchmark_groups"]),
        )
        if not groups:
            raise ValueError("candidates contain no full query group to benchmark")

        def measure_rerank(index: int, candidate: RuntimeCandidate) -> LevelResult:
            with _level_clients(config, spec, index, candidate, output_dir) as clients:
                return _measure_rerank(spec, candidate, groups, clients, clock)

        return measure_rerank
    texts, characters = _sample_texts(config, int(config.get("benchmark_items", 0)))

    def measure_embedding(index: int, candidate: RuntimeCandidate) -> LevelResult:
        with _level_clients(config, spec, index, candidate, output_dir) as clients:
            return _measure_embedding(
                spec, candidate, texts, characters, clients, clock
            )

    return measure_embedding


@contextmanager
def _level_clients(
    config: dict,
    spec: ModelSpec,
    index: int,
    candidate: RuntimeCandidate,
    output_dir: Path,
) -> Iterator[Clients]:
    level_dir = output_dir / f"level-{index}"
    level_config = {
        **config,
        "runtime_overrides": candidate.to_dict(),
        "output_dir": str(level_dir),
    }
    telemetry = RuntimeTelemetry("benchmark", spec.name, level_dir)
    try:
        # A fresh server per level, so no slot or KV state carries over.
        with managed_model_servers(level_config, telemetry=telemetry) as servers:
            yield [
                (server_index, LlamaCppClient(server.base_url))
                for server_index, server in enumerate(servers)
            ]
    finally:
        telemetry.close()
        telemetry.write_report()


def _measure_rerank(
    spec: ModelSpec,
    candidate: RuntimeCandidate,
    groups: tuple[RerankGroup, ...],
    clients: Clients,
    clock: Callable[[], float],
) -> LevelResult:
    latencies: list[float] = []
    scores: dict[str, float] = {}

    async def operation(
        resource: tuple[int, LlamaCppClient], _index: int, group: RerankGroup
    ) -> None:
        _server_index, client = resource
        started = clock()
        values = await asyncio.to_thread(
            client.rerank_native, group.query, list(group.documents), spec.name
        )
        latencies.append(max(0.0, clock() - started))
        scores.update(zip(group.score_keys(), values, strict=True))

    asyncio.run(stream_map_ordered(groups[:1], clients, 1, operation, float("inf")))
    latencies.clear()
    scores.clear()
    started = clock()
    asyncio.run(
        stream_map_ordered(
            groups, clients, candidate.concurrency, operation, float("inf")
        )
    )
    elapsed = max(0.0, clock() - started)
    return LevelResult(
        BenchmarkMeasurement(
            candidate,
            sum(len(group.documents) for group in groups),
            sum(group.characters for group in groups),
            elapsed,
            latency_p50_seconds=percentile(latencies, 0.50),
            latency_p95_seconds=percentile(latencies, 0.95),
        ),
        scores,
    )


def _measure_embedding(
    spec: ModelSpec,
    candidate: RuntimeCandidate,
    texts: list[str],
    characters: int,
    clients: Clients,
    clock: Callable[[], float],
) -> LevelResult:
    size = candidate.request_batch_size
    batches = [texts[start : start + size] for start in range(0, len(texts), size)]
    if not batches:
        raise ValueError("benchmark input contains no text")
    latencies: list[float] = []

    async def operation(
        resource: tuple[int, LlamaCppClient], _index: int, batch: list[str]
    ) -> None:
        _server_index, client = resource
        started = clock()
        await asyncio.to_thread(
            client.embed, batch, spec.name, spec.vector_dimension or 0
        )
        latencies.append(max(0.0, clock() - started))

    asyncio.run(stream_map_ordered(batches[:1], clients, 1, operation, float("inf")))
    latencies.clear()
    started = clock()
    asyncio.run(
        stream_map_ordered(
            batches, clients, candidate.concurrency, operation, float("inf")
        )
    )
    return LevelResult(
        BenchmarkMeasurement(
            candidate,
            len(texts),
            characters,
            max(0.0, clock() - started),
            latency_p50_seconds=percentile(latencies, 0.50),
            latency_p95_seconds=percentile(latencies, 0.95),
        ),
        {},
    )


def _candidate_rows(config: dict) -> Iterator[dict]:
    path = resolve_input_file(config, "candidates")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sample_texts(config: dict, count: int) -> tuple[list[str], int]:
    stage = str(config.get("stage", ""))
    path = resolve_input_file(config, "input")
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                str(
                    row.get("query", "")
                    if "query" in stage
                    else row.get("embedding_text", row.get("text", ""))
                )
            )
    sample = stratified_sample(
        rows, count, key=lambda item: (len(item), item), length=len
    )
    return list(sample), sum(len(item) for item in sample)


def _server_log_tail(level_dir: Path) -> str:
    return "\n".join(
        f"{path.name}:\n"
        + path.read_text(encoding="utf-8", errors="replace")[
            -SERVER_LOG_TAIL_CHARACTERS:
        ]
        for path in sorted(level_dir.glob("server-*.log"))
    )


def _seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _markdown(report: BenchmarkReport) -> str:
    recommendation = (
        json.dumps(report.recommendation.to_dict(), sort_keys=True)
        if report.recommendation
        else "none"
    )
    rows = [
        f"| {item.candidate.server_slots} | {item.candidate.physical_batch_size} "
        f"| {item.candidate.request_batch_size} | {item.candidate.concurrency} "
        f"| {item.items} | {item.items_per_second:.2f} "
        f"| {_seconds(item.latency_p50_seconds)} | {_seconds(item.latency_p95_seconds)} "
        f"| {item.status} | {item.error_category or ''} |"
        for item in report.measurements
    ]
    return (
        "# Runtime benchmark\n\n"
        f"- stage: `{report.workload.stage}`\n"
        f"- model: `{report.workload.model}`\n"
        f"- recommendation: `{recommendation}`\n\n"
        "| slots | ubatch | batch | concurrency | items | items/s | p50 s | p95 s "
        "| status | error |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n" + "\n".join(rows) + "\n"
    )


def main() -> int:
    config = json.loads(
        Path(os.environ["KAGGLE_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
    )
    run_benchmark_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Build full-candidate levels in the benchmark stage**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/stages.py`:
- replace line 17 `from seed_pipeline.runtime.benchmarking import BenchmarkLevel` with `from seed_pipeline.runtime.benchmarking import KAGGLE_RERANK_BENCHMARK_GROUPS` and add `JSONValue` to the `seed_pipeline.integrations.kaggle.models` import;
- in `BenchmarkStage` change `contract_version: int = 1` to `contract_version: int = 2`;
- replace everything from `base_job = self.base.build_job(base_request)` through the closing `)` of `config.update(...)` with:

```python
base_job = self.base.build_job(base_request)
levels: list[JSONValue] = [candidate.to_dict() for candidate in search_space.candidates]
benchmark: dict[str, JSONValue] = {"benchmark_levels": levels}
if spec.kind is ModelKind.RERANKER:
    # Each measured request is one full query group, as in production.
    benchmark["benchmark_groups"] = KAGGLE_RERANK_BENCHMARK_GROUPS
    benchmark["benchmark_items"] = (
        KAGGLE_RERANK_BENCHMARK_GROUPS * search_space.candidates[0].request_batch_size
    )
else:
    benchmark["benchmark_items"] = request.benchmark_items or base_job.expected_total
identity = JobIdentity.create(
    stage=self.name,
    contract_version=self.contract_version,
    model=request.model,
    model_sha256=spec.sha256,
    input_sha256=base_job.input_bundle.sha256,
    runtime_parameters=benchmark,
)
output_dir = request.output_dir / self.name.value / spec.slug / identity.sha256[:12]
config = dict(base_job.worker_config)
config.update(
    {
        "stage": self.name.value,
        "identity": identity.payload,
        "job_sha256": identity.sha256,
        **benchmark,
    }
)
```

- [ ] **Step 9: Store the recommended candidate as the profile**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py` add `from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement, describe_levels` and replace the whole `_load_benchmark_selection` function with:

```python
def _load_benchmark_selection(
    artifact_path: Path, search_space: RuntimeSearchSpace
) -> tuple[RuntimeCandidate, tuple[BenchmarkMeasurement, ...], str, int]:
    data_path = Path(artifact_path)
    manifest_path = data_path.with_name("manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid benchmark artifact: {data_path}") from exc
    try:
        measurements = tuple(
            BenchmarkMeasurement.from_dict(row["measurement"]) for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark measurement row is malformed") from exc
    runtime = manifest.get("runtime") or {}
    recommendation = runtime.get("recommendation")
    if recommendation is None:
        raise RuntimeError(
            "runtime benchmark found no valid level:\n" + describe_levels(measurements)
        )
    try:
        selected = RuntimeCandidate.from_dict(recommendation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark recommendation is malformed") from exc
    if selected not in search_space.candidates:
        raise RuntimeError("benchmark recommendation is not in the search space")
    if not any(
        item.status == "ok" and item.candidate == selected for item in measurements
    ):
        raise RuntimeError("benchmark recommendation has no successful measurement")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("job_sha256"), str
    ):
        raise RuntimeError("benchmark manifest is missing job identity")
    sample_count = runtime.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise RuntimeError("benchmark manifest has no sample count")
    return selected, measurements, identity["job_sha256"], sample_count
```

In `ensure_runtime_profile` replace

```python
    selected, measurements, benchmark_job_sha256 = _load_benchmark_selection(
        result.artifact_path, space
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=512,
        measurements=measurements,
        benchmark_job_sha256=benchmark_job_sha256,
    )
```

with:

```python
selected, measurements, benchmark_job_sha256, sample_count = _load_benchmark_selection(
    result.artifact_path, space
)
profile = RuntimeProfile.create(
    identity,
    selected,
    sample_count=sample_count,
    measurements=[item.to_dict() for item in measurements],
    benchmark_job_sha256=benchmark_job_sha256,
)
```

- [ ] **Step 10: Document the Kaggle benchmark**

In `seed-pipeline/docs/guides/cli-reference.md` replace the paragraph under `## Runtime profiling trên Kaggle` with:

```text
Stage production tự benchmark một lần nếu chưa có profile hợp lệ và lưu ở `data/cache/kaggle_profiles/<workload>/<model>.json`; profile mất hiệu lực khi model, runtime, topology hoặc search space đổi. Mỗi mức là một cấu hình server đầy đủ (`-np`, `-ub`, số tài liệu mỗi request, số request đồng thời) và server khởi động lại khi đổi mức. Reranker chạy `--reranking --kv-unified -np N -c UB -b UB -ub UB`; tải đo là 32 nhóm câu hỏi × 30 ứng viên chọn phân tầng theo tổng số ký tự, cộng một nhóm khởi động không tính giờ. Mức nào có điểm lệch mức hợp lệ đầu tiên quá `1e-3` bị ghi `invalid` (`score_mismatch`); stage chọn mức có số cặp/giây cao nhất. Mọi mức đều lỗi thì stage dừng và in trạng thái cùng đuôi log server của từng mức.
```

- [ ] **Step 11: Run the tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime tests/integrations/kaggle tests/test_docs.py`
Expected: PASS. `grep -rn 'BenchmarkLevel\|embedding_levels\|benchmark_candidates' src tests` prints nothing.

- [ ] **Step 12: Run the Seed gate**

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/runtime/benchmarking.py seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/benchmark.py seed-pipeline/src/seed_pipeline/integrations/kaggle/stages.py seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py seed-pipeline/docs/guides/cli-reference.md seed-pipeline/tests/runtime/test_benchmarking.py seed-pipeline/tests/integrations/kaggle/test_benchmark_worker.py seed-pipeline/tests/integrations/kaggle/test_stages.py seed-pipeline/tests/integrations/kaggle/test_auto_profile.py
git diff --cached --name-status
git commit -m "feat(seed): benchmark full runtime candidates end to end" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Batched CPU reranker in compose, driven by runtime candidates

**Files:**
- Modify: `compose.yaml` (backend env lines 57–60; `llama-reranker` environment from `# One slot keeps the shared…` through `LLAMA_ARG_THREADS`)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/compose.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/runtime/catalog.py` (delete `ModelSpec.local_parallel`)
- Modify: `.env.example` (CPU tuning block), `backend/.env.example` (`MAX_CONCURRENT` comment), `README.md` (bullet list under the compose commands)
- Modify: `backend/src/pharma_agent/infrastructure/settings.py:109` (comment), `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md:327`
- Test: `seed-pipeline/tests/runtime/test_compose.py` (whole file), `seed-pipeline/tests/runtime/test_compose_reranker_integration.py` (new)

**Interfaces:**
- Consumes: Task 5 `RuntimeCandidate` (with `threads`), `LOCAL_RERANK_SEARCH_SPACE`, `reranker_candidate`; Task 4 compose `LLAMA_ARG_RERANKING: "true"`.
- Produces (`seed_pipeline.runtime.compose`):
  - `LlamaCppComposeManager.ensure(role: str, spec: ModelSpec, gguf_root: Path, runtime: RuntimeCandidate | None = None) -> str` — with `runtime`, sets `LLAMA_RERANKER_PARALLEL`, `LLAMA_RERANKER_UBATCH_SIZE` and (when set) `LLAMA_RERANKER_THREADS` before `docker compose up -d --force-recreate llama-reranker`; without it compose defaults or the root `.env` apply.
  - `LlamaCppComposeManager.logs(role: str, *, tail: int = 80, environment: Mapping[str, str] | None = None) -> str`
  - `reranker_environment(runtime: RuntimeCandidate) -> dict[str, str]`
  - `compose_llama_cpp_image(compose_file: Path) -> str`
- Produces (stack): `llama-reranker` runs `LLAMA_ARG_RERANKING=true`, `LLAMA_ARG_KV_UNIFIED=true`, `LLAMA_ARG_N_PARALLEL=${LLAMA_RERANKER_PARALLEL:-16}`, `LLAMA_ARG_CTX_SIZE`/`LLAMA_ARG_BATCH`/`LLAMA_ARG_UBATCH=${LLAMA_RERANKER_UBATCH_SIZE:-8192}`, `LLAMA_ARG_THREADS=${LLAMA_RERANKER_THREADS:-8}`; backend `PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT="1"`, `MAX_CANDIDATES=${RERANK_MAX_CANDIDATES:-15}`, `TIMEOUT_SECONDS=${RERANK_TIMEOUT_SECONDS:-120}`. Plan C replaces the three reranker defaults with values measured on the i5-13420H.

- [ ] **Step 1: Write the failing unit tests**

Replace the whole of `seed-pipeline/tests/runtime/test_compose.py` with:

```python
from dataclasses import replace
from pathlib import Path

import pytest

from seed_pipeline.config.paths import COMPOSE_FILE
from seed_pipeline.runtime.catalog import LOCAL_RERANK_SEARCH_SPACE, require_model
from seed_pipeline.runtime.compose import (
    LlamaCppComposeManager,
    compose_llama_cpp_image,
    file_sha256,
    reranker_environment,
)
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate


class RecordingRunner:
    def __init__(self, logs: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str] | None]] = []
        self.logs = logs

    def run(self, args, env=None, capture_output=False):
        self.calls.append((list(args), env))
        return self.logs if capture_output else ""


class HealthyClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def health(self) -> None:
        return None


def _manager(tmp_path: Path, runner: RecordingRunner) -> LlamaCppComposeManager:
    return LlamaCppComposeManager(
        tmp_path / "compose.yaml",
        runner=runner,
        client_factory=HealthyClient,
        environment={},
    )


def _spec(tmp_path: Path, model: str):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    return replace(
        require_model(model),
        canonical_filename=artifact.name,
        byte_size=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )


def test_reranker_without_runtime_keeps_the_compose_defaults(tmp_path: Path) -> None:
    runner = RecordingRunner()

    endpoint = _manager(tmp_path, runner).ensure(
        "reranker", _spec(tmp_path, "qwen3-reranker:4b-fp16"), tmp_path
    )

    command, environment = runner.calls[0]
    assert command[-3:] == ["-d", "--force-recreate", "llama-reranker"]
    assert environment is not None
    assert environment["LLAMA_RERANKER_MODEL"] == "model.gguf"
    for name in (
        "LLAMA_RERANKER_PARALLEL",
        "LLAMA_RERANKER_UBATCH_SIZE",
        "LLAMA_RERANKER_THREADS",
        "LLAMA_RERANKER_RERANKING",
    ):
        assert name not in environment
    assert endpoint == "http://127.0.0.1:11435"


def test_reranker_runtime_recreates_the_service_with_its_level(tmp_path: Path) -> None:
    runner = RecordingRunner()
    level = LOCAL_RERANK_SEARCH_SPACE.candidates[3]

    _manager(tmp_path, runner).ensure(
        "reranker", _spec(tmp_path, "qwen3-reranker:4b-fp16"), tmp_path, runtime=level
    )

    _command, environment = runner.calls[0]
    assert environment is not None
    assert (
        environment["LLAMA_RERANKER_PARALLEL"],
        environment["LLAMA_RERANKER_UBATCH_SIZE"],
        environment["LLAMA_RERANKER_THREADS"],
    ) == ("16", "8192", "12")


def test_runtime_is_rejected_for_the_embedding_service(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llama-reranker"):
        _manager(tmp_path, RecordingRunner()).ensure(
            "embedding",
            _spec(tmp_path, "qwen3-embedding:4b-fp16"),
            tmp_path,
            runtime=LOCAL_RERANK_SEARCH_SPACE.candidates[0],
        )


def test_reranker_environment_needs_one_size_for_context_batch_and_ubatch() -> None:
    with pytest.raises(ValueError, match="one size"):
        reranker_environment(RuntimeCandidate(16, 1, 15, 8192, 8192, 4096))


def test_logs_reads_the_service_tail(tmp_path: Path) -> None:
    runner = RecordingRunner(logs="model loaded")

    assert _manager(tmp_path, runner).logs("reranker") == "model loaded"
    command, _environment = runner.calls[0]
    assert command[-4:] == ["logs", "--tail", "80", "llama-reranker"]


def test_compose_pins_the_llama_cpp_server_image() -> None:
    assert (
        compose_llama_cpp_image(COMPOSE_FILE)
        == "ghcr.io/ggml-org/llama.cpp:server-b10920"
    )


def test_compose_reranker_serves_batched_unified_kv() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    block = text[text.index("  llama-reranker:") : text.index("  postgres:")]

    for line in (
        'LLAMA_ARG_RERANKING: "true"',
        'LLAMA_ARG_KV_UNIFIED: "true"',
        'LLAMA_ARG_N_PARALLEL: "${LLAMA_RERANKER_PARALLEL:-16}"',
        'LLAMA_ARG_CTX_SIZE: "${LLAMA_RERANKER_UBATCH_SIZE:-8192}"',
        'LLAMA_ARG_BATCH: "${LLAMA_RERANKER_UBATCH_SIZE:-8192}"',
        'LLAMA_ARG_UBATCH: "${LLAMA_RERANKER_UBATCH_SIZE:-8192}"',
        'LLAMA_ARG_THREADS: "${LLAMA_RERANKER_THREADS:-8}"',
    ):
        assert line in block
    assert "KV_UNIFIED_PER_SLOT" not in block


def test_compose_backend_sends_one_rerank_request_at_a_time() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")

    assert 'PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT: "1"' in text
    assert (
        'PHARMA_RETRIEVAL__RERANK__MAX_CANDIDATES: "${RERANK_MAX_CANDIDATES:-15}"'
        in text
    )
    assert (
        'PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS: "${RERANK_TIMEOUT_SECONDS:-120}"'
        in text
    )
```

- [ ] **Step 2: Run them to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_compose.py`
Expected: FAIL — `ImportError: cannot import name 'compose_llama_cpp_image'`.

- [ ] **Step 3: Rewrite the compose manager**

Replace the whole of `seed-pipeline/src/seed_pipeline/runtime/compose.py` with:

```python
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from seed_pipeline.runtime.catalog import ModelKind, ModelSpec
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

_LLAMA_CPP_IMAGE = re.compile(
    r"^\s*image:\s*(ghcr\.io/ggml-org/llama\.cpp:\S+)\s*$", re.MULTILINE
)


class ModelArtifactError(RuntimeError):
    pass


class ComposeStartupError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_artifact(root: Path, spec: ModelSpec) -> Path:
    model = Path(root) / spec.canonical_filename
    if not model.is_file() or model.is_symlink():
        raise ModelArtifactError(f"Model file is missing or invalid: {model}")
    actual_size = model.stat().st_size
    if actual_size != spec.byte_size:
        raise ModelArtifactError(
            f"Model byte size mismatch for {spec.name}: {actual_size} != {spec.byte_size}"
        )
    actual_hash = file_sha256(model)
    if actual_hash != spec.sha256:
        raise ModelArtifactError(
            f"Model SHA-256 mismatch for {spec.name}: {actual_hash} != {spec.sha256}"
        )
    return model


def compose_llama_cpp_image(compose_file: Path) -> str:
    """The llama.cpp server image tag that compose.yaml pins."""
    match = _LLAMA_CPP_IMAGE.search(Path(compose_file).read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{compose_file} declares no llama.cpp server image")
    return match.group(1)


def reranker_environment(runtime: RuntimeCandidate) -> dict[str, str]:
    """Root .env variables that compose.yaml maps to llama-reranker's LLAMA_ARG_*."""
    if not (
        runtime.context_per_slot
        == runtime.logical_batch_size
        == runtime.physical_batch_size
    ):
        raise ValueError(
            "reranker runtime must use one size for context, batch and ubatch"
        )
    environment = {
        "LLAMA_RERANKER_PARALLEL": str(runtime.server_slots),
        "LLAMA_RERANKER_UBATCH_SIZE": str(runtime.physical_batch_size),
    }
    if runtime.threads is not None:
        environment["LLAMA_RERANKER_THREADS"] = str(runtime.threads)
    return environment


class SubprocessRunner:
    def run(self, args, env=None, capture_output=False):
        completed = subprocess.run(
            list(args),
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
        return completed.stdout or ""


class LlamaCppComposeManager:
    def __init__(
        self,
        compose_file: Path,
        runner=None,
        client_factory=LlamaCppClient,
        environment=None,
        startup_timeout: float = 180.0,
        poll_interval: float = 2.0,
    ) -> None:
        self.compose_file = Path(compose_file)
        self.runner = runner or SubprocessRunner()
        self.client_factory = client_factory
        self.environment = dict(os.environ if environment is None else environment)
        self.startup_timeout = float(startup_timeout)
        self.poll_interval = float(poll_interval)

    def ensure(
        self,
        role: str,
        spec: ModelSpec,
        gguf_root: Path,
        runtime: RuntimeCandidate | None = None,
    ) -> str:
        if role not in {"embedding", "reranker"}:
            raise ValueError(f"Unsupported llama.cpp service role: {role}")
        if role == "embedding" and spec.kind is not ModelKind.EMBEDDING:
            raise ValueError(f"Embedding service cannot load {spec.kind.value} model")
        if role == "reranker" and spec.kind is not ModelKind.RERANKER:
            raise ValueError(f"Reranker service cannot load {spec.kind.value} model")
        if runtime is not None and role != "reranker":
            raise ValueError(
                "runtime settings apply to the llama-reranker service only"
            )
        verify_model_artifact(gguf_root, spec)
        service = f"llama-{role}"
        prefix = f"LLAMA_{role.upper()}"
        environment = dict(self.environment)
        environment["GGUF_DIR"] = str(Path(gguf_root).resolve())
        environment[f"{prefix}_MODEL"] = spec.canonical_filename
        if runtime is not None:
            environment.update(reranker_environment(runtime))
        command = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            "up",
            "-d",
            "--force-recreate",
            service,
        ]
        self.runner.run(command, env=environment)
        port = environment.get(
            f"{prefix}_PORT", "11434" if role == "embedding" else "11435"
        )
        endpoint = f"http://127.0.0.1:{port}"
        client = self.client_factory(endpoint)
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                client.health()
                return endpoint
            except Exception as exc:
                last_error = exc
                time.sleep(self.poll_interval)
        logs = self.logs(role, environment=environment)
        raise ComposeStartupError(
            f"{service} did not become healthy: {last_error}\n{logs[-8000:]}"
        )

    def logs(
        self,
        role: str,
        *,
        tail: int = 80,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        return self.runner.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "logs",
                "--tail",
                str(tail),
                f"llama-{role}",
            ],
            env=dict(self.environment if environment is None else environment),
            capture_output=True,
        )


def resolve_server(
    mode: str,
    external_urls: list[str],
    spec: ModelSpec,
    manager: LlamaCppComposeManager,
    gguf_root: Path,
) -> list[str]:
    if mode == "external":
        endpoints = [value.rstrip("/") for value in external_urls if value.strip()]
        if not endpoints:
            raise ValueError("External server mode requires --llama-server-url")
        return endpoints
    if mode != "compose":
        raise ValueError(f"Unsupported server mode: {mode}")
    role = "embedding" if spec.kind is ModelKind.EMBEDDING else "reranker"
    return [manager.ensure(role, spec, gguf_root)]
```

In `seed-pipeline/src/seed_pipeline/runtime/catalog.py` delete the field `local_parallel: int = 1` from `ModelSpec` (`grep -rn local_parallel src tests` must print nothing afterwards).

- [ ] **Step 4: Batch the compose reranker and pin the backend call pattern**

In `compose.yaml` replace, in `llama-reranker`'s environment, the lines from `# One slot keeps the shared system/instruction/query prefix cached between candidates.` through `LLAMA_ARG_THREADS: "${LLAMA_RERANKER_THREADS:-8}"` with:

```yaml
      # Rank pooling computes each document in one pass and frees its KV right after, so one
      # unified KV pool the size of a physical batch serves every slot: -c = -b = -ub. The
      # ubatch must hold the longest rerank prompt (1,664 tokens), so keep it >= 2048.
      LLAMA_ARG_KV_UNIFIED: "true"
      # Starting values. `seed rerank --backend local --benchmark` measures this machine and
      # prints the LLAMA_RERANKER_* values to set in the root .env.
      LLAMA_ARG_N_PARALLEL: "${LLAMA_RERANKER_PARALLEL:-16}"
      LLAMA_ARG_CTX_SIZE: "${LLAMA_RERANKER_UBATCH_SIZE:-8192}"
      LLAMA_ARG_BATCH: "${LLAMA_RERANKER_UBATCH_SIZE:-8192}"
      LLAMA_ARG_UBATCH: "${LLAMA_RERANKER_UBATCH_SIZE:-8192}"
      # Prompt processing dominates reranking; give it the performance-core threads.
      LLAMA_ARG_THREADS: "${LLAMA_RERANKER_THREADS:-8}"
```

and in `x-backend` replace lines 57–60 (`# One request per reranker slot…` through `PHARMA_RETRIEVAL__RERANK__MAX_CANDIDATES`) with:

```yaml
    # One /v1/rerank request carries every candidate of a search round and llama-server
    # spreads its documents over the slots, so the backend sends one request at a time.
    PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT: "1"
    # CPU profile: rerank only the best RRF candidates (RRF alone puts 95.7% of answers in top 10).
    PHARMA_RETRIEVAL__RERANK__MAX_CANDIDATES: "${RERANK_MAX_CANDIDATES:-15}"
    # At least twice the CPU p95 of one rerank request (latency_p95_seconds of the benchmark).
    PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS: "${RERANK_TIMEOUT_SECONDS:-120}"
```

The `llama-embedding` service keeps `LLAMA_ARG_KV_UNIFIED_PER_SLOT` (compose runs `b10920`; only reranker flags must also exist on Kaggle's `b9637`).

In `.env.example` replace the lines from `# LLAMA_RERANKER_THREADS=8` through `# RERANK_MAX_CANDIDATES=15` with:

```text
# Reranker server: slots (-np), one size for -c/-b/-ub (at least 2048) and threads.
# LLAMA_RERANKER_PARALLEL=16
# LLAMA_RERANKER_UBATCH_SIZE=8192
# LLAMA_RERANKER_THREADS=8
# RERANK_MAX_CANDIDATES=15
# Backend timeout for one rerank request; keep it at least twice the measured CPU p95.
# RERANK_TIMEOUT_SECONDS=120
```

In `backend/.env.example` replace `# PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT=2       # match the server's --parallel slots` with:

```text
# PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT=2       # concurrent /v1/rerank requests, each with a whole search round
# PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS=120
```

In `backend/src/pharma_agent/infrastructure/settings.py` replace the comment `# Match the server's parallel slots (llama-server --parallel / LLAMA_RERANKER_PARALLEL).` with `# Concurrent /v1/rerank requests; each carries a whole search round (compose sets 1).`

In `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` replace line 327 with:

```text
| `retrieval.rerank.max_concurrent` / `timeout_seconds` | 2 / 120; compose đặt 1 request đồng thời (mỗi request chứa mọi ứng viên của vòng search) và timeout ít nhất gấp đôi p95 CPU đo được |
```

In `README.md` add after the reranker bullet (the one ending `đặt \`LLAMA_RERANKER_PROTOCOL=none\` để bỏ rerank.`):

```text
- Reranker trên CPU: `LLAMA_RERANKER_PARALLEL` (`-np`), `LLAMA_RERANKER_UBATCH_SIZE` (dùng chung cho `-c`,
  `-b`, `-ub`, tối thiểu 2048) và `LLAMA_RERANKER_THREADS` trong `.env` ở root; `RERANK_TIMEOUT_SECONDS` là
  timeout của backend cho một request rerank.
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime tests/evaluation`
Expected: PASS.

- [ ] **Step 6: Write the Docker integration test**

Create `seed-pipeline/tests/runtime/test_compose_reranker_integration.py`:

```python
"""llama-reranker from compose.yaml scores a long document with the batched flags."""

import os
import shutil
import subprocess
from collections.abc import Iterator

import pytest
import requests

from seed_pipeline.config.paths import COMPOSE_FILE, GGUF_ROOT
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.client import LlamaCppClient, LlamaCppRequestError
from seed_pipeline.runtime.compose import LlamaCppComposeManager, SubprocessRunner
from seed_pipeline.runtime.runtime_profiles import reranker_candidate

pytestmark = pytest.mark.integration

MODEL = "qwen3-reranker:0.6b-fp16"
PROJECT = "seed-rerank-integration"
PORT = "18435"
# The smallest ubatch the catalog allows: a ~1,600-token document must still fit in one pass.
LEVEL = reranker_candidate(
    server_slots=4, ubatch=2048, request_batch_size=3, concurrency=1, threads=4
)
QUERY = "Paracetamol người lớn uống tối đa bao nhiêu một ngày?"
RELEVANT = (
    "Người lớn uống paracetamol tối đa 4 g mỗi ngày, chia nhiều lần, "
    "mỗi lần cách nhau ít nhất 4 giờ."
)
OFF_TOPIC = "Xe buýt số 36 chạy từ bến Long Biên đến Linh Đàm, 15 phút một chuyến."
SENTENCE = (
    "Paracetamol được chuyển hoá chủ yếu ở gan qua liên hợp glucuronid và sulfat; "
    "khi quá liều, chất chuyển hoá NAPQI tích tụ và gây hoại tử tế bào gan. "
)


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(["docker", "info"], capture_output=True, check=False).returncode
        == 0
    )


@pytest.fixture(scope="module")
def reranker_endpoint() -> Iterator[str]:
    spec = require_model(MODEL)
    if not (GGUF_ROOT / spec.canonical_filename).is_file():
        pytest.skip(f"{GGUF_ROOT / spec.canonical_filename} is missing")
    if not _docker_available():
        pytest.skip("Docker is not available")
    environment = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": PROJECT,
        "LLAMA_RERANKER_PORT": PORT,
    }
    manager = LlamaCppComposeManager(
        COMPOSE_FILE, environment=environment, startup_timeout=600
    )
    try:
        yield manager.ensure("reranker", spec, GGUF_ROOT, runtime=LEVEL)
    finally:
        SubprocessRunner().run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down"], env=environment
        )


def _token_count(endpoint: str, text: str) -> int:
    response = requests.post(f"{endpoint}/tokenize", json={"content": text}, timeout=60)
    response.raise_for_status()
    return len(response.json()["tokens"])


def test_native_rerank_scores_a_long_document_in_one_pass(
    reranker_endpoint: str,
) -> None:
    long_document = SENTENCE * (1600 // _token_count(reranker_endpoint, SENTENCE))
    assert 1500 <= _token_count(reranker_endpoint, long_document) <= 1700

    try:
        scores = LlamaCppClient(reranker_endpoint, timeout=600).rerank_native(
            QUERY, [RELEVANT, OFF_TOPIC, long_document], MODEL
        )
    except LlamaCppRequestError as exc:
        pytest.fail(f"llama-server rejected a rerank document: {exc}")

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[0] > scores[1]
```

- [ ] **Step 7: Run the integration test inside tmux**

Run in a tmux window (model load can take minutes):

```bash
tmux new-window -n rerank-integration 'cd seed-pipeline && uv run pytest -q -m integration tests/runtime/test_compose_reranker_integration.py; read -r'
```

Expected: `1 passed`. `1 skipped` means `ai-models/gguf/qwen3-reranker-0.6b-f16.gguf` or Docker is missing — report it instead of claiming success. Afterwards `docker compose -p seed-rerank-integration ps -a` lists no container.

- [ ] **Step 8: Run both gates**

Run the Seed gate and the Backend gate. Expected: PASS (the integration test is deselected by `-m "not integration"`).

- [ ] **Step 9: Commit**

```bash
git add compose.yaml .env.example backend/.env.example README.md backend/src/pharma_agent/infrastructure/settings.py backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md seed-pipeline/src/seed_pipeline/runtime/compose.py seed-pipeline/src/seed_pipeline/runtime/catalog.py seed-pipeline/tests/runtime/test_compose.py seed-pipeline/tests/runtime/test_compose_reranker_integration.py
git diff --cached --name-status
git commit -m "feat(compose): serve the CPU reranker with batched unified-KV slots" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Local rerank sends one request per query

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py:197-224` (scoring loop of `LocalRerankBackend.run`)
- Test: `seed-pipeline/tests/evaluation/conftest.py` (`complete_run` helper and a two-candidate run), `seed-pipeline/tests/evaluation/test_rerank_service.py`

**Interfaces:**
- Consumes: Task 2 `LlamaCppReranker.rerank(query, candidates)` (one native request).
- Produces: `LocalRerankBackend.run(request)` calls `reranker.rerank(query, unscored_candidates_of_that_query)` once per query that has unscored pairs; test fixture `two_candidate_run` (run with one query and chunks `chunk-1`, `chunk-2`).

- [ ] **Step 1: Write the failing test**

In `seed-pipeline/tests/evaluation/conftest.py` add below `OneCandidateRetriever`:

```python
class TwoCandidateRetriever:
    def search(self, query: str, *, limit: int):
        del query, limit
        return [
            RetrievalCandidate(
                chunk_id="chunk-1",
                score=0.75,
                rank=1,
                source="test",
                payload={"chunk_id": "chunk-1", "section_id": "section-1"},
                document_text="candidate document",
            ),
            RetrievalCandidate(
                chunk_id="chunk-2",
                score=0.5,
                rank=2,
                source="test",
                payload={"chunk_id": "chunk-2", "section_id": "section-2"},
                document_text="second candidate document",
            ),
        ]
```

and replace the `complete_run` fixture with:

```python
def _create_run(tmp_path: Path, retriever: object, candidate_k: int) -> Path:
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_path.write_text(
        '{"query_id":"query-1","query":"test query","relevant_section_ids":["section-1"]}\n',
        encoding="utf-8",
    )
    identity = RunIdentity(
        evaluation_path=str(evaluation_path),
        evaluation_sha256=sha256_file(evaluation_path),
        collection_name="collection",
        embedding_model="embeddinggemma:300m",
        query_embeddings_sha256="query-cache",
        retriever="hybrid",
        candidate_k=candidate_k,
        rrf_k=2,
        limit=None,
        prefetch_k=candidate_k,
    )
    root = tmp_path / "run"
    workspace = RunWorkspace.open_or_create(root, identity)
    artifact = build_candidate_artifact(
        rows=[{"query_id": "query-1", "query": "test query"}],
        retriever=retriever,
        output_path=root / "candidates" / "candidates.jsonl",
        identity={"evaluation_sha256": identity.evaluation_sha256},
        candidate_k=candidate_k,
    )
    workspace.record_candidates(artifact)
    return root


@pytest.fixture
def complete_run(tmp_path: Path) -> Path:
    return _create_run(tmp_path, OneCandidateRetriever(), 1)


@pytest.fixture
def two_candidate_run(tmp_path: Path) -> Path:
    return _create_run(tmp_path, TwoCandidateRetriever(), 2)
```

Append to `seed-pipeline/tests/evaluation/test_rerank_service.py`:

```python
class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, candidates):
        self.calls.append((query, [candidate.chunk_id for candidate in candidates]))
        return [
            candidate.with_rerank_score(float(index), index)
            for index, candidate in enumerate(candidates, start=1)
        ]


def test_local_rerank_scores_each_query_in_one_request(two_candidate_run):
    reranker = RecordingReranker()

    result = LocalRerankBackend(reranker_factory=lambda _spec, _timeout: reranker).run(
        request(two_candidate_run, "qwen3-reranker:0.6b-fp16")
    )

    assert reranker.calls == [("test query", ["chunk-1", "chunk-2"])]
    assert "scored=2" in result.actions
    assert result.artifact_dir is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation/test_rerank_service.py::test_local_rerank_scores_each_query_in_one_request`
Expected: FAIL — two calls with one chunk each.

- [ ] **Step 3: Group the local scoring loop by query**

In `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py` replace lines 197–224 (from the comment `# A complete cache must not start a model server…` through `processed += 1`) with:

```python
# A complete cache must not start a model server, so the reranker is created
# only when the first query with an unscored candidate is found.
reranker: Reranker | None = None
processed = 0
for record in CandidateArtifactReader.from_data_path(candidate_bundle.data_path):
    row = {"query_id": record["query_id"], "query": record["query"]}
    candidates = [_candidate(item) for item in record["candidates"]]
    unscored = [
        candidate
        for candidate in candidates
        if cache.key_for(request.model, row, candidate) not in cache.records
    ]
    if not unscored:
        continue
    if reranker is None:
        reranker = self.reranker_factory(spec, request.request_timeout_seconds)
    # One /v1/rerank request per query, like the Kaggle worker and the backend.
    for scored in reranker.rerank(row["query"], unscored):
        cache.set(
            request.model,
            row,
            scored,
            scored.rerank_score or 0.0,
            protocol=spec.reranker_protocol,
        )
    processed += len(unscored)
```

(`--force` already cleared the run's keys with `cache.replace_keys(expected)` above, so every candidate is unscored then.)

- [ ] **Step 4: Run the tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation`
Expected: PASS, including `test_forced_rerank_rescores_into_the_same_variant` (`scored=1`) and `test_local_rerank_from_a_complete_cache_starts_no_reranker`.

- [ ] **Step 5: Run the Seed gate**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py seed-pipeline/tests/evaluation/conftest.py seed-pipeline/tests/evaluation/test_rerank_service.py
git diff --cached --name-status
git commit -m "feat(seed): score each local rerank query in one request" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Local CPU benchmark command and local profiles

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py:60` (add `LOCAL_PROFILE_DIR`)
- Create: `seed-pipeline/src/seed_pipeline/evaluation/local_rerank_benchmark.py`
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py` (imports, `RerankRequest`, `LocalRerankBackend`, `KaggleRerankBackend._run_kaggle`)
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py` (`--benchmark`)
- Modify (docs): `seed-pipeline/docs/guides/cli-reference.md`, `seed-pipeline/docs/guides/evaluation.md` (section 4), `seed-pipeline/docs/guides/workflow-local-only.md` (section 4), `seed-pipeline/docs/guides/downstream.md:21`, `seed-pipeline/README.md:55`, `seed-pipeline/data/README.md`, `README.md`, `.env.example`
- Test: `seed-pipeline/tests/evaluation/test_local_rerank_benchmark.py` (new), `seed-pipeline/tests/evaluation/test_rerank_service.py`, `seed-pipeline/tests/cli/test_rerank_command.py`, `seed-pipeline/tests/config/test_paths.py`, `seed-pipeline/tests/test_repository_data_policy.py`

**Interfaces:**
- Consumes: Task 5 `ModelSpec.local_rerank_search_space`, `LOCAL_RERANK_SEARCH_SPACE`; Task 8 `sample_rerank_groups`, `rerank_groups_from_rows`, `check_score_consistency`, `recommend(objective="latency")`, `describe_levels`, `percentile`, `BenchmarkMeasurement`, `LevelResult`, `LOCAL_RERANK_BENCHMARK_GROUPS`; Task 9 `LlamaCppComposeManager.ensure(..., runtime=)`, `.logs`, `reranker_environment`, `compose_llama_cpp_image`; Task 10 grouped local scoring.
- Produces:
  - `seed_pipeline.config.paths.LOCAL_PROFILE_DIR = CACHE_DIR / "local_profiles"` (profiles at `LOCAL_PROFILE_DIR / "rerank" / "<slug>.json"`)
  - `seed_pipeline.evaluation.local_rerank_benchmark`: `CPUINFO_PATH`, `LOCAL_TOPOLOGY = "cpu_compose"`, `cpu_model_name(cpuinfo_path: Path = CPUINFO_PATH) -> str`, `local_rerank_search_space(spec: ModelSpec) -> RuntimeSearchSpace`, `local_rerank_profile_identity(spec: ModelSpec, *, compose_file: Path = COMPOSE_FILE, cpuinfo_path: Path = CPUINFO_PATH) -> RuntimeProfileIdentity`, `load_local_rerank_profile(spec: ModelSpec, *, profile_root: Path = LOCAL_PROFILE_DIR, compose_file: Path = COMPOSE_FILE, cpuinfo_path: Path = CPUINFO_PATH) -> RuntimeProfile | None`, `RerankServer` (Protocol: `start(runtime) -> str`, `logs() -> str`), `ComposeRerankServer(manager, spec, gguf_root)`, `LocalRerankBenchmarkResult(profile, path, measurements)` with `selected_measurement`, `run_local_rerank_benchmark(*, spec, candidate_data_path, server, client_factory, profile_root=LOCAL_PROFILE_DIR, compose_file=COMPOSE_FILE, cpuinfo_path=CPUINFO_PATH, clock=time.monotonic) -> LocalRerankBenchmarkResult`
  - `LocalRerankBackend(reranker_factory: Callable[[ModelSpec, float], Reranker] | None = None, benchmark_runner: Callable[[ModelSpec, Path, float], LocalRerankBenchmarkResult] | None = None)`; `RerankRequest.benchmark: bool` (existing field) now drives the local benchmark; `RerankRequest.benchmark_pairs` removed. Result actions: `selected=server_slots=<n> ubatch=<n> threads=<n>`, `latency_p95_seconds=<s>`, `env=LLAMA_RERANKER_PARALLEL=… LLAMA_RERANKER_THREADS=… LLAMA_RERANKER_UBATCH_SIZE=…`; `benchmark_report` is the profile path.
  - CLI: `seed rerank --run RUN --backend local --benchmark --model MODEL`. With `--backend kaggle`, `--benchmark` raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/evaluation/test_local_rerank_benchmark.py`:

```python
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from seed_pipeline.evaluation.local_rerank_benchmark import (
    ComposeRerankServer,
    LocalRerankBenchmarkResult,
    cpu_model_name,
    load_local_rerank_profile,
    run_local_rerank_benchmark,
)
from seed_pipeline.runtime.catalog import (
    LOCAL_RERANK_SEARCH_SPACE,
    ModelSpec,
    require_model,
)
from seed_pipeline.runtime.compose import LlamaCppComposeManager, file_sha256

CPU = "13th Gen Intel(R) Core(TM) i5-13420H"
MODEL = "qwen3-reranker:4b-fp16"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingRunner:
    """docker compose stand-in: records every `up` environment, can fail some levels."""

    def __init__(
        self, *, failing_ubatches: frozenset[str] = frozenset(), logs: str = ""
    ):
        self.ups: list[dict[str, str]] = []
        self.failing_ubatches = failing_ubatches
        self.logs = logs

    def run(self, args, env=None, capture_output=False):
        if capture_output:
            return self.logs
        environment = dict(env or {})
        self.ups.append(environment)
        if environment.get("LLAMA_RERANKER_UBATCH_SIZE") in self.failing_ubatches:
            raise subprocess.CalledProcessError(1, list(args))
        return ""


class HealthyClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def health(self) -> None:
        return None


class ScoringClient:
    """Scores by document length; latency and drift depend on the level being served."""

    def __init__(
        self,
        runner: RecordingRunner,
        clock: FakeClock,
        *,
        drifting_ubatch: str | None = None,
    ) -> None:
        self.runner = runner
        self.clock = clock
        self.drifting_ubatch = drifting_ubatch
        self.calls: list[int] = []

    def rerank_native(
        self, query: str, documents: list[str], model: str
    ) -> list[float]:
        del query, model
        level = self.runner.ups[-1]
        self.calls.append(len(documents))
        threads = int(level["LLAMA_RERANKER_THREADS"])
        ubatch = level["LLAMA_RERANKER_UBATCH_SIZE"]
        # ubatch 8192 with 12 threads is the fastest level.
        self.clock.now += (1.0 if ubatch == "8192" else 2.0) * 8 / threads
        drift = 0.01 if ubatch == self.drifting_ubatch else 0.0
        return [len(document) / 1000 + drift for document in documents]


@dataclass(frozen=True)
class Bench:
    root: Path
    spec: ModelSpec
    candidates: Path
    compose_file: Path
    cpuinfo: Path

    def run(
        self, runner: RecordingRunner, client: ScoringClient, clock: FakeClock
    ) -> LocalRerankBenchmarkResult:
        manager = LlamaCppComposeManager(
            self.compose_file,
            runner=runner,
            client_factory=HealthyClient,
            environment={},
        )
        return run_local_rerank_benchmark(
            spec=self.spec,
            candidate_data_path=self.candidates,
            server=ComposeRerankServer(manager, self.spec, self.root),
            client_factory=lambda _endpoint: client,
            profile_root=self.root / "profiles",
            compose_file=self.compose_file,
            cpuinfo_path=self.cpuinfo,
            clock=clock,
        )


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        "services:\n  llama-reranker:\n"
        "    image: ghcr.io/ggml-org/llama.cpp:server-b10920\n",
        encoding="utf-8",
    )
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(f"processor\t: 0\nmodel name\t: {CPU}\n", encoding="utf-8")
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    spec = replace(
        require_model(MODEL),
        canonical_filename=artifact.name,
        byte_size=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )
    candidates = tmp_path / "candidates.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for query in range(8):
            row = {
                "query_id": f"q{query}",
                "query": f"query {query}",
                "candidates": [
                    {
                        "chunk_id": f"c{query}-{document}",
                        "document_text": "x" * (10 * query + document + 1),
                    }
                    for document in range(15)
                ],
            }
            handle.write(json.dumps(row) + "\n")
    return Bench(tmp_path, spec, candidates, compose_file, cpuinfo)


def test_benchmark_recreates_the_reranker_with_each_level_environment(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()

    bench.run(runner, ScoringClient(runner, clock), clock)

    assert [
        (
            up["LLAMA_RERANKER_PARALLEL"],
            up["LLAMA_RERANKER_UBATCH_SIZE"],
            up["LLAMA_RERANKER_THREADS"],
        )
        for up in runner.ups
    ] == [
        ("16", "4096", "8"),
        ("16", "4096", "12"),
        ("16", "8192", "8"),
        ("16", "8192", "12"),
        ("16", "16384", "8"),
        ("16", "16384", "12"),
    ]


def test_each_level_sends_one_warm_up_and_six_full_groups(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()
    client = ScoringClient(runner, clock)

    bench.run(runner, client, clock)

    assert client.calls == [15] * (6 * 7)


def test_lowest_p95_level_becomes_the_local_profile(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()

    result = bench.run(runner, ScoringClient(runner, clock), clock)

    assert result.profile.selected == LOCAL_RERANK_SEARCH_SPACE.candidates[3]
    assert (
        result.path
        == bench.root / "profiles" / "rerank" / "qwen3_reranker_4b_fp16.json"
    )
    assert result.profile.identity.payload["machine_shape"] == CPU
    assert result.profile.sample_count == 90
    assert result.selected_measurement.latency_p95_seconds == pytest.approx(8 / 12)
    assert (
        load_local_rerank_profile(
            bench.spec,
            profile_root=bench.root / "profiles",
            compose_file=bench.compose_file,
            cpuinfo_path=bench.cpuinfo,
        )
        == result.profile
    )


def test_levels_whose_scores_drift_are_invalid(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()

    result = bench.run(
        runner, ScoringClient(runner, clock, drifting_ubatch="16384"), clock
    )

    assert [item.error_category for item in result.measurements] == [
        None,
        None,
        None,
        None,
        "score_mismatch",
        "score_mismatch",
    ]


def test_a_level_that_fails_to_start_keeps_the_service_logs(bench: Bench):
    runner = RecordingRunner(
        failing_ubatches=frozenset({"16384"}), logs="failed to allocate compute buffer"
    )
    clock = FakeClock()

    result = bench.run(runner, ScoringClient(runner, clock), clock)

    failed = result.measurements[4]
    assert (failed.status, failed.error_category) == ("invalid", "CalledProcessError")
    assert failed.log_tail is not None
    assert "failed to allocate compute buffer" in failed.log_tail
    assert result.profile.selected == LOCAL_RERANK_SEARCH_SPACE.candidates[3]


def test_every_level_failing_stops_with_the_log_tails(bench: Bench):
    runner = RecordingRunner(
        failing_ubatches=frozenset({"4096", "8192", "16384"}),
        logs="failed to allocate compute buffer",
    )
    clock = FakeClock()

    with pytest.raises(RuntimeError, match="failed to allocate compute buffer"):
        bench.run(runner, ScoringClient(runner, clock), clock)

    assert not (bench.root / "profiles").exists()


def test_cpu_model_name_reads_proc_cpuinfo_and_falls_back(tmp_path: Path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(f"model name\t: {CPU}\n", encoding="utf-8")

    assert cpu_model_name(cpuinfo) == CPU
    assert cpu_model_name(tmp_path / "missing")
```

Append to `seed-pipeline/tests/evaluation/test_rerank_service.py` (add `from pathlib import Path`, `from seed_pipeline.evaluation.local_rerank_benchmark import LocalRerankBenchmarkResult`, `from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement`, `from seed_pipeline.runtime.catalog import LOCAL_RERANK_SEARCH_SPACE` and `from seed_pipeline.runtime.runtime_profiles import RuntimeProfile, RuntimeProfileIdentity` to the imports):

```python
def test_local_benchmark_reports_the_selected_level_without_registering(
    complete_run,
):
    selected = LOCAL_RERANK_SEARCH_SPACE.candidates[2]
    measurement = BenchmarkMeasurement(
        selected, 90, 1000, 3.0, latency_p50_seconds=0.4, latency_p95_seconds=0.5
    )
    identity = RuntimeProfileIdentity.create(
        workload="rerank",
        model="qwen3-reranker:4b-fp16",
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        inference_cache_policy_sha256="c" * 64,
        machine_shape="cpu",
        topology="cpu_compose",
        search_space=LOCAL_RERANK_SEARCH_SPACE,
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=90,
        measurements=[measurement.to_dict()],
        benchmark_job_sha256="d" * 64,
    )
    report = Path("profiles/rerank/qwen3_reranker_4b_fp16.json")
    seen = []

    def fake_benchmark(spec, candidate_data_path, timeout):
        seen.append((spec.name, candidate_data_path, timeout))
        return LocalRerankBenchmarkResult(profile, report, (measurement,))

    def no_server(_spec, _timeout):
        raise AssertionError("a benchmark must not score the run")

    result = LocalRerankBackend(
        reranker_factory=no_server, benchmark_runner=fake_benchmark
    ).run(replace(request(complete_run, "qwen3-reranker:4b-fp16"), benchmark=True))

    assert seen == [
        (
            "qwen3-reranker:4b-fp16",
            complete_run / "candidates" / "candidates.jsonl",
            5.0,
        )
    ]
    assert result.actions == (
        "selected=server_slots=16 ubatch=8192 threads=8",
        "latency_p95_seconds=0.5",
        "env=LLAMA_RERANKER_PARALLEL=16 LLAMA_RERANKER_THREADS=8 "
        "LLAMA_RERANKER_UBATCH_SIZE=8192",
    )
    assert (result.benchmark_levels, result.benchmark_report) == (1, report)
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_local_reranker_starts_compose_with_the_stored_local_profile(monkeypatch):
    spec = require_model("qwen3-reranker:4b-fp16")
    selected = LOCAL_RERANK_SEARCH_SPACE.candidates[2]
    ensured = []

    class FakeManager:
        def __init__(self, compose_file):
            self.compose_file = compose_file

        def ensure(self, role, ensured_spec, gguf_root, runtime=None):
            del gguf_root
            ensured.append((role, ensured_spec.name, runtime))
            return "http://127.0.0.1:11435"

    monkeypatch.setattr(
        rerank_service,
        "load_local_rerank_profile",
        lambda _spec: SimpleNamespace(selected=selected),
    )
    monkeypatch.setattr(rerank_service, "LlamaCppComposeManager", FakeManager)

    LocalRerankBackend._default_reranker(spec, 5.0)

    assert ensured == [("reranker", spec.name, selected)]


def test_kaggle_backend_leaves_benchmarks_to_the_runtime_profile(complete_run):
    with pytest.raises(ValueError, match="--backend local"):
        KaggleRerankBackend().run(
            replace(request(complete_run, "qwen3-reranker:0.6b-fp16"), benchmark=True)
        )
```

In `seed-pipeline/tests/cli/test_rerank_command.py` replace `test_rerank_benchmark_rejects_local_backend` with:

```python
def test_rerank_benchmark_reaches_the_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "LocalRerankBackend", fake_backend(captured, incomplete=False)
    )

    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "local",
            "--benchmark",
            "--model",
            "qwen3-reranker:4b-fp16",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].benchmark is True
```

In `seed-pipeline/tests/config/test_paths.py` add after the `KAGGLE_PROFILE_DIR` assertion:

```python
    assert paths.LOCAL_PROFILE_DIR == paths.CACHE_DIR / "local_profiles"
```

In `seed-pipeline/tests/test_repository_data_policy.py` add `("data/cache/local_profiles/rerank/model.json", True),` after `("data/cache/rerank_scores/model.jsonl", True),`.

- [ ] **Step 2: Run them to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation/test_local_rerank_benchmark.py tests/evaluation/test_rerank_service.py tests/cli/test_rerank_command.py tests/config/test_paths.py tests/test_repository_data_policy.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'seed_pipeline.evaluation.local_rerank_benchmark'`, `AttributeError: … LOCAL_PROFILE_DIR`, and exit code 2 for `--benchmark`.

- [ ] **Step 3: Add the local profile folder**

In `seed-pipeline/src/seed_pipeline/config/paths.py` add below `KAGGLE_PROFILE_DIR = CACHE_DIR / "kaggle_profiles"`:

```python
LOCAL_PROFILE_DIR = CACHE_DIR / "local_profiles"
```

- [ ] **Step 4: Create the local benchmark module**

Create `seed-pipeline/src/seed_pipeline/evaluation/local_rerank_benchmark.py`:

```python
"""Choose the CPU llama-reranker configuration by measuring it end to end."""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from seed_pipeline.artifacts.jsonl import iter_jsonl_objects
from seed_pipeline.config.paths import COMPOSE_FILE, LOCAL_PROFILE_DIR
from seed_pipeline.runtime.benchmarking import (
    LOCAL_RERANK_BENCHMARK_GROUPS,
    BenchmarkMeasurement,
    LevelResult,
    RerankGroup,
    check_score_consistency,
    describe_levels,
    percentile,
    recommend,
    rerank_groups_from_rows,
    sample_rerank_groups,
)
from seed_pipeline.runtime.catalog import ModelSpec
from seed_pipeline.runtime.compose import (
    LlamaCppComposeManager,
    compose_llama_cpp_image,
)
from seed_pipeline.runtime.runtime_profiles import (
    RuntimeCandidate,
    RuntimeProfile,
    RuntimeProfileIdentity,
    RuntimeProfileStore,
    RuntimeSearchSpace,
    canonical_sha256,
)
from seed_pipeline.runtime.server_policy import inference_cache_policy

CPUINFO_PATH = Path("/proc/cpuinfo")
LOCAL_TOPOLOGY = "cpu_compose"


class RerankServer(Protocol):
    def start(self, runtime: RuntimeCandidate) -> str: ...

    def logs(self) -> str: ...


class NativeRerankClient(Protocol):
    def rerank_native(
        self, query: str, documents: list[str], model: str
    ) -> list[float]: ...


@dataclass(frozen=True)
class ComposeRerankServer:
    """The compose llama-reranker service, recreated for every level."""

    manager: LlamaCppComposeManager
    spec: ModelSpec
    gguf_root: Path

    def start(self, runtime: RuntimeCandidate) -> str:
        return self.manager.ensure(
            "reranker", self.spec, self.gguf_root, runtime=runtime
        )

    def logs(self) -> str:
        return self.manager.logs("reranker")


@dataclass(frozen=True)
class LocalRerankBenchmarkResult:
    profile: RuntimeProfile
    path: Path
    measurements: tuple[BenchmarkMeasurement, ...]

    @property
    def selected_measurement(self) -> BenchmarkMeasurement:
        return next(
            item
            for item in self.measurements
            if item.status == "ok" and item.candidate == self.profile.selected
        )


def cpu_model_name(cpuinfo_path: Path = CPUINFO_PATH) -> str:
    try:
        text = cpuinfo_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        key, _separator, value = line.partition(":")
        if key.strip() == "model name" and value.strip():
            return value.strip()
    return platform.processor() or platform.machine() or "unknown-cpu"


def local_rerank_search_space(spec: ModelSpec) -> RuntimeSearchSpace:
    if spec.local_rerank_search_space is None:
        raise ValueError(f"{spec.name} has no local rerank search space")
    return spec.local_rerank_search_space


def local_rerank_profile_identity(
    spec: ModelSpec,
    *,
    compose_file: Path = COMPOSE_FILE,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> RuntimeProfileIdentity:
    return RuntimeProfileIdentity.create(
        workload="rerank",
        model=spec.name,
        model_sha256=spec.sha256,
        runtime_sha256=canonical_sha256(
            {"image": compose_llama_cpp_image(compose_file)}
        ),
        inference_cache_policy_sha256=inference_cache_policy(spec).sha256,
        machine_shape=cpu_model_name(cpuinfo_path),
        topology=LOCAL_TOPOLOGY,
        search_space=local_rerank_search_space(spec),
    )


def load_local_rerank_profile(
    spec: ModelSpec,
    *,
    profile_root: Path = LOCAL_PROFILE_DIR,
    compose_file: Path = COMPOSE_FILE,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> RuntimeProfile | None:
    identity = local_rerank_profile_identity(
        spec, compose_file=compose_file, cpuinfo_path=cpuinfo_path
    )
    return RuntimeProfileStore(profile_root).load(identity, model_slug=spec.slug)


def run_local_rerank_benchmark(
    *,
    spec: ModelSpec,
    candidate_data_path: Path,
    server: RerankServer,
    client_factory: Callable[[str], NativeRerankClient],
    profile_root: Path = LOCAL_PROFILE_DIR,
    compose_file: Path = COMPOSE_FILE,
    cpuinfo_path: Path = CPUINFO_PATH,
    clock: Callable[[], float] = time.monotonic,
) -> LocalRerankBenchmarkResult:
    space = local_rerank_search_space(spec)
    documents_per_group = space.candidates[0].request_batch_size
    groups = sample_rerank_groups(
        rerank_groups_from_rows(
            iter_jsonl_objects(candidate_data_path),
            documents_per_group=documents_per_group,
        ),
        LOCAL_RERANK_BENCHMARK_GROUPS,
    )
    if not groups:
        raise ValueError(
            f"{candidate_data_path} has no query with {documents_per_group} candidates"
        )
    measurements = check_score_consistency(
        [
            _measure_level(spec, runtime, groups, server, client_factory, clock)
            for runtime in space.candidates
        ]
    )
    selected = recommend(measurements, objective="latency")
    if selected is None:
        raise RuntimeError(
            "no local rerank level produced valid scores:\n"
            + describe_levels(measurements)
        )
    identity = local_rerank_profile_identity(
        spec, compose_file=compose_file, cpuinfo_path=cpuinfo_path
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=sum(len(group.documents) for group in groups),
        measurements=[item.to_dict() for item in measurements],
        benchmark_job_sha256=canonical_sha256(
            {
                "identity_sha256": identity.sha256,
                "query_ids": [group.query_id for group in groups],
            }
        ),
    )
    path = RuntimeProfileStore(profile_root).save(profile, model_slug=spec.slug)
    return LocalRerankBenchmarkResult(profile, path, measurements)


def _measure_level(
    spec: ModelSpec,
    runtime: RuntimeCandidate,
    groups: Sequence[RerankGroup],
    server: RerankServer,
    client_factory: Callable[[str], NativeRerankClient],
    clock: Callable[[], float],
) -> LevelResult:
    latencies: list[float] = []
    scores: dict[str, float] = {}
    try:
        client = client_factory(server.start(runtime))
        # Warm-up: the first group once, untimed, after the service was recreated.
        client.rerank_native(groups[0].query, list(groups[0].documents), spec.name)
        for group in groups:
            started = clock()
            values = client.rerank_native(group.query, list(group.documents), spec.name)
            latencies.append(max(0.0, clock() - started))
            scores.update(zip(group.score_keys(), values, strict=True))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return LevelResult(
            BenchmarkMeasurement.invalid(
                runtime, type(exc).__name__, f"{exc}\n{_server_logs(server)}"
            ),
            {},
        )
    return LevelResult(
        BenchmarkMeasurement(
            runtime,
            sum(len(group.documents) for group in groups),
            sum(group.characters for group in groups),
            sum(latencies),
            latency_p50_seconds=percentile(latencies, 0.50),
            latency_p95_seconds=percentile(latencies, 0.95),
        ),
        scores,
    )


def _server_logs(server: RerankServer) -> str:
    try:
        return server.logs()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return f"server logs unavailable: {exc}"
```

- [ ] **Step 5: Wire the benchmark and the stored profile into the rerank service**

In `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py`:
- add imports:

```python
from seed_pipeline.evaluation.local_rerank_benchmark import (
    ComposeRerankServer,
    LocalRerankBenchmarkResult,
    load_local_rerank_profile,
    run_local_rerank_benchmark,
)
```

  change `from seed_pipeline.runtime.catalog import ModelKind, require_model` to `from seed_pipeline.runtime.catalog import ModelKind, ModelSpec, require_model` and `from seed_pipeline.runtime.compose import LlamaCppComposeManager, resolve_server` to `from seed_pipeline.runtime.compose import LlamaCppComposeManager, reranker_environment`;
- delete the field `benchmark_pairs: int = 512` from `RerankRequest`;
- replace `class LocalRerankBackend:` from its `def __init__` through the end of `_default_reranker` with:

```python
class LocalRerankBackend:
    def __init__(
        self,
        reranker_factory: Callable[[ModelSpec, float], Reranker] | None = None,
        benchmark_runner: Callable[[ModelSpec, Path, float], LocalRerankBenchmarkResult]
        | None = None,
    ):
        self.reranker_factory = reranker_factory or self._default_reranker
        self.benchmark_runner = benchmark_runner or self._default_benchmark

    @staticmethod
    def _default_reranker(spec: ModelSpec, timeout: float) -> Reranker:
        # A stored local benchmark profile sets the llama-reranker level; without one
        # compose defaults (or the root .env) apply.
        profile = load_local_rerank_profile(spec)
        manager = LlamaCppComposeManager(COMPOSE_FILE)
        endpoint = manager.ensure(
            "reranker",
            spec,
            GGUF_ROOT,
            runtime=profile.selected if profile is not None else None,
        )
        return LlamaCppReranker(spec, LlamaCppClient(endpoint, timeout=timeout))

    @staticmethod
    def _default_benchmark(
        spec: ModelSpec, candidate_data_path: Path, timeout: float
    ) -> LocalRerankBenchmarkResult:
        manager = LlamaCppComposeManager(COMPOSE_FILE)
        return run_local_rerank_benchmark(
            spec=spec,
            candidate_data_path=candidate_data_path,
            server=ComposeRerankServer(manager, spec, GGUF_ROOT),
            client_factory=lambda endpoint: LlamaCppClient(endpoint, timeout=timeout),
        )

    def _benchmark(
        self,
        request: RerankRequest,
        spec: ModelSpec,
        candidate_data_path: Path,
        variant_sha256: str,
    ) -> RerankStageResult:
        result = self.benchmark_runner(
            spec, candidate_data_path, request.request_timeout_seconds
        )
        selected = result.profile.selected
        environment = " ".join(
            f"{name}={value}"
            for name, value in sorted(reranker_environment(selected).items())
        )
        return RerankStageResult(
            None,
            variant_sha256,
            None,
            (
                f"selected=server_slots={selected.server_slots} "
                f"ubatch={selected.physical_batch_size} threads={selected.threads}",
                "latency_p95_seconds="
                f"{result.selected_measurement.latency_p95_seconds}",
                f"env={environment}",
            ),
            benchmark_report=result.path,
            benchmark_levels=len(result.measurements),
        )
```

- in `LocalRerankBackend.run` replace

```python
        if request.benchmark:
            raise ValueError("benchmark requires --backend kaggle")
```

  with

```python
        if request.benchmark and request.dry_run:
            raise ValueError("--benchmark cannot be combined with --dry-run")
```

  and insert directly after `identity = RerankVariantIdentity.create(candidate_bundle.manifest.data_sha256, request.model)`:

```python
if request.benchmark:
    return self._benchmark(request, spec, candidate_bundle.data_path, identity.sha256)
```

- in `KaggleRerankBackend._run_kaggle` insert before `spec = require_model(request.model)`:

```python
        if request.benchmark:
            raise ValueError(
                "--benchmark measures the CPU reranker with --backend local; Kaggle "
                "benchmarks run automatically when no runtime profile matches"
            )
```

- [ ] **Step 6: Add the CLI flag**

In `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py` add the parameter after `request_timeout_seconds`:

```python
benchmark: Annotated[
    bool,
    typer.Option(
        "--benchmark",
        help="Measure local llama-reranker levels and store the lowest-p95 profile.",
    ),
] = (False,)
```

and pass `benchmark=benchmark,` to `RerankRequest(...)`.

- [ ] **Step 7: Document native-only rerank, the CPU benchmark and local profiles**

In `seed-pipeline/docs/guides/cli-reference.md`:
- add `seed rerank --run NAME --backend local --benchmark --model MODEL` below `seed rerank --run NAME --backend local|kaggle --model MODEL` in the command list;
- in `## Rerank và metrics`, after the paragraph that ends `khi cache đã đủ, rerank không khởi động model.`, add:

~~~text
Rerank chỉ gọi `POST /v1/rerank` với file GGUF bản convert classifier; mỗi request là một câu hỏi cùng các ứng viên chưa có điểm, ở local lẫn Kaggle. `--backend local` dùng service `llama-reranker` của `../compose.yaml`: nếu `data/cache/local_profiles/rerank/<model>.json` khớp model, CPU (`/proc/cpuinfo`) và image llama.cpp của compose thì service được dựng theo profile, không thì theo mặc định của compose.

`--backend local --benchmark` đo từng mức của search space CPU (`-np` 16; `-ub` 4096/8192/16384, dùng chung cho `-c` và `-b`; `--threads` 8/12). Mỗi mức dựng lại `llama-reranker`, chạy một nhóm khởi động không tính giờ rồi 6 nhóm câu hỏi × 15 ứng viên chọn phân tầng theo tổng số ký tự. Mức có điểm lệch mức hợp lệ đầu tiên quá `1e-3` bị loại; lệnh chọn p95 thấp nhất (bằng nhau thì số cặp/giây cao hơn), lưu profile và in `selected=`, `latency_p95_seconds=`, `env=`. Chép `env=` vào `.env` ở gốc repo và đặt `RERANK_TIMEOUT_SECONDS` ít nhất gấp đôi `latency_p95_seconds`.

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```
~~~

In `seed-pipeline/docs/guides/evaluation.md`, section `## 4. Rerank`, add after the paragraph that starts `` `--dry-run` in `missing_pairs=N` ``:

~~~text
Mọi reranker được gọi qua `/v1/rerank`. Kết quả `bge-reranker-v2-gemma:f16` ở mục 6 là kết quả cuối cùng: model đã bỏ khỏi catalog, report của nó vẫn nằm ở `reports/rerank/bge_reranker_v2_gemma_f16/`.

Cấu hình `llama-reranker` cho backend trên CPU được chọn bằng benchmark đầu-cuối trên máy production (chạy trong tmux: mỗi mức nạp lại model 4B trên CPU):

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```

Lệnh lưu `data/cache/local_profiles/rerank/<model>.json` và in các biến `LLAMA_RERANKER_*` cho `.env` ở gốc repo. Trên Kaggle, benchmark chạy tự động trước lần chấm đầu tiên khi chưa có profile khớp (xem [CLI reference](cli-reference.md)).
~~~

In `seed-pipeline/docs/guides/workflow-local-only.md`, section `## 4. Evaluation`, insert above `uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend local --model qwen3-reranker:4b-fp16`:

```text
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```

and add after that code block (before `Mỗi baseline cần ít nhất 30 candidates/query…`):

```text
`--benchmark` chọn cấu hình `llama-reranker` có p95 thấp nhất trên máy này và lưu profile ở `data/cache/local_profiles/rerank/`; lệnh rerank sau đó dựng service theo profile và gửi một request `/v1/rerank` cho mỗi câu hỏi. Benchmark và rerank cả run chạy lâu, nên chạy trong tmux.
```

In `seed-pipeline/docs/guides/downstream.md` line 21 replace `` và `kaggle_profiles/<workload>/<model>.json`. `` with `` , `kaggle_profiles/<workload>/<model>.json` và `local_profiles/<workload>/<model>.json`. ``

In `seed-pipeline/README.md` line 55 replace `` `data/cache/kaggle_profiles/<workload>/<model>.json`: cache của `seed embed queries`, `seed rerank` và profile runtime Kaggle. `` with `` `data/cache/kaggle_profiles/<workload>/<model>.json`, `data/cache/local_profiles/<workload>/<model>.json`: cache của `seed embed queries`, `seed rerank`, profile runtime Kaggle và profile reranker CPU local. ``

In `seed-pipeline/data/README.md` add under the line `│                 rerank_scores/<model>.jsonl, kaggle_profiles/<workload>/<model>.json`:

```text
│                 local_profiles/<workload>/<model>.json
```

In `README.md` append to the `Reranker trên CPU:` bullet added in Task 9:

```text
  Đo lại cho máy khác trong `seed-pipeline/`:
  `uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16`,
  rồi chép dòng `env=` lệnh in ra vào `.env` và đặt `RERANK_TIMEOUT_SECONDS` ít nhất gấp đôi `latency_p95_seconds`.
```

In `.env.example` add below `# Reranker server: slots (-np), one size for -c/-b/-ub (at least 2048) and threads.`:

```text
# Measure a machine from seed-pipeline/ with
#   uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
# and copy the printed env= values here.
```

- [ ] **Step 8: Run the tests to verify they pass**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation tests/cli tests/config tests/test_repository_data_policy.py tests/test_docs.py`
Expected: PASS (`test_documented_seed_commands_parse` covers the new `--benchmark` lines).

- [ ] **Step 9: Run both gates**

Run the Seed gate and the Backend gate. Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/config/paths.py seed-pipeline/src/seed_pipeline/evaluation/local_rerank_benchmark.py seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py seed-pipeline/src/seed_pipeline/cli/commands/rerank.py seed-pipeline/docs/guides/cli-reference.md seed-pipeline/docs/guides/evaluation.md seed-pipeline/docs/guides/workflow-local-only.md seed-pipeline/docs/guides/downstream.md seed-pipeline/README.md seed-pipeline/data/README.md README.md .env.example seed-pipeline/tests/evaluation/test_local_rerank_benchmark.py seed-pipeline/tests/evaluation/test_rerank_service.py seed-pipeline/tests/cli/test_rerank_command.py seed-pipeline/tests/config/test_paths.py seed-pipeline/tests/test_repository_data_policy.py
git diff --cached --name-status
git commit -m "feat(seed): benchmark the CPU reranker and store local profiles" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| 4.1 seed-pipeline deletions (contracts, prompts, client completion API, worker/benchmark/local branches, cache policy fields, `compare_cache_arms`, `cache_comparison`, catalog `protocol`) | 2 |
| 4.1 Gemma: catalog, `run.json`, artifact, score cache, Kaggle profile; report kept | 2, 3 |
| 4.1 backend deletions, settings `native_rerank | none`, spec and `.env.example` | 4 |
| 4.2 server flags, `-c = -ub`, `-ub` ≥ 2048, candidate fields, derived concurrency, Kaggle and CPU search spaces | 5 (Kaggle command), 9 (compose) |
| 4.3 full-candidate levels, workload, warm-up, restart per level, metrics, `score_mismatch`, selection rules, all-invalid stop, Kaggle and local profiles | 8, 11 |
| 4.4 one request per query group, shared queue, `reuse_payload` without `runtime_profile` | 6, 7 (dependency owner is Plan B) |
| 4.5 local per-query scoring, profile use, `--benchmark` with per-level env, compose defaults, `.env.example` | 9, 10, 11 |
| 4.6 backend `NativeReranker` unchanged, `MAX_CONCURRENT` 1, `MAX_CANDIDATES` 15, `TIMEOUT_SECONDS` | 9 |
| Section 6 unit tests for 4.1–4.6, backend tests, Docker integration test | 2–11 |
| Section 7 docs for 4.1–4.6 | 4, 8, 9, 11 |
