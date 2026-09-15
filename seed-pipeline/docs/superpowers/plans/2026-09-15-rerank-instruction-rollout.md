# Rerank Instruction Experiment and Native Rerank Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> Tasks 1–3 are code tasks and suit subagents. Tasks 4–17 operate on real data, Docker and Kaggle, several need the author's confirmation, and they must run inline with superpowers:executing-plans.

**Goal:** Add a stratified sample to `seed retrieve` and a paired bootstrap to `seed metrics`, choose the Qwen3-Reranker instruction by measurement on 1,000 queries, then rescore `qwen3-reranker` 0.6b, 4b and 8b natively on all 300,000 pairs, measure CPU defaults for production and update the evaluation guide and progress report.

**Architecture:** Two small library additions behind the existing CLI: `load_query_rows` gains a deterministic largest-remainder stratified sample recorded in `RunIdentity`, and a new `evaluation/metric_comparison.py` pairs per-query rows of two published metrics reports and runs a standard-library percentile bootstrap, exposed as `seed metrics compare` (the `metrics` command becomes a Typer group whose callback keeps `seed metrics --run …` unchanged). Everything else is a runbook that calls Plan A (native batched rerank, Kaggle and local benchmarks) and Plan B (`--kaggle-account auto`, partial-artifact merge, persistent logs) commands in tmux, derives GGUF files with the official `gguf-new-metadata` tool, and records measured values in the catalog, compose and docs.

**Tech Stack:** Python 3.12, uv workspace, Typer, pytest (warnings are errors), ruff, pyrefly, `gguf` 0.19.0 (`uv run --no-project --with`, no project dependency), llama.cpp (Kaggle `b9637`, compose `b10920`), Kaggle CLI 2.2.3, Docker Compose, tmux.

**Spec:** seed-pipeline/docs/superpowers/specs/2026-09-15-native-rerank-optimisation-design.md (sections 4.7 and 5)

## Global Constraints

- Execute after Plan A (native-only rerank, batched server, `seed rerank --backend local --benchmark`) and Plan B (`--kaggle-account auto`, partial-artifact merge into the local cache, logs under `data/work/logs/`) are merged and both projects' gates are green.
- f16 only: every GGUF file is an f16 original or derived from one by changing metadata; no quantisation.
- Stage explicit paths only. Never stage the other session's files (`seed-pipeline/src/seed_pipeline/corpus/processing/clean_markdown_corpus.py`, `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py`, `seed-pipeline/pyproject.toml`, `uv.lock`, `seed-pipeline/data/corpus/*/manifest.json`, `seed-pipeline/data/sources/vietnamese_valid_syllables.json`, `seed-pipeline/tests/corpus/`, `seed-pipeline/tests/orchestration/`). `report/report.md` is tracked (the author committed it in `18f5050`); Task 16 commits it with an explicit path.
- No new dependencies: the bootstrap uses `random` and `statistics`; GGUF tooling runs as `uv run --no-project --with gguf==0.19.0 …`.
- Fix lint and type findings in code; no `noqa`, `type: ignore`, `pyrefly: ignore`, rule ignores or relaxed warning filters.
- pytest runs with `filterwarnings = ["error"]`; a warning is a failure.
- **Seed gate** (in `seed-pipeline/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check && uv run pytest -q`
- **Backend gate** (in `backend/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check && uv run pytest -q`
- Commit from the repository root after `git diff --cached --name-status` shows only the task's paths; pre-commit runs on commit (fix and re-stage, never `--no-verify`). Message style `feat(seed): …` (`chore(seed-data): …` for run data, `docs(seed): …` for docs), ending with the line `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Long-running commands run in tmux session `thesis-rerank`, one window per job. Console output is appended to `seed-pipeline/data/work/logs/rollout/<window>.console.log` followed by an `exit=<code>` line; Plan B writes `seed-pipeline/data/work/logs/rerank/<model-slug>.log` itself.
- Stop and ask the author before any Kaggle GPU job or remote deletion, except the full 300,000-pair scoring of the three `qwen3-reranker` models (Task 14), which the author already approved.
- Delete superseded model files, score caches, runtime profiles and checkpoints after the replacement is verified; keep no backups.
- The decision rule of spec 4.7.4 is committed in Task 8, before any sample scoring, and is not changed after measuring.
- Command locations: `uv run seed …` in `seed-pipeline/`, `uv run pharma-agent …` in `backend/`, `docker compose …` and GGUF tooling at the repository root `/home/andv/personal/thesis`.
- Secrets are never printed: Kaggle credentials are read from `seed-pipeline/.env` inside a subshell (`set -a; . ./.env; set +a`).
- Fixed names: sample run `hybrid-qwen4b-p50-k30-rrf2-sample1000`; experimental model `qwen3-reranker:0.6b-fp16-vimed` (slug `qwen3_reranker_0_6b_fp16_vimed`); derived files `ai-models/gguf/qwen3-reranker-{0.6b,4b,8b}-f16-vimed.gguf` (Kaggle dataset slugs of 47, 45 and 45 characters, within Kaggle's 50-character limit).

## File Structure

| Area | Files | Responsibility |
| --- | --- | --- |
| Sample | `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py`, `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py`, `seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py` | `--sample/--sample-seed`, stratified selection, sample in run identity (Task 1) |
| Comparison | `seed-pipeline/src/seed_pipeline/evaluation/metric_comparison.py` (new) | `paired_bootstrap`, `run_metric_comparison` (Task 2) |
| Metrics CLI | `seed-pipeline/src/seed_pipeline/cli/commands/metrics.py`, `seed-pipeline/src/seed_pipeline/cli/app.py` | `seed metrics` group with `compare` (Task 3) |
| Catalog | `seed-pipeline/src/seed_pipeline/runtime/catalog.py`, `seed-pipeline/tests/runtime/test_catalog.py` | Experimental model, then the decided canonical files (Tasks 8, 11, 12) |
| Tests | `seed-pipeline/tests/evaluation/test_backend_retrieval.py`, `seed-pipeline/tests/evaluation/test_run_workspace.py`, `seed-pipeline/tests/cli/test_retrieve_command.py`, `seed-pipeline/tests/evaluation/test_metric_comparison.py` (new), `seed-pipeline/tests/cli/test_metrics_command.py` | Tasks 1–3 |
| Docs | `seed-pipeline/docs/guides/cli-reference.md` (Tasks 1, 3), `seed-pipeline/docs/guides/evaluation.md` (Tasks 8, 10, 11 or 12, 15, 16) | CLI and instruction experiment |
| Config | `compose.yaml`, `.env.example` | Decided reranker file (Task 11), measured CPU defaults (Task 13) |
| Run data (tracked parts) | `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/` (Tasks 5, 15), `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000/` (Tasks 7, 10) | `run.json`, `manifest.json`, `report.md` |
| Report (tracked) | `report/report.md` | Tables 5.1–5.3 and remarks 5.4 (Task 16) |
| One-off scripts (ignored) | `seed-pipeline/data/work/rollout/{retire_stale_variants,score_distribution,report_rows,profile_summary}.py` | Tasks 5, 10, 13, 15 |
| Models (ignored) | `ai-models/gguf/qwen3-reranker-*-f16-vimed.gguf` | Tasks 8, 11 |

---

## Phase 1: Code

### Task 1: Stratified sample for `seed retrieve`

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py` (`RunIdentity`)
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py` (`RetrieveRequest`, `load_query_rows`, `run_retrieval`; new `sample_quotas`, `stratified_sample`)
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py`
- Modify: `seed-pipeline/docs/guides/cli-reference.md`
- Test: `seed-pipeline/tests/evaluation/test_backend_retrieval.py`, `seed-pipeline/tests/evaluation/test_run_workspace.py`, `seed-pipeline/tests/cli/test_retrieve_command.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `seed_pipeline.evaluation.backend_retrieval.load_query_rows(path: Path, limit: int | None = None, *, sample: int | None = None, sample_seed: int = 0) -> list[dict]` (`limit` and `sample` mutually exclusive; stratified by `eval_group`; largest-remainder allocation, remainder ties to the group seen first in the file; output in file order)
  - `seed_pipeline.evaluation.backend_retrieval.sample_quotas(counts: dict[str, int], sample: int) -> dict[str, int]`
  - `seed_pipeline.evaluation.backend_retrieval.stratified_sample(rows: Sequence[dict], sample: int, *, seed: int) -> list[dict]`
  - `RetrieveRequest.sample: int | None = None`, `RetrieveRequest.sample_seed: int = 0`
  - `RunIdentity.sample: int | None = None`, `RunIdentity.sample_seed: int | None = None` (the seed is recorded only when `sample` is set)
  - CLI `seed retrieve --sample N [--sample-seed S]`; `--limit` with `--sample` exits 2

- [ ] **Step 1: Write the failing retrieval tests**

In `seed-pipeline/tests/evaluation/test_backend_retrieval.py`:

Add `from collections import Counter` after `import uuid`, and replace `from seed_pipeline.evaluation.backend_retrieval import RetrieveRequest, run_retrieval` with:

```python
from seed_pipeline.evaluation.backend_retrieval import (
    RetrieveRequest,
    load_query_rows,
    run_retrieval,
    sample_quotas,
)
```

Replace `_evaluation` with:

```python
def _evaluation(tmp_path: Path) -> Path:
    path = tmp_path / "evaluation.jsonl"
    rows = [
        {"query_id": "q1", "query": ADULT, "eval_group": "formulary"},
        {"query_id": "q2", "query": CHILD, "eval_group": "leaflet"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
```

Replace `_request` with:

```python
def _request(
    tmp_path: Path,
    *,
    retriever: str = "hybrid",
    prefetch_k: int | None = 50,
    cache_rows: dict[str, str] | None = None,
    sample: int | None = None,
    sample_seed: int = 0,
) -> RetrieveRequest:
    env_file = tmp_path / "backend.env"
    env_file.write_text(
        f"PHARMA_RETRIEVAL__EMBEDDING__MODEL={MODEL}\n"
        "PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=4\n"
        "PHARMA_LANGFUSE__PUBLIC_KEY=pk\nPHARMA_LANGFUSE__SECRET_KEY=sk\n",
        encoding="utf-8",
    )
    rows = {"q1": ADULT, "q2": CHILD} if cache_rows is None else cache_rows
    return RetrieveRequest(
        evaluation_path=_evaluation(tmp_path),
        run_root=tmp_path / "run",
        retriever=retriever,
        candidate_k=2,
        prefetch_k=prefetch_k,
        rrf_k=2,
        limit=None,
        force=False,
        backend_env_file=env_file,
        query_embeddings=_query_cache(tmp_path, rows),
        sample=sample,
        sample_seed=sample_seed,
    )
```

In `test_hybrid_retrieval_uses_cached_query_vectors`, after `assert identity.query_embeddings_sha256 is not None` add:

```python
    assert (identity.limit, identity.sample, identity.sample_seed) == (None, None, None)
```

Append to the end of the file:

```python
GOLD_GROUP_SIZES = {
    "patient_natural": 500,
    "leaflet": 2500,
    "chunk_risk": 1000,
    "formulary": 5000,
    "noisy_confuser": 500,
    "multi_intent": 500,
}


def _gold(tmp_path: Path, sizes: dict[str, int]) -> Path:
    path = tmp_path / "gold.jsonl"
    rows = [
        {
            "query_id": f"{group}-{number:05d}",
            "query": f"câu hỏi {group} {number}",
            "eval_group": group,
        }
        for group, size in sizes.items()
        for number in range(size)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_sample_takes_ten_percent_of_every_eval_group(tmp_path: Path) -> None:
    rows = load_query_rows(_gold(tmp_path, GOLD_GROUP_SIZES), sample=1000)

    assert Counter(row["eval_group"] for row in rows) == {
        "patient_natural": 50,
        "leaflet": 250,
        "chunk_risk": 100,
        "formulary": 500,
        "noisy_confuser": 50,
        "multi_intent": 50,
    }


def test_sample_rounds_by_largest_remainder_then_file_order() -> None:
    assert sample_quotas({"a": 5, "b": 3, "c": 2}, 3) == {"a": 1, "b": 1, "c": 1}
    assert sample_quotas({"a": 1, "b": 1}, 1) == {"a": 1, "b": 0}


def test_sample_is_repeatable_per_seed_and_keeps_file_order(tmp_path: Path) -> None:
    path = _gold(tmp_path, {"formulary": 100, "leaflet": 100})
    position = {
        row["query_id"]: index for index, row in enumerate(load_query_rows(path))
    }

    first = [row["query_id"] for row in load_query_rows(path, sample=20, sample_seed=0)]
    again = [row["query_id"] for row in load_query_rows(path, sample=20, sample_seed=0)]
    other = [row["query_id"] for row in load_query_rows(path, sample=20, sample_seed=1)]

    assert first == again
    assert first != other
    assert [position[query_id] for query_id in first] == sorted(
        position[query_id] for query_id in first
    )


def test_sample_rejects_limit_oversize_and_rows_without_a_group(tmp_path: Path) -> None:
    path = _gold(tmp_path, {"formulary": 3})
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"query_id": "q1", "query": "câu hỏi"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_query_rows(path, 2, sample=2)
    with pytest.raises(ValueError, match="exceeds the 3 evaluation rows"):
        load_query_rows(path, sample=4)
    with pytest.raises(ValueError, match="eval_group on every row"):
        load_query_rows(plain, sample=1)


def test_sampled_run_records_the_sample_in_its_identity(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    result = run_retrieval(
        _request(tmp_path, sample=1, sample_seed=7), backend_factory=FakeFactory(stack)
    )

    assert result.artifact.query_count == 1
    identity = load_run_record(tmp_path / "run" / "run.json").identity
    assert (identity.limit, identity.sample, identity.sample_seed) == (None, 1, 7)
```

- [ ] **Step 2: Write the failing workspace and CLI tests**

In `seed-pipeline/tests/evaluation/test_run_workspace.py`, add `from dataclasses import replace` after `import json`, and append:

```python
def test_run_json_written_before_sampling_loads_as_unsampled(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    payload = json.loads(workspace.record_path.read_text(encoding="utf-8"))
    del payload["identity"]["sample"], payload["identity"]["sample_seed"]
    workspace.record_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_run_record(workspace.record_path).identity

    assert loaded == identity()
    assert (loaded.sample, loaded.sample_seed) == (None, None)


def test_a_different_sample_seed_is_an_identity_conflict(tmp_path: Path) -> None:
    sampled = replace(identity(), sample=1000, sample_seed=0)
    RunWorkspace.open_or_create(tmp_path / "run", sampled)

    with pytest.raises(RunConflictError, match="sample_seed"):
        RunWorkspace.open_or_create(tmp_path / "run", replace(sampled, sample_seed=1))
```

In `seed-pipeline/tests/cli/test_retrieve_command.py`, in `test_retrieve_passes_the_run_tree_and_backend_defaults` add after `assert request.query_embeddings is None`:

```python
    assert (request.limit, request.sample, request.sample_seed) == (None, None, 0)
```

and append:

```python
def test_retrieve_forwards_the_stratified_sample(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(retrieve_command, "run_retrieval", _fake_run(captured))

    result = runner.invoke(
        app,
        [
            "retrieve",
            "--run",
            "experiment-sample1000",
            "--sample",
            "1000",
            "--sample-seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert (request.limit, request.sample, request.sample_seed) == (None, 1000, 7)


def test_retrieve_rejects_limit_together_with_sample(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(retrieve_command, "run_retrieval", _fake_run(captured))

    result = runner.invoke(
        app, ["retrieve", "--run", "experiment", "--limit", "5", "--sample", "10"]
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    assert captured == {}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation/test_backend_retrieval.py tests/evaluation/test_run_workspace.py tests/cli/test_retrieve_command.py`
Expected: pytest stops at collection with `ImportError: cannot import name 'sample_quotas'` from `test_backend_retrieval.py`. Then run `uv run pytest -q tests/evaluation/test_run_workspace.py tests/cli/test_retrieve_command.py`: the two new workspace tests fail (`KeyError: 'sample'`, `TypeError` from `replace(..., sample=...)`), `test_retrieve_forwards_the_stratified_sample` fails with exit code 2 (`No such option: --sample`), `test_retrieve_rejects_limit_together_with_sample` fails because the output names the unknown option instead of `mutually exclusive`, and `test_retrieve_passes_the_run_tree_and_backend_defaults` fails with `AttributeError` on `request.sample`.

- [ ] **Step 4: Add the sample fields to `RunIdentity`**

In `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py`, replace the `RunIdentity` dataclass with:

```python
@dataclass(frozen=True)
class RunIdentity:
    evaluation_path: str
    evaluation_sha256: str
    collection_name: str
    embedding_model: str
    query_embeddings_sha256: str | None
    retriever: str
    candidate_k: int
    rrf_k: int
    limit: int | None
    prefetch_k: int | None = None
    release_id: str | None = None
    chunker_version: str | None = None
    # Stratified sample of the evaluation file (seed retrieve --sample); None for runs
    # over the whole file or its first `limit` rows, including run.json files written
    # before sampling existed.
    sample: int | None = None
    sample_seed: int | None = None
```

- [ ] **Step 5: Implement the sample in `backend_retrieval.py`**

In `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py`, add `import random` after `import json`, and add two fields at the end of `RetrieveRequest`:

```python
    query_embeddings: Path | None = None
    sample: int | None = None
    sample_seed: int = 0
```

Replace the whole `def load_query_rows(...)` function with:

```python
SAMPLE_STRATUM = "eval_group"


def _read_query_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid evaluation JSON at {path}:{line_number}"
                    ) from exc
                if (
                    not isinstance(row, dict)
                    or not row.get("query_id")
                    or not str(row.get("query") or "").strip()
                ):
                    raise ValueError(
                        "Evaluation row requires query_id and query at "
                        f"{path}:{line_number}"
                    )
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    except OSError as exc:
        raise ValueError(f"Evaluation JSONL is missing: {path}") from exc
    if not rows:
        raise ValueError(f"Evaluation JSONL is empty: {path}")
    return rows


def sample_quotas(counts: dict[str, int], sample: int) -> dict[str, int]:
    """Split `sample` over strata in proportion to their size (largest remainder).

    Remainder ties go to the stratum seen first (dict order is file order), so the
    allocation depends only on the evaluation file and `sample`.
    """
    total = sum(counts.values())
    quotas = {group: sample * count // total for group, count in counts.items()}
    by_remainder = sorted(counts, key=lambda group: -(sample * counts[group] % total))
    for group in by_remainder[: sample - sum(quotas.values())]:
        quotas[group] += 1
    return quotas


def stratified_sample(rows: Sequence[dict], sample: int, *, seed: int) -> list[dict]:
    if sample > len(rows):
        raise ValueError(f"--sample {sample} exceeds the {len(rows)} evaluation rows")
    positions: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        group = row.get(SAMPLE_STRATUM)
        if not isinstance(group, str) or not group:
            raise ValueError(
                f"--sample needs {SAMPLE_STRATUM} on every row; "
                f"query {row['query_id']} has none"
            )
        positions.setdefault(group, []).append(index)
    quotas = sample_quotas(
        {group: len(indices) for group, indices in positions.items()}, sample
    )
    rng = random.Random(seed)
    chosen = sorted(
        index
        for group, indices in positions.items()
        for index in rng.sample(indices, quotas[group])
    )
    return [rows[index] for index in chosen]


def load_query_rows(
    path: Path,
    limit: int | None = None,
    *,
    sample: int | None = None,
    sample_seed: int = 0,
) -> list[dict]:
    if limit is not None and sample is not None:
        raise ValueError("--limit and --sample are mutually exclusive")
    if sample is not None and sample < 1:
        raise ValueError("--sample must be >= 1")
    rows = _read_query_rows(path, limit)
    if sample is None:
        return rows
    return stratified_sample(rows, sample, seed=sample_seed)
```

In `run_retrieval`, replace `rows = load_query_rows(request.evaluation_path, request.limit)` with:

```python
    rows = load_query_rows(
        request.evaluation_path,
        request.limit,
        sample=request.sample,
        sample_seed=request.sample_seed,
    )
```

and in the `RunIdentity(...)` call replace `chunker_version=CHUNKER_VERSION,` with:

```python
chunker_version = (CHUNKER_VERSION,)
sample = (request.sample,)
sample_seed = (request.sample_seed if request.sample is not None else None,)
```

- [ ] **Step 6: Add the CLI options**

In `seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py`, replace `limit: Annotated[int | None, typer.Option("--limit")] = None,` with:

```python
limit: Annotated[
    int | None,
    typer.Option("--limit", help="First N rows of the evaluation file"),
] = (None,)
sample: Annotated[
    int | None,
    typer.Option(
        "--sample",
        help="Stratified sample of N rows, proportional per eval_group",
        callback=lambda _c, _p, value: (
            None if value is None else positive_int(str(value))
        ),
    ),
] = (None,)
sample_seed: Annotated[
    int, typer.Option("--sample-seed", help="Random seed for --sample")
] = (0,)
```

Replace the start of the body `    request = RetrieveRequest(` with:

```python
    if limit is not None and sample is not None:
        raise typer.BadParameter("--limit and --sample are mutually exclusive")
    request = RetrieveRequest(
```

and replace `        query_embeddings=query_embeddings,\n    )` (the end of the `RetrieveRequest(...)` call) with:

```python
        query_embeddings=query_embeddings,
        sample=sample,
        sample_seed=sample_seed,
    )
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q tests/evaluation/test_backend_retrieval.py tests/evaluation/test_run_workspace.py tests/cli/test_retrieve_command.py`
Expected: all tests pass.

- [ ] **Step 8: Document the options**

In `seed-pipeline/docs/guides/cli-reference.md`:

Replace the line `seed retrieve --run NAME` in the command list with `seed retrieve --run NAME [--limit N | --sample N [--sample-seed S]]`.

Replace `` - Run identity: evaluation path/sha256, collection Qdrant, `embedding_model`, `query_embeddings_sha256`, `retriever`, K, `limit`, `release_id`, `chunker_version`. `` with:

```markdown
- Run identity: evaluation path/sha256, collection Qdrant, `embedding_model`, `query_embeddings_sha256`, `retriever`, K, `limit`, `sample`, `sample_seed`, `release_id`, `chunker_version`.
```

After the paragraph that starts `` Candidate `chunk_id` là nhãn vị trí `` insert:

````markdown
`--limit N` lấy N dòng đầu của file gold. `--sample N` lấy mẫu phân tầng theo `eval_group`: mỗi nhóm nhận số câu tỉ lệ với cỡ nhóm, làm tròn theo phần dư lớn nhất (phần dư bằng nhau thì nhóm xuất hiện trước trong file được ưu tiên), chọn ngẫu nhiên trong nhóm bằng `--sample-seed` (mặc định 0) và giữ thứ tự của file. Hai tuỳ chọn loại trừ nhau. Với bộ gold 10.000 câu, `--sample 1000` cho 500 `formulary`, 250 `leaflet`, 100 `chunk_risk`, 50 `patient_natural`, 50 `noisy_confuser`, 50 `multi_intent`.

```bash
uv run seed retrieve --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --sample 1000 --sample-seed 0
```
````

- [ ] **Step 9: Run the seed gate**

Run (in `seed-pipeline/`): `uv run ruff format src tests`, then the **Seed gate**.
Expected: every command exits 0.

- [ ] **Step 10: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py \
  seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py \
  seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py \
  seed-pipeline/tests/evaluation/test_backend_retrieval.py \
  seed-pipeline/tests/evaluation/test_run_workspace.py \
  seed-pipeline/tests/cli/test_retrieve_command.py \
  seed-pipeline/docs/guides/cli-reference.md
git diff --cached --name-status
git commit -m "feat(seed): add a stratified --sample to seed retrieve" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Paired bootstrap comparison of two reranker reports

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/evaluation/metric_comparison.py`
- Test: `seed-pipeline/tests/evaluation/test_metric_comparison.py`

**Interfaces:**
- Consumes (existing code): `seed_pipeline.evaluation.metrics_artifacts.report_dir(run_root, *, top_k, window_size, model=None) -> Path`, `publish_metrics_artifact(...)` (tests only), `seed_pipeline.artifacts.bundle.load_bundle(root, *, expected_type, require_complete)`, `seed_pipeline.artifacts.jsonl.iter_jsonl_objects(path)`. Per-query rows of `metrics.jsonl` have `query_id`, `hit@{3,5,10,30}`, `mrr` (reciprocal rank of the first hit within the top K, i.e. MRR@K) and, for `multi_required` rows, `multi_section_recall@K` and `multi_all_hit@K`; the report manifest identity carries `evaluation_sha256` and `candidate_data_sha256`.
- Produces:
  - `seed_pipeline.evaluation.metric_comparison.BootstrapResult(mean_difference: float, ci_low: float, ci_high: float, resamples: int)` (frozen dataclass)
  - `paired_bootstrap(baseline: Sequence[float], candidate: Sequence[float], *, resamples: int, seed: int) -> BootstrapResult` (difference is `candidate − baseline`; 95% percentile interval; `random.Random(seed)`)
  - `MetricComparisonRequest(run_root: Path, baseline_model: str, candidate_model: str, metric: str = "mrr", top_k: int = DEFAULT_TOP_K, window_size: int = 3, resamples: int = 10_000, seed: int = 0)`
  - `MetricComparison(metric: str, query_count: int, baseline_mean: float, candidate_mean: float, bootstrap: BootstrapResult)` with property `candidate_wins: bool` (`bootstrap.ci_low > 0`)
  - `run_metric_comparison(request: MetricComparisonRequest) -> MetricComparison`

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/evaluation/test_metric_comparison.py`:

```python
from dataclasses import replace
from pathlib import Path

import pytest

from seed_pipeline.evaluation.artifact_contracts import ArtifactContractError
from seed_pipeline.evaluation.metric_comparison import (
    BootstrapResult,
    MetricComparisonRequest,
    paired_bootstrap,
    run_metric_comparison,
)
from seed_pipeline.evaluation.metrics_artifacts import publish_metrics_artifact
from seed_pipeline.evaluation.variant_identity import MetricsArtifactIdentity

BASELINE = "qwen3-reranker:0.6b-fp16"
CANDIDATE = "qwen3-reranker:4b-fp16"
BALANCED = [1.0 if index % 2 else -1.0 for index in range(1000)]


def test_a_constant_improvement_has_a_zero_width_interval() -> None:
    result = paired_bootstrap([0.5, 0.25, 0.0], [1.0, 0.75, 0.5], resamples=200, seed=0)

    assert result == BootstrapResult(0.5, 0.5, 0.5, 200)


def test_a_clear_improvement_puts_the_lower_bound_above_zero() -> None:
    result = paired_bootstrap(
        [0.0] * 1000, [0.5] * 900 + [-0.5] * 100, resamples=2000, seed=0
    )

    assert result.mean_difference == pytest.approx(0.4)
    assert 0 < result.ci_low < result.mean_difference < result.ci_high


def test_balanced_differences_keep_zero_inside_the_interval() -> None:
    result = paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=0)

    assert result.mean_difference == 0.0
    assert result.ci_low < 0 < result.ci_high


def test_the_same_seed_repeats_and_another_seed_resamples_differently() -> None:
    first = paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=0)

    assert paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=0) == first
    assert paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=1) != first


@pytest.mark.parametrize(
    ("baseline", "candidate", "resamples", "message"),
    [
        ([0.1], [0.1, 0.2], 10, "one baseline value per candidate value"),
        ([], [], 10, "at least one query"),
        ([0.1], [0.2], 1, "--resamples must be >= 2"),
    ],
)
def test_invalid_bootstrap_inputs_are_rejected(
    baseline: list[float], candidate: list[float], resamples: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(baseline, candidate, resamples=resamples, seed=0)


def _publish(
    run: Path,
    model: str,
    mrr: dict[str, float],
    *,
    candidate_data_sha256: str = "c" * 64,
) -> None:
    rows = [
        {"query_id": query_id, "hit@10": 1, "mrr": value}
        for query_id, value in mrr.items()
    ]
    publish_metrics_artifact(
        run,
        MetricsArtifactIdentity.create(
            evaluation_sha256="e" * 64,
            candidate_data_sha256=candidate_data_sha256,
            top_k=30,
            window_size=3,
            rerank_variant_sha256=f"{model} variant",
        ),
        {"count": len(rows), "mrr": sum(mrr.values())},
        {"eval_group": {}, "difficulty": {}},
        rows,
        top_k=30,
        window_size=3,
        model=model,
        variant_sha256=f"{model} variant",
    )


def _request(run: Path) -> MetricComparisonRequest:
    return MetricComparisonRequest(
        run, BASELINE, CANDIDATE, metric="mrr", top_k=30, window_size=3, resamples=500
    )


def test_comparison_pairs_the_same_queries_from_both_reports(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5, "q2": 0.0, "q3": 0.5, "q4": 0.0})
    _publish(tmp_path, CANDIDATE, {"q4": 1.0, "q3": 1.0, "q2": 1.0, "q1": 1.0})

    comparison = run_metric_comparison(_request(tmp_path))

    assert (comparison.metric, comparison.query_count) == ("mrr", 4)
    assert (comparison.baseline_mean, comparison.candidate_mean) == (0.25, 1.0)
    assert comparison.bootstrap.mean_difference == 0.75
    assert comparison.bootstrap.ci_low >= 0.5
    assert comparison.candidate_wins is True


def test_equal_reports_keep_the_baseline(tmp_path: Path) -> None:
    same = {"q1": 0.5, "q2": 1.0}
    _publish(tmp_path, BASELINE, same)
    _publish(tmp_path, CANDIDATE, same)

    comparison = run_metric_comparison(_request(tmp_path))

    assert comparison.bootstrap == BootstrapResult(0.0, 0.0, 0.0, 500)
    assert comparison.candidate_wins is False


def test_reports_over_different_candidates_are_not_compared(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5})
    _publish(tmp_path, CANDIDATE, {"q1": 1.0}, candidate_data_sha256="d" * 64)

    with pytest.raises(ArtifactContractError, match="candidate_data_sha256"):
        run_metric_comparison(_request(tmp_path))


def test_reports_over_different_queries_are_not_compared(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5, "q2": 0.5})
    _publish(tmp_path, CANDIDATE, {"q1": 1.0, "q3": 1.0})

    with pytest.raises(ArtifactContractError, match="different queries"):
        run_metric_comparison(_request(tmp_path))


def test_a_missing_metric_and_one_model_twice_are_rejected(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5})
    _publish(tmp_path, CANDIDATE, {"q1": 1.0})

    with pytest.raises(ArtifactContractError, match="Metric hit@3 is missing"):
        run_metric_comparison(replace(_request(tmp_path), metric="hit@3"))
    with pytest.raises(ValueError, match="different rerankers"):
        run_metric_comparison(replace(_request(tmp_path), candidate_model=BASELINE))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation/test_metric_comparison.py`
Expected: collection error `ModuleNotFoundError: No module named 'seed_pipeline.evaluation.metric_comparison'`.

- [ ] **Step 3: Implement the module**

Create `seed-pipeline/src/seed_pipeline/evaluation/metric_comparison.py`:

```python
"""Paired bootstrap comparison of two rerankers on per-query metrics (spec 4.7)."""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.artifacts.bundle import load_bundle
from seed_pipeline.artifacts.jsonl import iter_jsonl_objects
from seed_pipeline.config.defaults import DEFAULT_TOP_K
from seed_pipeline.evaluation.artifact_contracts import ArtifactContractError
from seed_pipeline.evaluation.metrics_artifacts import report_dir

# Both reports must score the same evaluation rows over the same candidates.
MATCHING_REPORT_FIELDS = ("evaluation_sha256", "candidate_data_sha256")


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    resamples: int


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> BootstrapResult:
    """95% percentile interval of mean(candidate - baseline) over resampled queries."""
    if len(baseline) != len(candidate):
        raise ValueError(
            "paired bootstrap needs one baseline value per candidate value "
            f"({len(baseline)} != {len(candidate)})"
        )
    if not baseline:
        raise ValueError("paired bootstrap needs at least one query")
    if resamples < 2:
        raise ValueError("--resamples must be >= 2")
    differences = [
        after - before for before, after in zip(baseline, candidate, strict=True)
    ]
    size = len(differences)
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(differences, k=size)) for _ in range(resamples)
    ]
    # n=40 cut points sit at 2.5%, 5%, ..., 97.5%: the first and last bound the 95% CI.
    cuts = statistics.quantiles(means, n=40, method="inclusive")
    return BootstrapResult(statistics.fmean(differences), cuts[0], cuts[-1], resamples)


@dataclass(frozen=True)
class MetricComparisonRequest:
    run_root: Path
    baseline_model: str
    candidate_model: str
    metric: str = "mrr"
    top_k: int = DEFAULT_TOP_K
    window_size: int = 3
    resamples: int = 10_000
    seed: int = 0


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    query_count: int
    baseline_mean: float
    candidate_mean: float
    bootstrap: BootstrapResult

    @property
    def candidate_wins(self) -> bool:
        """Spec 4.7 decision rule: adopt the candidate only if the CI is above 0."""
        return self.bootstrap.ci_low > 0


def load_query_metric(
    report: Path, metric: str
) -> tuple[dict[str, Any], dict[str, float]]:
    bundle = load_bundle(report, expected_type="metrics_report", require_complete=True)
    values: dict[str, float] = {}
    for row in iter_jsonl_objects(bundle.data_path):
        if metric not in row:
            raise ArtifactContractError(
                f"Metric {metric} is missing for query {row.get('query_id')} "
                f"in {bundle.data_path}"
            )
        values[str(row["query_id"])] = float(row[metric])
    return dict(bundle.manifest.identity), values


def run_metric_comparison(request: MetricComparisonRequest) -> MetricComparison:
    if request.baseline_model == request.candidate_model:
        raise ValueError("--baseline and --candidate must name different rerankers")
    baseline_report = report_dir(
        request.run_root,
        top_k=request.top_k,
        window_size=request.window_size,
        model=request.baseline_model,
    )
    candidate_report = report_dir(
        request.run_root,
        top_k=request.top_k,
        window_size=request.window_size,
        model=request.candidate_model,
    )
    baseline_identity, baseline = load_query_metric(baseline_report, request.metric)
    candidate_identity, candidate = load_query_metric(candidate_report, request.metric)
    for name in MATCHING_REPORT_FIELDS:
        if baseline_identity.get(name) != candidate_identity.get(name):
            raise ArtifactContractError(
                f"Reports {baseline_report} and {candidate_report} differ in {name}; "
                "run seed metrics for both rerankers on the same run"
            )
    if baseline.keys() != candidate.keys():
        raise ArtifactContractError(
            f"Reports {baseline_report} and {candidate_report} cover different queries"
        )
    query_ids = sorted(baseline)
    before = [baseline[query_id] for query_id in query_ids]
    after = [candidate[query_id] for query_id in query_ids]
    bootstrap = paired_bootstrap(
        before, after, resamples=request.resamples, seed=request.seed
    )
    return MetricComparison(
        request.metric,
        len(query_ids),
        statistics.fmean(before),
        statistics.fmean(after),
        bootstrap,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/evaluation/test_metric_comparison.py`
Expected: 12 passed. (10,000 resamples over 1,000 queries take about 0.7 s on the development machine; the tests use at most 2,000.)

- [ ] **Step 5: Run the seed gate**

Run: `uv run ruff format src tests`, then the **Seed gate**.
Expected: every command exits 0.

- [ ] **Step 6: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/evaluation/metric_comparison.py \
  seed-pipeline/tests/evaluation/test_metric_comparison.py
git diff --cached --name-status
git commit -m "feat(seed): add a paired bootstrap comparison of reranker metrics" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `seed metrics compare` command

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/metrics.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/app.py`
- Modify: `seed-pipeline/docs/guides/cli-reference.md`
- Test: `seed-pipeline/tests/cli/test_metrics_command.py`

**Interfaces:**
- Consumes (Task 2): `MetricComparisonRequest`, `MetricComparison`, `BootstrapResult`, `run_metric_comparison(request) -> MetricComparison`.
- Produces:
  - `seed_pipeline.cli.commands.metrics.metrics_app` (Typer group, `invoke_without_command=True`); `seed metrics --run R [--model M] [--top-k K] [--window-size N] [--force]` behaves as before, and without `--run` or a subcommand exits 2.
  - `seed metrics compare --run R --baseline M1 --candidate M2 [--metric mrr] [--top-k 10] [--window-size 3] [--resamples 10000] [--seed 0]`, printing `metric`, `queries`, `baseline` (`model`, `mean`), `candidate` (`model`, `mean`), `mean_difference`, `ci95_low`, `ci95_high`, `resamples`, `seed`, `decision` (the candidate model name when `ci95_low > 0`, otherwise the baseline model name).

- [ ] **Step 1: Write the failing CLI tests**

In `seed-pipeline/tests/cli/test_metrics_command.py`, add after `from seed_pipeline.config.paths import run_dir`:

```python
from seed_pipeline.evaluation.metric_comparison import (
    BootstrapResult,
    MetricComparison,
    MetricComparisonRequest,
)
```

and append:

```python
def fake_comparison(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, MetricComparisonRequest],
    bootstrap: BootstrapResult,
) -> None:
    def fake_run(request: MetricComparisonRequest) -> MetricComparison:
        captured["request"] = request
        return MetricComparison("mrr", 1000, 0.8, 0.81, bootstrap)

    monkeypatch.setattr(metrics_command, "run_metric_comparison", fake_run)


COMPARE = [
    "--json",
    "metrics",
    "compare",
    "--run",
    "sample",
    "--baseline",
    "qwen3-reranker:0.6b-fp16",
    "--candidate",
    "qwen3-reranker:4b-fp16",
    "--top-k",
    "30",
]


def test_metrics_compare_adopts_the_candidate_above_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MetricComparisonRequest] = {}
    fake_comparison(monkeypatch, captured, BootstrapResult(0.01, 0.002, 0.018, 10_000))

    result = runner.invoke(app, COMPARE)

    assert result.exit_code == 0, result.output
    assert captured["request"] == MetricComparisonRequest(
        run_dir("sample"),
        "qwen3-reranker:0.6b-fp16",
        "qwen3-reranker:4b-fp16",
        "mrr",
        30,
        3,
        10_000,
        0,
    )
    details = json.loads(result.stdout)["details"]
    assert details["decision"] == "qwen3-reranker:4b-fp16"
    assert (details["ci95_low"], details["ci95_high"]) == (0.002, 0.018)
    assert details["candidate"] == {"model": "qwen3-reranker:4b-fp16", "mean": 0.81}


def test_metrics_compare_keeps_the_baseline_when_zero_is_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MetricComparisonRequest] = {}
    fake_comparison(monkeypatch, captured, BootstrapResult(0.01, -0.001, 0.02, 10_000))

    result = runner.invoke(app, COMPARE)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["details"]["decision"] == (
        "qwen3-reranker:0.6b-fp16"
    )


def test_metrics_without_run_or_subcommand_is_a_usage_error() -> None:
    result = runner.invoke(app, ["metrics"])

    assert result.exit_code == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/cli/test_metrics_command.py`
Expected: the two `compare` tests fail with `AttributeError: <module 'seed_pipeline.cli.commands.metrics' …> has no attribute 'run_metric_comparison'`; `test_metrics_without_run_or_subcommand_is_a_usage_error` and the existing tests pass (today `--run` is simply required).

- [ ] **Step 3: Turn `metrics` into a group with `compare`**

Replace the whole of `seed-pipeline/src/seed_pipeline/cli/commands/metrics.py` with:

```python
from __future__ import annotations

from typing import Annotated

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import DEFAULT_TOP_K
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.metric_comparison import (
    MetricComparisonRequest,
    run_metric_comparison,
)
from seed_pipeline.evaluation.metrics_service import MetricsRequest, run_metrics

metrics_app = typer.Typer(add_completion=False, invoke_without_command=True)


@metrics_app.callback()
def metrics(
    ctx: typer.Context,
    run: Annotated[str | None, typer.Option("--run")] = None,
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    model: Annotated[str | None, typer.Option("--model")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Write metrics reports for a run; `compare` tests two rerankers."""
    if ctx.invoked_subcommand is not None:
        return
    if run is None:
        raise typer.BadParameter(
            "required unless a subcommand is given", param_hint="'--run'"
        )
    request = MetricsRequest(run_dir(run), top_k, window_size, model, force)
    run_handler(state_from_context(ctx), lambda: _run(request))


def _run(request: MetricsRequest) -> CommandResult:
    result = run_metrics(request)
    return CommandResult(
        "metrics",
        CommandStatus.COMPLETE,
        result.baseline.artifact_dir,
        {
            "baseline": {
                "artifact": str(result.baseline.artifact_dir),
                "metrics_sha256": result.baseline.metrics_sha256,
            },
            "reranked": [
                {
                    "model": item.model,
                    "variant_sha256": item.variant_sha256,
                    "metrics_sha256": item.metrics_sha256,
                    "artifact": str(item.artifact_dir),
                }
                for item in sorted(result.reranked, key=lambda item: item.model or "")
            ],
        },
    )


@metrics_app.command("compare")
def compare(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")],
    baseline: Annotated[
        str, typer.Option("--baseline", help="Reranker the decision starts from")
    ],
    candidate: Annotated[
        str, typer.Option("--candidate", help="Reranker adopted only if it wins")
    ],
    metric: Annotated[
        str, typer.Option("--metric", help="Per-query key of metrics.jsonl")
    ] = "mrr",
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    resamples: Annotated[int, typer.Option("--resamples")] = 10_000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Paired bootstrap of candidate minus baseline over the run's queries.

    The candidate wins only when the 95% interval's lower bound is above 0.
    """
    request = MetricComparisonRequest(
        run_dir(run), baseline, candidate, metric, top_k, window_size, resamples, seed
    )
    run_handler(state_from_context(ctx), lambda: _compare(request))


def _compare(request: MetricComparisonRequest) -> CommandResult:
    comparison = run_metric_comparison(request)
    bootstrap = comparison.bootstrap
    winner = request.candidate_model if comparison.candidate_wins else None
    return CommandResult(
        "metrics compare",
        CommandStatus.COMPLETE,
        request.run_root,
        {
            "metric": comparison.metric,
            "queries": comparison.query_count,
            "baseline": {
                "model": request.baseline_model,
                "mean": comparison.baseline_mean,
            },
            "candidate": {
                "model": request.candidate_model,
                "mean": comparison.candidate_mean,
            },
            "mean_difference": bootstrap.mean_difference,
            "ci95_low": bootstrap.ci_low,
            "ci95_high": bootstrap.ci_high,
            "resamples": bootstrap.resamples,
            "seed": request.seed,
            "decision": winner or request.baseline_model,
        },
    )
```

In `seed-pipeline/src/seed_pipeline/cli/app.py`, replace `from seed_pipeline.cli.commands.metrics import metrics` with `from seed_pipeline.cli.commands.metrics import metrics_app`, and replace `app.command("metrics")(metrics)` with `app.add_typer(metrics_app, name="metrics")`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/cli/test_metrics_command.py tests/evaluation/test_metric_comparison.py`
Expected: all pass, including the unchanged `test_metrics_rejects_removed_options` (exit 2 for `--variant` and `--output-dir`).

Run: `uv run seed metrics compare --help`
Expected: lists `--run`, `--baseline`, `--candidate`, `--metric`, `--top-k`, `--window-size`, `--resamples`, `--seed`.

- [ ] **Step 5: Document the command**

In `seed-pipeline/docs/guides/cli-reference.md`, replace the command-list line `seed metrics --run NAME` with:

```text
seed metrics --run NAME
seed metrics compare --run NAME --baseline MODEL --candidate MODEL
```

After the paragraph that starts `Report nằm ở \`reports/baseline/top<K>-window<N>/\`` insert:

````markdown
`seed metrics compare` so sánh hai reranker đã có report cùng cutoff trên cùng run. Lệnh ghép từng câu hỏi trong `metrics.jsonl` của hai report, bootstrap theo cặp (lấy mẫu lại có hoàn lại các câu hỏi `--resamples` lần bằng `--seed`) cho hiệu trung bình `candidate − baseline` của `--metric` (mặc định `mrr`, tức MRR@K), rồi in trung bình của từng model, hiệu, khoảng tin cậy 95% (`ci95_low`, `ci95_high`, phân vị 2,5% và 97,5%) và `decision`. `decision` là `--candidate` khi `ci95_low > 0`, ngược lại là `--baseline`. Lệnh dừng nếu hai report khác evaluation, candidates hoặc tập câu hỏi.

```bash
uv run seed metrics compare --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --baseline qwen3-reranker:0.6b-fp16 --candidate qwen3-reranker:0.6b-fp16-vimed --metric mrr --top-k 30 --resamples 10000 --seed 0
```
````

- [ ] **Step 6: Run the seed gate**

Run: `uv run ruff format src tests`, then the **Seed gate**.
Expected: every command exits 0.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/cli/commands/metrics.py \
  seed-pipeline/src/seed_pipeline/cli/app.py \
  seed-pipeline/tests/cli/test_metrics_command.py \
  seed-pipeline/docs/guides/cli-reference.md
git diff --cached --name-status
git commit -m "feat(seed): add seed metrics compare for the rerank decision rule" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: Rollout runbook

Job window convention used by every long-running step (the example is Task 14's 4b job):

```bash
tmux new-window -t thesis-rerank -n full-4b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:full-4b 'uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account auto 2>&1 | tee -a data/work/logs/rollout/full-4b.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/full-4b.console.log' Enter
```

The window keeps its shell after the command ends. Check progress with `tail -n 5 data/work/logs/rollout/<window>.console.log` (and, for rerank jobs, `tail -n 20 data/work/logs/rerank/<model-slug>.log`). Exit codes: `0` complete, `3` resumable incomplete, `130` detached from a still-running kernel, `1` failure.

**Resume after an interruption** (WSL restart, closed terminal, exit `130`): run Task 4 Step 8 to recreate the session, then for every job window whose console log does not end with `exit=0`, recreate the window and send the same command again. Rerank jobs reattach to running kernels and continue from the local score cache and checkpoints (Plan B).

### Task 4: Rollout preconditions

**Files:** none changed.

**Interfaces:**
- Consumes (Plan A): `seed rerank --backend kaggle --model M --run R` (benchmarks automatically when no valid profile exists, then scores with batched `/v1/rerank`), `seed rerank --backend local --benchmark --model M --run R` (prints the chosen configuration, writes `data/cache/local_profiles/rerank/<slug>.json`), native-only catalog without `bge-reranker-v2-gemma:f16`.
- Consumes (Plan B): `--kaggle-account auto`, partial artifacts merged into `data/cache/rerank_scores/<slug>.jsonl` on every session, logs at `data/work/logs/rerank/<slug>.log`, exit 3 with a quota table and `refreshAt` when no account has 1 hour of GPU quota.
- Consumes (Tasks 1–3): `seed retrieve --sample/--sample-seed`, `seed metrics compare`.
- Produces: tmux session `thesis-rerank`, directories `seed-pipeline/data/work/rollout/` and `seed-pipeline/data/work/logs/rollout/`, the confirmed derived file names.

- [ ] **Step 1: Check the tree**

```bash
cd /home/andv/personal/thesis
git log --oneline -40
git status --short -- seed-pipeline/src/seed_pipeline/runtime/catalog.py compose.yaml .env.example seed-pipeline/docs/guides seed-pipeline/data/evaluation/runs seed-pipeline/tests/runtime
```

Expected: the log contains the commits of Plan A, Plan B and Tasks 1–3; the status command prints nothing. If it lists a file, another session is editing it: stop and ask the author.

- [ ] **Step 2: Run both gates**

Run the **Seed gate** in `seed-pipeline/` and the **Backend gate** in `backend/`.
Expected: every command exits 0.

- [ ] **Step 3: Check the consumed interfaces**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run seed rerank --help
uv run seed retrieve --help | grep -E -- "--sample"
uv run seed metrics compare --help | grep -E -- "--resamples"
uv run python - <<'EOF'
from seed_pipeline.runtime.catalog import MODEL_CATALOG, require_model

assert "bge-reranker-v2-gemma:f16" not in MODEL_CATALOG
for name in ("qwen3-reranker:0.6b-fp16", "qwen3-reranker:4b-fp16", "qwen3-reranker:8b-fp16"):
    spec = require_model(name)
    contract = getattr(spec, "rerank_contract", None)
    print(name, spec.canonical_filename, spec.sha256[:12], contract.sha256 if contract else None)
EOF
```

Expected: `seed rerank --help` lists `--benchmark` and a `--kaggle-account` whose help mentions `auto`; the two greps print one line each; the script prints `qwen3-reranker-0.6b-f16.gguf fa726a72c1af`, `qwen3-reranker-4b-f16.gguf c4de2e3e4179`, `qwen3-reranker-8b-f16.gguf a53322f79360`, each followed by the native contract sha256. If any check fails, stop and ask the author.

- [ ] **Step 4: Check the Kaggle dataset slug length of the derived files**

```bash
uv run python - <<'EOF'
from dataclasses import replace

from seed_pipeline.runtime.catalog import require_model

for size in ("0.6b", "4b", "8b"):
    spec = replace(
        require_model(f"qwen3-reranker:{size}-fp16"),
        canonical_filename=f"qwen3-reranker-{size}-f16-vimed.gguf",
    )
    print(len(spec.gguf_dataset_slug), spec.gguf_dataset_slug)
EOF
```

Expected: `47 vector-cache-gguf-qwen3-reranker-0-6b-f16-vimed`, `45 vector-cache-gguf-qwen3-reranker-4b-f16-vimed`, `45 vector-cache-gguf-qwen3-reranker-8b-f16-vimed`. Kaggle dataset slugs are limited to 50 characters (the limit `CheckpointService.reference` already enforces); the `-vimed` suffix was chosen because `-vi-medical` gives 52 characters for 0.6b. If any printed length is above 50, stop and ask the author.

- [ ] **Step 5: Check the progress report**

```bash
cd /home/andv/personal/thesis
test -f report/report.md && grep -n "^### 5\." report/report.md
```

Expected: `### 5.1 Kết quả tổng`, `### 5.2 Theo nhóm câu hỏi (Hit@10)`, `### 5.3 Tốc độ reranker (Kaggle T4, mẫu 512 cặp)`, `### 5.4 Nhận xét`. The file is tracked since commit `18f5050`; if it is missing, the branch does not contain that commit: stop and ask the author.

- [ ] **Step 6: Check disk space**

Run: `df -h /home/andv/personal/thesis`
Expected: at least 30 GB available (derived 4b and 8b files add 23.2 GB before the originals are deleted). Otherwise stop and ask the author.

- [ ] **Step 7: Record the Kaggle GPU quota**

```bash
cd /home/andv/personal/thesis/seed-pipeline
( set -a; . ./.env; set +a
  for n in 1 2 3; do
    user_var="KAGGLE_ACC${n}_USERNAME"; token_var="KAGGLE_ACC${n}_API_TOKEN"
    echo "== acc${n}"
    env -u KAGGLE_KEY KAGGLE_USERNAME="${!user_var}" KAGGLE_API_TOKEN="${!token_var}" \
      uv run kaggle quota -v | grep -E "^(resource|GPU)"
  done )
```

Expected: a `resource,used,remaining,total,refreshAt` header and one `GPU,…` line per account. Keep the output in the task notes; Task 9 and Task 14 compare against it.

- [ ] **Step 8: Create the tmux session and log folders**

```bash
command -v tmux
mkdir -p /home/andv/personal/thesis/seed-pipeline/data/work/rollout /home/andv/personal/thesis/seed-pipeline/data/work/logs/rollout
tmux has-session -t thesis-rerank 2>/dev/null || tmux new-session -d -s thesis-rerank -n shell -c /home/andv/personal/thesis/seed-pipeline
tmux list-windows -t thesis-rerank
```

Expected: a tmux path, and a window list that contains `shell`.

---

### Task 5: Retire the completion-era Qwen3 variants of the rrf2 run

`run.json` of `hybrid-qwen4b-p50-k30-rrf2` still registers `qwen3-reranker` 0.6b, 4b and 8b variants scored with `completion_logprobs` (4b with the old mradermacher file). Their identities no longer match the native catalog, so `seed rerank` and `seed rerank --dry-run` stop with `RunConflictError` until they are unregistered. Their `reports/rerank/<slug>/` stay until Task 15 replaces them, so the committed numbers keep matching the report meanwhile.

**Files:**
- Create (ignored): `seed-pipeline/data/work/rollout/retire_stale_variants.py`
- Modify: `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json`
- Delete: `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/qwen3_reranker_{0_6b,4b,8b}_fp16/`

**Interfaces:**
- Consumes: `load_run_record`, `RunWorkspace`, `RunRecord`, `RerankVariantIdentity.create(candidate_data_sha256, model)`, `load_bundle`, `MODEL_CATALOG`.
- Produces: a run whose registered variants all match the catalog, so Tasks 6 and 14 can run `seed rerank` on it.

- [ ] **Step 1: Create the script**

Create `seed-pipeline/data/work/rollout/retire_stale_variants.py`:

```python
"""Unregister rrf2 rerank variants that no longer match the catalog. One-off."""

from __future__ import annotations

import shutil
import sys

from seed_pipeline.artifacts.bundle import load_bundle
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunRecord,
    RunWorkspace,
    load_run_record,
)
from seed_pipeline.evaluation.variant_identity import RerankVariantIdentity
from seed_pipeline.runtime.catalog import MODEL_CATALOG

RUN = run_dir("hybrid-qwen4b-p50-k30-rrf2")
RESCORED = {
    "qwen3-reranker:0.6b-fp16",
    "qwen3-reranker:4b-fp16",
    "qwen3-reranker:8b-fp16",
}


def main(apply: bool) -> None:
    record = load_run_record(RUN / "run.json")
    if record.candidates_dir is None:
        raise SystemExit(f"STOP: {RUN} has no candidates")
    workspace = RunWorkspace(RUN, record.identity, record.origin)
    candidates = load_bundle(
        workspace.resolve(record.candidates_dir),
        expected_type="retrieval_candidates",
        require_complete=True,
    )
    keep: dict[str, RerankVariantRecord] = {}
    retire: list[str] = []
    for slug, variant in sorted(record.rerank_variants.items()):
        if variant.model not in MODEL_CATALOG:
            raise SystemExit(
                f"STOP: {variant.model} is in run.json but not in the catalog"
            )
        current = RerankVariantIdentity.create(
            candidates.manifest.data_sha256, variant.model
        )
        if current.sha256 == variant.variant_sha256:
            keep[slug] = variant
        elif variant.model in RESCORED:
            retire.append(slug)
        else:
            raise SystemExit(
                f"STOP: {variant.model} no longer matches its registered variant; "
                "ask the author"
            )
    print(f"retire: {', '.join(retire) or 'nothing to retire'}")
    print(f"keep: {', '.join(sorted(keep)) or 'none'}")
    if not apply or not retire:
        return
    for slug in retire:
        if workspace.rerank_dir(slug).exists():
            shutil.rmtree(workspace.rerank_dir(slug))
    workspace.write_record(
        RunRecord(record.identity, record.origin, record.candidates_dir, keep)
    )
    print("applied")


if __name__ == "__main__":
    main(apply=sys.argv[1:] == ["--apply"])
```

- [ ] **Step 2: Preview**

Run (in `seed-pipeline/`): `uv run python data/work/rollout/retire_stale_variants.py`
Expected: `retire: qwen3_reranker_0_6b_fp16, qwen3_reranker_4b_fp16, qwen3_reranker_8b_fp16` and `keep: bge_reranker_v2_m3_f16`. If it prints `retire: nothing to retire` (Plan A already did this), skip to Task 6. If it prints `STOP`, ask the author; in particular a `bge-reranker-v2-m3:f16` mismatch means Plan A changed the native contract hash and the m3 results would need rescoring.

- [ ] **Step 3: Apply**

Run: `uv run python data/work/rollout/retire_stale_variants.py --apply`
Expected: the same two lines, then `applied`. Running the preview again prints `retire: nothing to retire`.

- [ ] **Step 4: Verify the run**

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --model qwen3-reranker:4b-fp16 --dry-run
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --model bge-reranker-v2-m3:f16 --top-k 30
git -C /home/andv/personal/thesis status --short -- seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2
```

Expected: the dry run prints `missing_pairs=300000` without starting a container; metrics completes and reuses the m3 report; the status shows `M …/run.json` and `D …/rerank/qwen3_reranker_{0_6b,4b,8b}_fp16/manifest.json` only.

- [ ] **Step 5: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json
git add -u seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank
git diff --cached --name-status
git commit -m "chore(seed-data): retire completion-era Qwen3 rerank variants of the rrf2 run" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Recover the 94,950 pairs of kernel `rerank-5f22fcadeede1072`

The 0.6b kernel of 2026-09-15 stopped after scoring 94,950 of 300,000 pairs natively with the original template. Its job identity predates Plan A, so neither the reconcile path nor the checkpoint slugs find it. Plan B's `seed rerank --recover-kernel` downloads its output and merges the records into the local score cache.

**Files:** none tracked.
- Produces (ignored): `seed-pipeline/data/cache/rerank_scores/qwen3_reranker_0_6b_fp16.jsonl`, `seed-pipeline/data/work/logs/rollout/recover-0_6b.console.log`

**Interfaces:**
- Consumes (Plan B Task 10): `seed rerank --backend kaggle --run R --model M --recover-kernel OWNER/KERNEL-SLUG`. It refuses a kernel that is not `COMPLETE` or `ERROR`, merges only records whose model and request-contract digests match the catalog (`native_rerank_contract().sha256` is frozen at `95b81f733a6695906ec4b9c0a30ab9588dc1f43bd9e64e45800101c87ee0eb48` by Plan A), and reports the actions `kernel=<ref>`, `recovered_pairs=N`, `missing_pairs=N`.
- Produces: 94,950 cached pairs that Task 9 and Task 14 reuse when the original template stays.

- [ ] **Step 1: Run the recovery in tmux**

```bash
tmux new-window -t thesis-rerank -n recover-0_6b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:recover-0_6b 'L=data/work/logs/rollout/recover-0_6b.console.log; ( uv run seed --json rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:0.6b-fp16 --recover-kernel doanvanan0209/rerank-5f22fcadeede1072 ) 2>&1 | tee -a "$L"; echo "exit=${PIPESTATUS[0]}" | tee -a "$L"' Enter
```

Expected in `data/work/logs/rollout/recover-0_6b.console.log`: the JSON result lists `kernel=doanvanan0209/rerank-5f22fcadeede1072`, `recovered_pairs=94950` and `missing_pairs=205050`, followed by `exit=0`. On `Prompt contract mismatch` or `Model digest mismatch` nothing was merged: ask the author whether Task 14 should rescore those pairs instead; do not edit records. If Kaggle reports the kernel output as unavailable, ask the author; Task 14 then scores those pairs again.

- [ ] **Step 2: Verify**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --model qwen3-reranker:0.6b-fp16 --dry-run
```

Expected: `missing_pairs=205050`, printed without starting a container.

---

### Task 7: Build the 1,000-query sample run

**Files:**
- Create: `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000/` (tracked: `run.json`, `candidates/manifest.json`)

**Interfaces:**
- Consumes (Task 1): `seed retrieve --sample 1000 --sample-seed 0`; the evaluation stack of `seed-pipeline/docs/guides/evaluation.md` sections 2–3; the query vector cache `data/cache/query_embeddings/qwen3_embedding_4b_fp16.jsonl`.
- Produces: run `hybrid-qwen4b-p50-k30-rrf2-sample1000` with identity `sample=1000`, `sample_seed=0`, 1,000 queries and up to 30,000 pairs (Tasks 9–10).

- [ ] **Step 1: Start the evaluation stack**

```bash
cd /home/andv/personal/thesis
POSTGRES_PORT=5434 QDRANT_HTTP_PORT=6335 QDRANT_GRPC_PORT=6336 docker compose -p thesis-eval up -d --wait postgres qdrant
```

Expected: both containers report `Healthy`.

- [ ] **Step 2: Import and publish the corpus for `qwen3-embedding:4b-fp16`**

```bash
tmux new-window -t thesis-rerank -n sample-retrieve -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:sample-retrieve 'export PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis PHARMA_QDRANT__URL=http://localhost:6335 PHARMA_RETRIEVAL__EMBEDDING__MODEL=qwen3-embedding:4b-fp16 PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=2560 PHARMA_RETRIEVAL__QDRANT_COLLECTION=eval_qwen3_embedding_4b_fp16' Enter
tmux send-keys -t thesis-rerank:sample-retrieve '(cd ../backend && uv run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish) 2>&1 | tee -a data/work/logs/rollout/sample-retrieve.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/sample-retrieve.console.log' Enter
```

Expected: the console log ends with `exit=0` and the import reports a published release. On any other exit, stop and ask the author.

- [ ] **Step 3: Retrieve the sample**

```bash
tmux send-keys -t thesis-rerank:sample-retrieve 'uv run seed retrieve --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --sample 1000 --sample-seed 0 2>&1 | tee -a data/work/logs/rollout/sample-retrieve.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/sample-retrieve.console.log' Enter
```

Expected (a few minutes): `status=complete`, `queries=1000`, then `exit=0`. After an interruption, send the same command again.

- [ ] **Step 4: Verify the sample**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run python - <<'EOF'
import json
from collections import Counter
from pathlib import Path

run = Path("data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000")
identity = json.loads((run / "run.json").read_text(encoding="utf-8"))["identity"]
print({key: identity[key] for key in ("retriever", "prefetch_k", "candidate_k", "rrf_k", "limit", "sample", "sample_seed")})
gold = {}
with open("data/evaluation/gold/section_retrieval_eval.jsonl", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            gold[row["query_id"]] = row["eval_group"]
groups, pairs = Counter(), 0
with open(run / "candidates" / "candidates.jsonl", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            record = json.loads(line)
            groups[gold[record["query_id"]]] += 1
            pairs += len(record["candidates"])
print(dict(groups), f"pairs={pairs}")
EOF
```

Expected: `{'retriever': 'hybrid', 'prefetch_k': 50, 'candidate_k': 30, 'rrf_k': 2, 'limit': None, 'sample': 1000, 'sample_seed': 0}`, then `{'patient_natural': 50, 'leaflet': 250, 'chunk_risk': 100, 'formulary': 500, 'noisy_confuser': 50, 'multi_intent': 50} pairs=30000`. A pair count slightly below 30,000 is acceptable only if some query returned fewer than 30 hits; note the exact number, because Task 9 checks against it.

- [ ] **Step 5: Stop the evaluation stack**

```bash
cd /home/andv/personal/thesis
docker compose -p thesis-eval down -v
```

- [ ] **Step 6: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000/run.json \
  seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000/candidates/manifest.json
git diff --cached --name-status
git commit -m "chore(seed-data): add the 1,000-query stratified sample of the rrf2 run" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Derive the vimed 0.6b GGUF, add the experimental model and commit the decision rule

The derivation was rehearsed on 2026-09-15 with `gguf` 0.19.0: two runs produced identical files of 1,197,634,336 bytes with sha256 `fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851`, only `tokenizer.chat_template.rerank` differed, and all 311 tensors were byte-identical. The steps below must reproduce exactly these values.

**Files:**
- Create (ignored): `ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf`, `seed-pipeline/data/work/vimed/qwen3-reranker-0.6b-f16.chat-templates.json`
- Modify: `seed-pipeline/src/seed_pipeline/runtime/catalog.py`
- Create or modify: `seed-pipeline/tests/runtime/test_catalog.py`
- Modify: `seed-pipeline/docs/guides/evaluation.md`

**Interfaces:**
- Consumes: `ai-models/gguf/qwen3-reranker-0.6b-f16.gguf` (sha256 `fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa`), `RERANKER_MODELS["qwen3-reranker:0.6b-fp16"]` as left by Plan A.
- Produces: catalog model `qwen3-reranker:0.6b-fp16-vimed` (same runtime, contract and topology as `qwen3-reranker:0.6b-fp16`; file `qwen3-reranker-0.6b-f16-vimed.gguf`, 1,197,634,336 bytes, sha256 `fa17b7c7…f851`); the committed decision rule in `evaluation.md`.

- [ ] **Step 1: Check the inputs**

```bash
cd /home/andv/personal/thesis
sha256sum ai-models/gguf/qwen3-reranker-0.6b-f16.gguf
test ! -e ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf && echo "output absent"
```

Expected: `fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa` and `output absent`. `gguf-new-metadata` asks for confirmation before overwriting, so a leftover output must be deleted first (`rm ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf`).

- [ ] **Step 2: Write the chat template list**

```bash
cd /home/andv/personal/thesis
SIZE=0.6b
mkdir -p seed-pipeline/data/work/vimed
uv run --no-project --with gguf==0.19.0 python - \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  seed-pipeline/data/work/vimed/qwen3-reranker-${SIZE}-f16.chat-templates.json <<'EOF'
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
```

Expected: `wrote seed-pipeline/data/work/vimed/qwen3-reranker-0.6b-f16.chat-templates.json`. Save the Python part of this heredoc (between `<<'EOF'` and `EOF`) as `seed-pipeline/data/work/vimed/build_chat_templates.py` too, because Task 11 runs it for 4b and 8b. The list carries `default` because `gguf-new-metadata --chat-template` replaces every `tokenizer.chat_template*` key.

- [ ] **Step 3: Derive the file**

```bash
cd /home/andv/personal/thesis
SIZE=0.6b
uv run --no-project --with gguf==0.19.0 gguf-new-metadata \
  --chat-template "$(cat seed-pipeline/data/work/vimed/qwen3-reranker-${SIZE}-f16.chat-templates.json)" \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16-vimed.gguf
```

Expected: `n_tensors = 311, total_size = 1.2G` and a finished `Writing` bar.

- [ ] **Step 4: Verify that only the rerank template changed**

```bash
cd /home/andv/personal/thesis
SIZE=0.6b
uv run --no-project --with gguf==0.19.0 python - \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16-vimed.gguf <<'EOF'
import hashlib
import sys

from gguf import GGUFReader

OLD_LINE = "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
NEW_LINE = "<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query\n"
TEMPLATE_KEYS = {"tokenizer.chat_template", "tokenizer.chat_template.rerank", "tokenizer.chat_templates"}


def fail(message):
    raise SystemExit(f"STOP: {message}")


def metadata(reader):
    return {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith("GGUF.")
    }


def tensor_sha256(path, tensor):
    digest = hashlib.sha256()
    remaining = int(tensor.n_bytes)
    with open(path, "rb") as handle:
        handle.seek(int(tensor.data_offset))
        while remaining:
            block = handle.read(min(remaining, 16 << 20))
            if not block:
                fail(f"{path} ends inside tensor {tensor.name}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


source_path, derived_path = sys.argv[1], sys.argv[2]
source, derived = GGUFReader(source_path), GGUFReader(derived_path)
before, after = metadata(source), metadata(derived)
changed = sorted(
    name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
)
if not set(changed) <= TEMPLATE_KEYS:
    fail(f"metadata outside the chat templates changed: {changed}")
if after.get("tokenizer.chat_template") != before["tokenizer.chat_template"]:
    fail("the default chat template changed")
if after.get("tokenizer.chat_templates") != ["rerank"]:
    fail("tokenizer.chat_templates is not ['rerank']")
expected = before["tokenizer.chat_template.rerank"].replace(OLD_LINE, NEW_LINE)
if NEW_LINE not in expected or after.get("tokenizer.chat_template.rerank") != expected:
    fail("the rerank template is not the original one with the Vietnamese medical instruction")
architecture = after["general.architecture"]
if after.get(f"{architecture}.pooling_type") != 4:
    fail("pooling_type is not rank (4)")
source_tensors = {tensor.name: tensor for tensor in source.tensors}
derived_tensors = {tensor.name: tensor for tensor in derived.tensors}
if "cls.output.weight" not in derived_tensors:
    fail("cls.output.weight is missing")
if list(source_tensors) != list(derived_tensors):
    fail("tensor names or order differ")
for name, tensor in source_tensors.items():
    other = derived_tensors[name]
    if tensor.tensor_type != other.tensor_type or list(tensor.shape) != list(other.shape):
        fail(f"tensor {name} changed type or shape")
    if tensor_sha256(source_path, tensor) != tensor_sha256(derived_path, other):
        fail(f"tensor {name} data differs")
print(f"verified tensors={len(source_tensors)} changed_metadata={changed}")
EOF
```

Expected: `verified tensors=311 changed_metadata=['tokenizer.chat_template.rerank']`. Save this heredoc as `seed-pipeline/data/work/vimed/verify.py` too (same content), because Task 11 runs it for 4b and 8b.

- [ ] **Step 5: Record size and digest**

```bash
cd /home/andv/personal/thesis
stat -c %s ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf
sha256sum ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf
```

Expected: `1197634336` and `fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851`. Any other value means a different tool version or source file: stop and ask the author.

- [ ] **Step 6: Write the failing catalog test**

If `seed-pipeline/tests/runtime/test_catalog.py` exists (Plan A may have created it), append the test below and add the import if missing; otherwise create the file with this content:

```python
from seed_pipeline.runtime.catalog import MODEL_CATALOG, require_model

VIMED = "qwen3-reranker:0.6b-fp16-vimed"


def test_vimed_reranker_is_the_0_6b_runtime_with_its_own_file() -> None:
    base = require_model("qwen3-reranker:0.6b-fp16")
    variant = require_model(VIMED)

    assert variant.name == VIMED
    assert variant.slug == "qwen3_reranker_0_6b_fp16_vimed"
    assert variant.canonical_filename == "qwen3-reranker-0.6b-f16-vimed.gguf"
    assert (variant.byte_size, variant.sha256) == (
        1_197_634_336,
        "fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851",
    )
    assert variant.kind is base.kind
    assert variant.topology is base.topology
    assert variant.rerank_contract == base.rerank_contract
    assert variant.rerank_search_space == base.rerank_search_space


def test_every_gguf_dataset_slug_fits_the_kaggle_limit() -> None:
    too_long = {
        name: spec.gguf_dataset_slug
        for name, spec in MODEL_CATALOG.items()
        if len(spec.gguf_dataset_slug) > 50
    }

    assert too_long == {}
```

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_catalog.py`
Expected: `test_vimed_reranker_is_the_0_6b_runtime_with_its_own_file` FAILS with `ValueError` (or the catalog's unknown-model error) for `qwen3-reranker:0.6b-fp16-vimed`; `test_every_gguf_dataset_slug_fits_the_kaggle_limit` passes (it guards the entry added next).

- [ ] **Step 7: Add the catalog entry**

In `seed-pipeline/src/seed_pipeline/runtime/catalog.py`, add `replace` to the `dataclasses` import (`from dataclasses import dataclass, replace`), and insert directly above `MODEL_CATALOG = {**EMBEDDING_MODELS, **RERANKER_MODELS}`:

```python
# Instruction experiment (docs/guides/evaluation.md): the 0.6b weights with the rerank
# chat template changed to a Vietnamese medical instruction by gguf-new-metadata.
RERANKER_MODELS["qwen3-reranker:0.6b-fp16-vimed"] = replace(
    RERANKER_MODELS["qwen3-reranker:0.6b-fp16"],
    name="qwen3-reranker:0.6b-fp16-vimed",
    canonical_filename="qwen3-reranker-0.6b-f16-vimed.gguf",
    byte_size=1_197_634_336,
    sha256="fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851",
)
```

Run: `uv run pytest -q tests/runtime/test_catalog.py`
Expected: PASS.

- [ ] **Step 8: Document the experiment and fix the decision rule**

In `seed-pipeline/docs/guides/evaluation.md`, insert a new section directly before the `## …` heading that contains `dense-text-embedding-3-large-k30`. Number it one higher than the rerank section and renumber the later `## N.` headings by one. With the numbering of 2026-09-15 it is `## 5.` and the following headings become `## 6.` and `## 7.`:

````markdown
## 5. Đo instruction trên tập con

Qwen3-Reranker đọc instruction từ template `rerank` trong file GGUF. Template gốc dùng `<Instruct>: Given a web search query, retrieve relevant passages that answer the query`. Phép đo so bản gốc với bản chỉ đổi dòng đó thành `<Instruct>: Given a Vietnamese medical retrieval query, retrieve relevant passages that answer the query`.

Quy tắc quyết định được chốt trước khi đo: bootstrap theo cặp trên 1.000 câu hỏi của tập con, 10.000 lần lấy mẫu lại, seed 0, cho hiệu MRR@30 (bản tiếng Việt − bản gốc). Cận dưới khoảng tin cậy 95% lớn hơn 0 thì cả ba model `qwen3-reranker` dùng instruction tiếng Việt; ngược lại giữ template gốc.

### Tập con

Dùng stack của mục 2 và biến môi trường của mục 3, rồi trong `seed-pipeline/`:

```bash
uv run seed retrieve --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --sample 1000 --sample-seed 0
```

Mẫu gồm 10% mỗi `eval_group`: 500 `formulary`, 250 `leaflet`, 100 `chunk_risk`, 50 `patient_natural`, 50 `noisy_confuser`, 50 `multi_intent`, tổng 30.000 cặp.

### File GGUF tiếng Việt y tế

Chạy ở thư mục gốc repo. `gguf` 0.19.0 chỉ chạy tạm qua `uv run --no-project --with`, không phải dependency của project. `gguf-new-metadata --chat-template` thay mọi khoá `tokenizer.chat_template*`, nên bước đầu ghi danh sách JSON gồm template `default` chép từ file gốc và template `rerank` đã đổi instruction.

```bash
SIZE=0.6b
mkdir -p seed-pipeline/data/work/vimed
uv run --no-project --with gguf==0.19.0 python - \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  seed-pipeline/data/work/vimed/qwen3-reranker-${SIZE}-f16.chat-templates.json <<'EOF'
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
  --chat-template "$(cat seed-pipeline/data/work/vimed/qwen3-reranker-${SIZE}-f16.chat-templates.json)" \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16.gguf \
  ai-models/gguf/qwen3-reranker-${SIZE}-f16-vimed.gguf
sha256sum ai-models/gguf/qwen3-reranker-${SIZE}-f16-vimed.gguf
```

Lệnh cho kết quả tất định. So với file gốc, mọi tensor giống từng byte và chỉ khoá `tokenizer.chat_template.rerank` khác.

| File | Kích thước (byte) | sha256 |
| --- | ---: | --- |
| `qwen3-reranker-0.6b-f16-vimed.gguf` | 1.197.634.336 | `fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851` |

Catalog gọi file này là `qwen3-reranker:0.6b-fp16-vimed`.

### Chấm và so sánh

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend kaggle --model qwen3-reranker:0.6b-fp16 --kaggle-account auto
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend kaggle --model qwen3-reranker:0.6b-fp16-vimed --kaggle-account auto
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --top-k 30
uv run seed metrics compare --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --baseline qwen3-reranker:0.6b-fp16 --candidate qwen3-reranker:0.6b-fp16-vimed --metric mrr --top-k 30 --resamples 10000 --seed 0
```
````

- [ ] **Step 9: Run the seed gate and commit**

Run the **Seed gate** in `seed-pipeline/`. Expected: every command exits 0.

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/runtime/catalog.py \
  seed-pipeline/tests/runtime/test_catalog.py \
  seed-pipeline/docs/guides/evaluation.md
git diff --cached --name-status
git commit -m "feat(seed): add the vimed Qwen3 0.6b reranker and fix the instruction decision rule" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Kaggle benchmark and sample scoring of both 0.6b files (ask first)

This task covers spec section 5 step 3 (the new 0.6b benchmark runs automatically before the first scoring session) and the scoring of spec 4.7.3.

**Files:**
- Produces (tracked, committed in Task 10): `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000/run.json`, `…/rerank/qwen3_reranker_0_6b_fp16/manifest.json`, `…/rerank/qwen3_reranker_0_6b_fp16_vimed/manifest.json`
- Produces (ignored): `seed-pipeline/data/cache/rerank_scores/qwen3_reranker_0_6b_fp16{,_vimed}.jsonl`, `seed-pipeline/data/cache/kaggle_profiles/rerank/qwen3_reranker_0_6b_fp16{,_vimed}.json`

**Interfaces:**
- Consumes (Plan A): `seed rerank --backend kaggle --model M --run R` with automatic benchmark; (Plan B): `--kaggle-account auto`, model dataset creation by `KAGGLE_SHARED_OWNER`, logs `data/work/logs/rerank/<slug>.log`; (Task 8): catalog model `qwen3-reranker:0.6b-fp16-vimed`; (Task 7): the sample run.
- Produces: complete rerank variants of both 0.6b files on the sample run.

- [ ] **Step 1: Count the missing pairs without starting anything**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend local --model qwen3-reranker:0.6b-fp16 --dry-run
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend local --model qwen3-reranker:0.6b-fp16-vimed --dry-run
```

Expected: the first `missing_pairs` is at most the Task 7 pair count (sample pairs identical to pairs recovered in Task 6 are already cached); the second equals the Task 7 pair count. No container starts.

- [ ] **Step 2: Ask the author**

Ask: "Start the Kaggle instruction A/B now? It runs two `seed rerank --backend kaggle --kaggle-account auto` jobs on the 1,000-query sample: `qwen3-reranker:0.6b-fp16` (<missing from Step 1> pairs) and `qwen3-reranker:0.6b-fp16-vimed` (<pairs from Step 1>). Each job first runs the new Kaggle benchmark (three `-ub` levels × 960 pairs plus warm-up); the vimed file is benchmarked separately because the profile identity includes the model sha256. The first vimed job uploads a 1.2 GB model dataset owned by the shared account." Stop until the author says yes.

- [ ] **Step 3: Launch both jobs**

```bash
tmux new-window -t thesis-rerank -n sample-0.6b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:sample-0.6b 'uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend kaggle --model qwen3-reranker:0.6b-fp16 --kaggle-account auto 2>&1 | tee -a data/work/logs/rollout/sample-0.6b.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/sample-0.6b.console.log' Enter
tmux new-window -t thesis-rerank -n sample-vimed -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:sample-vimed 'uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --backend kaggle --model qwen3-reranker:0.6b-fp16-vimed --kaggle-account auto 2>&1 | tee -a data/work/logs/rollout/sample-vimed.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/sample-vimed.console.log' Enter
```

- [ ] **Step 4: Monitor until both windows end with `exit=0`**

```bash
cd /home/andv/personal/thesis/seed-pipeline
tail -n 3 data/work/logs/rollout/sample-0.6b.console.log data/work/logs/rollout/sample-vimed.console.log
tail -n 20 data/work/logs/rerank/qwen3_reranker_0_6b_fp16.log data/work/logs/rerank/qwen3_reranker_0_6b_fp16_vimed.log
```

Expected: Plan B's logs show the chosen account, quota before and after each session, the benchmark levels and scoring progress; both console logs end with `exit=0`. On `exit=3` with a quota table, wait until the printed `refreshAt` and send the same command again. On `exit=130` or after an interruption, send the same command again. On `exit=1`, show the last 50 lines of the model log to the author and stop.

- [ ] **Step 5: Verify the variants and profiles**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

run = Path("data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000")
variants = json.loads((run / "run.json").read_text(encoding="utf-8"))["rerank_variants"]
for slug in ("qwen3_reranker_0_6b_fp16", "qwen3_reranker_0_6b_fp16_vimed"):
    manifest = json.loads((run / "rerank" / slug / "manifest.json").read_text(encoding="utf-8"))
    profile = Path("data/cache/kaggle_profiles/rerank") / f"{slug}.json"
    print(
        slug,
        variants[slug]["model"],
        manifest["complete"],
        manifest["total"],
        manifest["identity"]["model_sha256"][:12],
        profile.is_file(),
    )
EOF
```

Expected: `qwen3_reranker_0_6b_fp16 qwen3-reranker:0.6b-fp16 30000 30000 fa726a72c1af True` and `qwen3_reranker_0_6b_fp16_vimed qwen3-reranker:0.6b-fp16-vimed 30000 30000 fa17b7c742ff True` (with the Task 7 pair count instead of 30000 if it was lower).

---

### Task 10: Decide the instruction

**Files:**
- Create (ignored): `seed-pipeline/data/work/rollout/score_distribution.py`, `seed-pipeline/data/work/logs/rollout/instruction-decision.json`
- Produces (tracked): reports of the sample run
- Modify: `seed-pipeline/docs/guides/evaluation.md`

**Interfaces:**
- Consumes (Task 3): `seed metrics compare`; (Task 9): both sample variants.
- Produces: the decision `vimed` (Task 11 runs, Task 12 is skipped) or `original` (Task 12 runs, Task 11 is skipped); `score_distribution.py` for Task 15.

- [ ] **Step 1: Compute the sample metrics**

Run (in `seed-pipeline/`): `uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --top-k 30`
Expected: `status=complete` with two `reranked` entries, and the folders `reports/baseline/top30-window3/`, `reports/rerank/qwen3_reranker_0_6b_fp16/top30-window3/`, `reports/rerank/qwen3_reranker_0_6b_fp16_vimed/top30-window3/` under the sample run.

- [ ] **Step 2: Check both score distributions**

Create `seed-pipeline/data/work/rollout/score_distribution.py`:

```python
"""Check that Qwen3 rerank scores spread from near 0 to near 1 (spec 5, step 7). One-off."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

LOW, HIGH, MEDIAN_SPREAD = 0.05, 0.95, 0.9


def main(path: str) -> int:
    by_query: defaultdict[str, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_query[str(record["query_id"])].append(float(record["score"]))
    scores = sorted(score for values in by_query.values() for score in values)
    spreads = [max(values) - min(values) for values in by_query.values()]
    cuts = statistics.quantiles(scores, n=100, method="inclusive")
    median_spread = statistics.median(spreads)
    print(
        f"pairs={len(scores)} queries={len(by_query)} min={scores[0]:.4f} "
        f"p01={cuts[0]:.4f} p50={cuts[49]:.4f} p99={cuts[98]:.4f} "
        f"max={scores[-1]:.4f} median_query_spread={median_spread:.4f}"
    )
    # Qwen3 rerank scores are softmax probabilities: they must stay inside [0, 1].
    passed = (
        0.0 <= scores[0] <= LOW
        and HIGH <= scores[-1] <= 1.0
        and median_spread > MEDIAN_SPREAD
    )
    print("distribution=PASS" if passed else "distribution=FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
```

On the completion-era files of 2026-09-15 this script printed `distribution=FAIL` for 8b (scores 0.3098–0.9121, median spread 0.3936) and `distribution=PASS` for 4b (0.0000–1.0000, median spread 0.9948).

```bash
cd /home/andv/personal/thesis/seed-pipeline
for slug in qwen3_reranker_0_6b_fp16 qwen3_reranker_0_6b_fp16_vimed; do
  uv run python data/work/rollout/score_distribution.py data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000/rerank/$slug/rerank_scores.jsonl
done
```

Expected: `distribution=PASS` for both. A `FAIL` means a scoring problem (for example a broken template) that invalidates the comparison: stop and ask the author; do not run Step 3.

- [ ] **Step 3: Apply the decision rule**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run seed --json metrics compare --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --baseline qwen3-reranker:0.6b-fp16 --candidate qwen3-reranker:0.6b-fp16-vimed --metric mrr --top-k 30 --resamples 10000 --seed 0 | tee data/work/logs/rollout/instruction-decision.json
uv run python - <<'EOF'
import json

with open("data/work/logs/rollout/instruction-decision.json", encoding="utf-8") as handle:
    details = json.load(handle)["details"]
assert (details["queries"], details["resamples"], details["seed"]) == (1000, 10000, 0)
print("MRR@30 original", round(details["baseline"]["mean"], 4))
print("MRR@30 vimed", round(details["candidate"]["mean"], 4))
print("difference", round(details["mean_difference"], 4))
print("ci95", round(details["ci95_low"], 4), round(details["ci95_high"], 4))
winner = details["decision"] == "qwen3-reranker:0.6b-fp16-vimed"
print("DECISION", "vimed" if winner else "original")
EOF
```

Expected: the assertion holds and the last line is `DECISION vimed` or `DECISION original`. The rule is fixed: do not rerun with other parameters.

- [ ] **Step 4: Record the result in the guide**

Append to the instruction section of `seed-pipeline/docs/guides/evaluation.md` (after `### Chấm và so sánh` and its code block):

```markdown
### Kết quả

| Biến thể | Hit@10 | MRR@30 |
| --- | ---: | ---: |
| Template gốc (`qwen3-reranker:0.6b-fp16`) | {HIT10_ORIGINAL} | {MRR_ORIGINAL} |
| Tiếng Việt y tế (`qwen3-reranker:0.6b-fp16-vimed`) | {HIT10_VI} | {MRR_VI} |

Hiệu MRR@30 (tiếng Việt − gốc) là {DIFFERENCE}, khoảng tin cậy 95% [{CI_LOW}; {CI_HIGH}] (1.000 câu, 10.000 lần lấy mẫu lại, seed 0). Quyết định: {DECISION_TEXT}.
```

Fill the braces as follows, with Vietnamese decimal commas: `{HIT10_*}` and `{MRR_*}` are the `| Hit@10 |` and `| MRR |` rows of `reports/rerank/qwen3_reranker_0_6b_fp16/top30-window3/report.md` and `reports/rerank/qwen3_reranker_0_6b_fp16_vimed/top30-window3/report.md` of the sample run (for example `96,18%`, `0,7823`); `{DIFFERENCE}`, `{CI_LOW}`, `{CI_HIGH}` are the Step 3 values rounded to 4 decimals; `{DECISION_TEXT}` is `dùng instruction tiếng Việt cho cả ba model` for `DECISION vimed`, otherwise `giữ template gốc`. No brace may remain in the file (`grep -n "{" seed-pipeline/docs/guides/evaluation.md` shows only lines inside code blocks).

- [ ] **Step 5: Commit the evidence**

The sample run is kept as evidence and is not recomputed after Task 11 or 12 (one of its variants then names a model that no longer exists or points to another file). Its ignored `metrics.jsonl` and score files stay on disk and are archived in Task 17, so the bootstrap can be recomputed from them.

```bash
cd /home/andv/personal/thesis
R=seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2-sample1000
git add $R/run.json \
  $R/rerank/qwen3_reranker_0_6b_fp16/manifest.json \
  $R/rerank/qwen3_reranker_0_6b_fp16_vimed/manifest.json \
  $R/reports/baseline/top30-window3/manifest.json \
  $R/reports/baseline/top30-window3/report.md \
  $R/reports/rerank/qwen3_reranker_0_6b_fp16/top30-window3/manifest.json \
  $R/reports/rerank/qwen3_reranker_0_6b_fp16/top30-window3/report.md \
  $R/reports/rerank/qwen3_reranker_0_6b_fp16_vimed/top30-window3/manifest.json \
  $R/reports/rerank/qwen3_reranker_0_6b_fp16_vimed/top30-window3/report.md \
  seed-pipeline/docs/guides/evaluation.md
git diff --cached --name-status
git commit -m "docs(seed): record the rerank instruction decision on the 1,000-query sample" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Apply the decision when the Vietnamese instruction wins

Run this task only after `DECISION vimed` in Task 10; otherwise skip to Task 12.

**Files:**
- Create (ignored): `ai-models/gguf/qwen3-reranker-4b-f16-vimed.gguf`, `ai-models/gguf/qwen3-reranker-8b-f16-vimed.gguf`
- Modify: `seed-pipeline/src/seed_pipeline/runtime/catalog.py`, `seed-pipeline/tests/runtime/test_catalog.py`, `compose.yaml`, `.env.example`, `seed-pipeline/docs/guides/evaluation.md`, and every other file listed by the grep in Step 6
- Delete (ignored): `ai-models/gguf/qwen3-reranker-{0.6b,4b,8b}-f16.gguf`, score caches, Kaggle profiles and staging folders of the original-template files and of the experiment model

**Interfaces:**
- Consumes (Task 8): `seed-pipeline/data/work/vimed/build_chat_templates.py`, `seed-pipeline/data/work/vimed/verify.py`.
- Produces: catalog `qwen3-reranker:{0.6b,4b,8b}-fp16` pointing at the `-vimed` files; no `qwen3-reranker:0.6b-fp16-vimed` entry; compose default reranker file `qwen3-reranker-4b-f16-vimed.gguf` (used by Tasks 13–16).

- [ ] **Step 1: Check the inputs**

```bash
cd /home/andv/personal/thesis
df -h /home/andv/personal/thesis
sha256sum ai-models/gguf/qwen3-reranker-4b-f16.gguf ai-models/gguf/qwen3-reranker-8b-f16.gguf
ls ai-models/gguf/qwen3-reranker-4b-f16-vimed.gguf ai-models/gguf/qwen3-reranker-8b-f16-vimed.gguf 2>&1
```

Expected: at least 24 GB available; `c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117` and `a53322f7936010458424a12f0f6d22291547e42fa85c16dd4730244d659cea96`; both `ls` targets reported as missing.

- [ ] **Step 2: Derive and verify the 4b and 8b files**

```bash
tmux new-window -t thesis-rerank -n derive-4b -c /home/andv/personal/thesis
tmux send-keys -t thesis-rerank:derive-4b 'L=seed-pipeline/data/work/logs/rollout/derive-4b.console.log; ( uv run --no-project --with gguf==0.19.0 python seed-pipeline/data/work/vimed/build_chat_templates.py ai-models/gguf/qwen3-reranker-4b-f16.gguf seed-pipeline/data/work/vimed/qwen3-reranker-4b-f16.chat-templates.json && uv run --no-project --with gguf==0.19.0 gguf-new-metadata --chat-template "$(cat seed-pipeline/data/work/vimed/qwen3-reranker-4b-f16.chat-templates.json)" ai-models/gguf/qwen3-reranker-4b-f16.gguf ai-models/gguf/qwen3-reranker-4b-f16-vimed.gguf && uv run --no-project --with gguf==0.19.0 python seed-pipeline/data/work/vimed/verify.py ai-models/gguf/qwen3-reranker-4b-f16.gguf ai-models/gguf/qwen3-reranker-4b-f16-vimed.gguf && stat -c %s ai-models/gguf/qwen3-reranker-4b-f16-vimed.gguf && sha256sum ai-models/gguf/qwen3-reranker-4b-f16-vimed.gguf ) 2>&1 | tee -a $L; echo "exit=${PIPESTATUS[0]}" | tee -a $L' Enter
tmux new-window -t thesis-rerank -n derive-8b -c /home/andv/personal/thesis
tmux send-keys -t thesis-rerank:derive-8b 'L=seed-pipeline/data/work/logs/rollout/derive-8b.console.log; ( uv run --no-project --with gguf==0.19.0 python seed-pipeline/data/work/vimed/build_chat_templates.py ai-models/gguf/qwen3-reranker-8b-f16.gguf seed-pipeline/data/work/vimed/qwen3-reranker-8b-f16.chat-templates.json && uv run --no-project --with gguf==0.19.0 gguf-new-metadata --chat-template "$(cat seed-pipeline/data/work/vimed/qwen3-reranker-8b-f16.chat-templates.json)" ai-models/gguf/qwen3-reranker-8b-f16.gguf ai-models/gguf/qwen3-reranker-8b-f16-vimed.gguf && uv run --no-project --with gguf==0.19.0 python seed-pipeline/data/work/vimed/verify.py ai-models/gguf/qwen3-reranker-8b-f16.gguf ai-models/gguf/qwen3-reranker-8b-f16-vimed.gguf && stat -c %s ai-models/gguf/qwen3-reranker-8b-f16-vimed.gguf && sha256sum ai-models/gguf/qwen3-reranker-8b-f16-vimed.gguf ) 2>&1 | tee -a $L; echo "exit=${PIPESTATUS[0]}" | tee -a $L' Enter
```

Expected in each console log: `wrote …chat-templates.json`, a finished `Writing` bar, `verified tensors=<N> changed_metadata=['tokenizer.chat_template.rerank']`, the byte size, the sha256, then `exit=0`. A `STOP:` line (for example a source file without the original rerank template) or another exit code: stop and ask the author. After an interruption, delete the partial `-vimed.gguf` output and send the same command again. Write down `SIZE_4B`, `SHA_4B`, `SIZE_8B`, `SHA_8B` from the logs.

- [ ] **Step 3: Write the failing catalog tests**

Replace the whole of `seed-pipeline/tests/runtime/test_catalog.py` content added in Task 8 (the `VIMED` constant and `test_vimed_reranker_is_the_0_6b_runtime_with_its_own_file`; keep tests Plan A put there) with:

```python
import pytest

from seed_pipeline.runtime.catalog import MODEL_CATALOG, require_model

VIMED_FILES = {
    "qwen3-reranker:0.6b-fp16": (
        "qwen3-reranker-0.6b-f16-vimed.gguf",
        1_197_634_336,
        "fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851",
    ),
    "qwen3-reranker:4b-fp16": (
        "qwen3-reranker-4b-f16-vimed.gguf",
        SIZE_4B,
        "SHA_4B",
    ),
    "qwen3-reranker:8b-fp16": (
        "qwen3-reranker-8b-f16-vimed.gguf",
        SIZE_8B,
        "SHA_8B",
    ),
}


@pytest.mark.parametrize(("model", "expected"), sorted(VIMED_FILES.items()))
def test_qwen3_rerankers_serve_the_vietnamese_medical_template(
    model: str, expected: tuple[str, int, str]
) -> None:
    spec = require_model(model)

    assert (spec.canonical_filename, spec.byte_size, spec.sha256) == expected


def test_the_instruction_experiment_model_is_gone() -> None:
    assert "qwen3-reranker:0.6b-fp16-vimed" not in MODEL_CATALOG
```

Substitute `SIZE_4B`, `SHA_4B`, `SIZE_8B`, `SHA_8B` with the Step 2 values (integers with `_` thousands separators, sha256 as a 64-character string). Keep existing imports Plan A put in the file (merge the import lines).

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_catalog.py`
Expected: FAIL on the three file assertions and on the experiment model still being present.

- [ ] **Step 4: Point the catalog at the new files**

In `seed-pipeline/src/seed_pipeline/runtime/catalog.py`:
- In the `"qwen3-reranker:0.6b-fp16"` entry replace `"qwen3-reranker-0.6b-f16.gguf"`, `1_197_634_304` and `"fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa"` with `"qwen3-reranker-0.6b-f16-vimed.gguf"`, `1_197_634_336` and `"fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851"`.
- In the `"qwen3-reranker:4b-fp16"` entry replace `"qwen3-reranker-4b-f16.gguf"`, `8_049_922_912` and `"c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117"` with `"qwen3-reranker-4b-f16-vimed.gguf"`, `SIZE_4B` and `"SHA_4B"` (Step 2 values).
- In the `"qwen3-reranker:8b-fp16"` entry replace `"qwen3-reranker-8b-f16.gguf"`, `15_141_207_744` and `"a53322f7936010458424a12f0f6d22291547e42fa85c16dd4730244d659cea96"` with `"qwen3-reranker-8b-f16-vimed.gguf"`, `SIZE_8B` and `"SHA_8B"` (Step 2 values).
- Delete the instruction-experiment block added in Task 8 (the comment and `RERANKER_MODELS["qwen3-reranker:0.6b-fp16-vimed"] = replace(...)`), and remove `replace` from the `dataclasses` import if nothing else uses it.

Run: `uv run pytest -q tests/runtime/test_catalog.py`
Expected: PASS.

- [ ] **Step 5: Delete the superseded local files**

```bash
cd /home/andv/personal/thesis
rm ai-models/gguf/qwen3-reranker-0.6b-f16.gguf ai-models/gguf/qwen3-reranker-4b-f16.gguf ai-models/gguf/qwen3-reranker-8b-f16.gguf
cd seed-pipeline
rm -f data/cache/rerank_scores/qwen3_reranker_0_6b_fp16.jsonl \
  data/cache/rerank_scores/qwen3_reranker_0_6b_fp16_vimed.jsonl \
  data/cache/rerank_scores/qwen3_reranker_4b_fp16.jsonl \
  data/cache/rerank_scores/qwen3_reranker_8b_fp16.jsonl \
  data/cache/kaggle_profiles/rerank/qwen3_reranker_0_6b_fp16.json \
  data/cache/kaggle_profiles/rerank/qwen3_reranker_0_6b_fp16_vimed.json \
  data/cache/kaggle_profiles/rerank/qwen3_reranker_4b_fp16.json \
  data/cache/kaggle_profiles/rerank/qwen3_reranker_8b_fp16.json
rm -rf data/work/kaggle-rerank-scores/qwen3_reranker_0_6b_fp16 \
  data/work/kaggle-rerank-scores/qwen3_reranker_0_6b_fp16_vimed \
  data/work/kaggle-rerank-scores/qwen3_reranker_4b_fp16 \
  data/work/kaggle-rerank-scores/qwen3_reranker_8b_fp16
rm -f data/work/vimed/*.chat-templates.json
ls ../ai-models/gguf | grep qwen3-reranker
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --model qwen3-reranker:4b-fp16 --dry-run
```

Expected: `ls` shows only the three `-vimed.gguf` rerankers; the dry run prints `missing_pairs=300000`. The deleted score caches were keyed by the original-template digests (or by the retired experiment name) and would otherwise fail to load with `Model digest mismatch` under the new catalog. The sample run folder stays untouched.

- [ ] **Step 6: Update file names in compose, env, docs and tests**

```bash
cd /home/andv/personal/thesis
git grep -n -E "qwen3-reranker-(0\.6b|4b|8b)-f16\.gguf" -- ':!seed-pipeline/docs/superpowers' ':!seed-pipeline/data'
```

Expected hits include `compose.yaml` (`LLAMA_ARG_MODEL: "/models/${LLAMA_RERANKER_MODEL:-qwen3-reranker-4b-f16.gguf}"`), `.env.example` (`# LLAMA_RERANKER_MODEL=qwen3-reranker-4b-f16.gguf`), the instruction section of `seed-pipeline/docs/guides/evaluation.md`, and Plan A's integration test if it names `qwen3-reranker-0.6b-f16.gguf`. Change them:
- `compose.yaml` and `.env.example`: `qwen3-reranker-4b-f16.gguf` → `qwen3-reranker-4b-f16-vimed.gguf`.
- Integration test: use `require_model("qwen3-reranker:0.6b-fp16").canonical_filename` instead of the literal file name, so the test does not silently skip after the original file was deleted.
- Any other code or guide hit outside the instruction section: the `-vimed` name of the same size.
- `seed-pipeline/docs/guides/evaluation.md`: leave the derivation commands (they start from the original files), replace the sentence `Catalog gọi file này là \`qwen3-reranker:0.6b-fp16-vimed\`.` and extend the table so the file part reads:

```markdown
| File | Kích thước (byte) | sha256 |
| --- | ---: | --- |
| `qwen3-reranker-0.6b-f16-vimed.gguf` | 1.197.634.336 | `fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851` |
| `qwen3-reranker-4b-f16-vimed.gguf` | {SIZE_4B với dấu chấm hàng nghìn} | `{SHA_4B}` |
| `qwen3-reranker-8b-f16-vimed.gguf` | {SIZE_8B với dấu chấm hàng nghìn} | `{SHA_8B}` |

Instruction tiếng Việt thắng (mục Kết quả), nên `qwen3-reranker:0.6b-fp16`, `qwen3-reranker:4b-fp16` và `qwen3-reranker:8b-fp16` trỏ tới ba file trên (lệnh trên với `SIZE=4b` và `SIZE=8b`); model thử nghiệm `qwen3-reranker:0.6b-fp16-vimed` đã bỏ. File gốc đã xoá; khi cần dựng lại, tải từ nguồn sau rồi kiểm sha256 trước khi chạy lệnh:

| File gốc | Nguồn | sha256 |
| --- | --- | --- |
| `qwen3-reranker-0.6b-f16.gguf` | `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp` | `fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa` |
| `qwen3-reranker-4b-f16.gguf` | `Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp` | `c4de2e3e4179d5bca95a2e960e07d225a565018e3bbb5e073f1777809091f117` |
| `qwen3-reranker-8b-f16.gguf` | `sinjab/Qwen3-Reranker-8B-F16-GGUF` | `a53322f7936010458424a12f0f6d22291547e42fa85c16dd4730244d659cea96` |
```

with the braces replaced by the Step 2 values (size written like `8.049.922.912`).

Run: `git grep -n -E "qwen3-reranker-(0\.6b|4b|8b)-f16\.gguf" -- ':!seed-pipeline/docs/superpowers' ':!seed-pipeline/data'`
Expected: only the derivation commands and the source table in `seed-pipeline/docs/guides/evaluation.md`.

- [ ] **Step 7: Run the gates and the reranker integration test**

Run the **Seed gate** in `seed-pipeline/` and the **Backend gate** in `backend/`, then in `seed-pipeline/`: `uv run pytest -q -m integration -k rerank`.
Expected: every command exits 0; the integration test runs (not skipped) against `qwen3-reranker-0.6b-f16-vimed.gguf`.

- [ ] **Step 8: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/runtime/catalog.py \
  seed-pipeline/tests/runtime/test_catalog.py \
  compose.yaml .env.example \
  seed-pipeline/docs/guides/evaluation.md
git add <each other file changed in Step 6, by explicit path>
git diff --cached --name-status
git commit -m "feat(seed): serve the Qwen3 rerankers with the Vietnamese medical instruction" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: Remove the superseded Kaggle datasets (ask first)**

```bash
cd /home/andv/personal/thesis/seed-pipeline
( set -a; . ./.env; set +a
  for n in 1 2 3; do
    user_var="KAGGLE_ACC${n}_USERNAME"; token_var="KAGGLE_ACC${n}_API_TOKEN"
    echo "== acc${n}"
    env -u KAGGLE_KEY KAGGLE_USERNAME="${!user_var}" KAGGLE_API_TOKEN="${!token_var}" \
      uv run kaggle datasets list --mine --search qwen3-reranker
  done )
```

Show the author this deletion proposal and wait for a yes: the model datasets `vector-cache-gguf-qwen3-reranker-0-6b-f16`, `vector-cache-gguf-qwen3-reranker-4b-f16`, `vector-cache-gguf-qwen3-reranker-8b-f16` (original-template files), and every `re-eval-rerank-qwen3-reranker-…-checkpoint` listed (all were produced with the original-template files or the retired experiment model; no future job identity can reuse them). Keep `vector-cache-gguf-qwen3-reranker-0-6b-f16-vimed` (the canonical 0.6b file now) and every non-reranker dataset. After the yes, delete each approved dataset with the credentials of the account that lists it, for example for acc1:

```bash
( set -a; . ./.env; set +a
  env -u KAGGLE_KEY KAGGLE_USERNAME="$KAGGLE_ACC1_USERNAME" KAGGLE_API_TOKEN="$KAGGLE_ACC1_API_TOKEN" \
    uv run kaggle datasets delete -y doanvanan0209/vector-cache-gguf-qwen3-reranker-0-6b-f16 )
```

Expected: the listing no longer shows the deleted datasets.

---

### Task 12: Apply the decision when the original template wins

Run this task only after `DECISION original` in Task 10; otherwise skip it.

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/runtime/catalog.py`, `seed-pipeline/tests/runtime/test_catalog.py`, `seed-pipeline/docs/guides/evaluation.md`
- Delete (ignored): `ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf`, the experiment model's score cache, Kaggle profile and staging folder

**Interfaces:**
- Consumes (Task 8): the experiment entry and test.
- Produces: catalog without `qwen3-reranker:0.6b-fp16-vimed`; canonical files unchanged (`qwen3-reranker-{0.6b,4b,8b}-f16.gguf`).

- [ ] **Step 1: Write the failing catalog test**

In `seed-pipeline/tests/runtime/test_catalog.py`, replace the `VIMED` constant and `test_vimed_reranker_is_the_0_6b_runtime_with_its_own_file` with:

```python
def test_the_instruction_experiment_model_is_gone() -> None:
    assert "qwen3-reranker:0.6b-fp16-vimed" not in MODEL_CATALOG
```

and change the import to `from seed_pipeline.runtime.catalog import MODEL_CATALOG` (merge with imports Plan A put there).

Run (in `seed-pipeline/`): `uv run pytest -q tests/runtime/test_catalog.py`
Expected: FAIL (the experiment model is still in the catalog).

- [ ] **Step 2: Remove the experiment model**

In `seed-pipeline/src/seed_pipeline/runtime/catalog.py`, delete the instruction-experiment block added in Task 8 and remove `replace` from the `dataclasses` import if nothing else uses it.

Run: `uv run pytest -q tests/runtime/test_catalog.py`
Expected: PASS.

- [ ] **Step 3: Delete the experiment's local files**

```bash
cd /home/andv/personal/thesis
rm ai-models/gguf/qwen3-reranker-0.6b-f16-vimed.gguf
cd seed-pipeline
rm -f data/cache/rerank_scores/qwen3_reranker_0_6b_fp16_vimed.jsonl \
  data/cache/kaggle_profiles/rerank/qwen3_reranker_0_6b_fp16_vimed.json
rm -rf data/work/kaggle-rerank-scores/qwen3_reranker_0_6b_fp16_vimed
rm -f data/work/vimed/*.chat-templates.json
ls ../ai-models/gguf | grep qwen3-reranker
```

Expected: only `qwen3-reranker-0.6b-f16.gguf`, `qwen3-reranker-4b-f16.gguf`, `qwen3-reranker-8b-f16.gguf`. The sample run folder stays untouched.

- [ ] **Step 4: Update the guide**

In `seed-pipeline/docs/guides/evaluation.md`, replace `Catalog gọi file này là \`qwen3-reranker:0.6b-fp16-vimed\`.` with:

```markdown
Template gốc thắng (mục Kết quả), nên model thử nghiệm `qwen3-reranker:0.6b-fp16-vimed` và file của nó đã bỏ; lệnh trên dựng lại đúng file có sha256 trong bảng.
```

- [ ] **Step 5: Run the gate and commit**

Run the **Seed gate** in `seed-pipeline/`. Expected: every command exits 0.

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/runtime/catalog.py \
  seed-pipeline/tests/runtime/test_catalog.py \
  seed-pipeline/docs/guides/evaluation.md
git diff --cached --name-status
git commit -m "chore(seed): drop the vimed reranker after the instruction experiment" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Remove the experiment's Kaggle datasets (ask first)**

```bash
cd /home/andv/personal/thesis/seed-pipeline
grep -h -o -E "re-eval-rerank-[a-z0-9-]+-checkpoint" data/work/logs/rerank/qwen3_reranker_0_6b_fp16_vimed.log | sort -u
( set -a; . ./.env; set +a
  env -u KAGGLE_KEY KAGGLE_USERNAME="$KAGGLE_ACC1_USERNAME" KAGGLE_API_TOKEN="$KAGGLE_ACC1_API_TOKEN" \
    uv run kaggle datasets list --mine --search vimed )
```

Show the author this deletion proposal and wait for a yes: the model dataset `doanvanan0209/vector-cache-gguf-qwen3-reranker-0-6b-f16-vimed` and the checkpoint datasets printed by the `grep` (checkpoints of the experiment's sample job; if the log names none, propose the model dataset only). After the yes, delete each with the credentials of its owning account:

```bash
( set -a; . ./.env; set +a
  env -u KAGGLE_KEY KAGGLE_USERNAME="$KAGGLE_ACC1_USERNAME" KAGGLE_API_TOKEN="$KAGGLE_ACC1_API_TOKEN" \
    uv run kaggle datasets delete -y doanvanan0209/vector-cache-gguf-qwen3-reranker-0-6b-f16-vimed )
```

Expected: the listing no longer shows the deleted datasets.

---

### Task 13: Local CPU benchmark for 4b and compose defaults

Spec section 5 lists this after the full scoring (step 6). It runs first here because the CPU measurement needs an idle machine, while the full Kaggle scoring (Task 14) keeps local processes busy merging score caches for days. It only depends on the decided 4b file.

**Files:**
- Modify: `compose.yaml`, `.env.example`
- Create (ignored): `seed-pipeline/data/work/rollout/profile_summary.py`, `seed-pipeline/data/cache/local_profiles/rerank/qwen3_reranker_4b_fp16.json` (written by Plan A's command)

**Interfaces:**
- Consumes (Plan A): `seed rerank --backend local --benchmark --model M --run R` (rebuilds `llama-reranker` per level with `LLAMA_ARG_N_PARALLEL`, `LLAMA_ARG_UBATCH`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_CTX_SIZE`, `LLAMA_ARG_KV_UNIFIED`, `LLAMA_ARG_THREADS`; levels `-np 16`, `-ub` 4096/8192/16384, threads 8/12, 15 documents per request; selects the lowest p95, ties by higher pairs/s; writes a profile with `identity` (`machine_shape`, `runtime`), `selected` and `measurements`, each measurement carrying its `candidate`).
- Produces: `compose.yaml` defaults for the `llama-reranker` service and `PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS` ≥ 2 × measured p95; the CPU row of report table 5.3 (Task 16).

- [ ] **Step 1: Make the machine idle**

```bash
cd /home/andv/personal/thesis
git diff --stat -- compose.yaml .env.example
docker compose ps --format '{{.Service}} {{.State}}'
grep -E "^(LLAMA_RERANKER_|RERANK_)" .env 2>/dev/null
```

Expected: no diff. Write down the running services, then stop every running service except `llama-reranker`, for example `docker compose stop backend frontend nginx llama-embedding cli-proxy-api` for the ones listed as running. If `.env` overrides reranker variables, tell the author that those overrides will hide the new defaults. Then run `uptime` and wait until the 1-minute load average is below 1.0.

- [ ] **Step 2: Run the benchmark**

```bash
tmux new-window -t thesis-rerank -n local-bench-4b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:local-bench-4b 'uv run seed rerank --backend local --benchmark --model qwen3-reranker:4b-fp16 --run hybrid-qwen4b-p50-k30-rrf2 2>&1 | tee -a data/work/logs/rollout/local-bench-4b.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/local-bench-4b.console.log' Enter
```

Expected: one result line per level (pairs/s, p50, p95, status), the chosen configuration, then `exit=0`, and `data/cache/local_profiles/rerank/qwen3_reranker_4b_fp16.json` exists. If every level fails, the command prints the server log tail: stop and ask the author. After an interruption, send the same command again (levels restart from the first).

- [ ] **Step 3: Read the selected level**

Create `seed-pipeline/data/work/rollout/profile_summary.py`:

```python
"""Print a runtime profile's selected level and its measurement. One-off."""

from __future__ import annotations

import json
import sys
from pathlib import Path

profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = profile["selected"]
matches = [
    item for item in profile["measurements"] if item.get("candidate") == selected
]
if len(matches) != 1:
    raise SystemExit(
        f"STOP: expected one measurement of the selected level, found {len(matches)}"
    )
print("identity", json.dumps(profile.get("identity"), sort_keys=True))
print("selected", json.dumps(selected, sort_keys=True))
print("measurement", json.dumps(matches[0], sort_keys=True))
```

Run (in `seed-pipeline/`): `uv run python data/work/rollout/profile_summary.py data/cache/local_profiles/rerank/qwen3_reranker_4b_fp16.json`
Expected: `identity` names the i5-13420H CPU and the compose llama.cpp image tag; `selected` has `server_slots` 16, a `physical_batch_size` of 4096, 8192 or 16384 and a thread count of 8 or 12; `measurement` has status `ok`, pairs per second and p95 per request. If Plan A named these keys differently, use its names for the same values. Write down `NP` (server slots), `UB` (physical batch size), `THREADS`, `P95_SECONDS`, `PAIRS_PER_SECOND`, and `TIMEOUT` = `P95_SECONDS` × 2 rounded up to a whole second.

- [ ] **Step 4: Write the measured defaults**

In `compose.yaml`, service `llama-reranker`: set the `${…:-default}` defaults of the variables feeding `LLAMA_ARG_N_PARALLEL` to `NP`, `LLAMA_ARG_UBATCH`, `LLAMA_ARG_BATCH` and `LLAMA_ARG_CTX_SIZE` to `UB`, and `LLAMA_ARG_THREADS` to `THREADS`; keep `LLAMA_ARG_KV_UNIFIED` true and `LLAMA_ARG_RERANKING` true. Directly above `LLAMA_ARG_N_PARALLEL` put this comment (values filled in):

```yaml
      # Measured on an Intel i5-13420H (12 threads, 23 GB RAM, llama.cpp server-b10920) with
      # `seed rerank --backend local --benchmark --model qwen3-reranker:4b-fp16`:
      # p95 P95_SECONDS s per 15-document request, PAIRS_PER_SECOND pairs/s. Other machines:
      # rerun the benchmark and override these variables in .env.
```

In the `x-backend` environment of `compose.yaml`: `PHARMA_RETRIEVAL__RERANK__MAX_CONCURRENT` must render as `1` (one request per search round; spec 4.6) and must not be tied to the server slot variable; add or update `PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS: "${RERANK_TIMEOUT_SECONDS:-TIMEOUT}"` next to `PHARMA_RETRIEVAL__RERANK__MAX_CANDIDATES`, with the comment `# At least twice the CPU p95 measured above.`

In `.env.example`, set the commented lines of the same wrapper variables to the new defaults and add `# RERANK_TIMEOUT_SECONDS=TIMEOUT` under `# RERANK_MAX_CANDIDATES=15`.

Every `NP`, `UB`, `THREADS`, `P95_SECONDS`, `PAIRS_PER_SECOND`, `TIMEOUT` above is replaced by the Step 3 number.

- [ ] **Step 5: Verify, restart and smoke-test**

```bash
cd /home/andv/personal/thesis
docker compose config llama-reranker backend | grep -E "LLAMA_ARG_(MODEL|N_PARALLEL|UBATCH|BATCH|CTX_SIZE|THREADS|KV_UNIFIED|RERANKING)|PHARMA_RETRIEVAL__RERANK__(MODEL|MAX_CONCURRENT|MAX_CANDIDATES|TIMEOUT_SECONDS)"
docker compose up -d --wait llama-reranker
curl -s http://localhost:11435/v1/rerank -H 'Content-Type: application/json' \
  -d '{"query": "liều paracetamol cho người lớn", "documents": ["Paracetamol: người lớn uống 500 mg đến 1 g mỗi 4 đến 6 giờ, tối đa 4 g mỗi ngày.", "Amoxicillin là kháng sinh nhóm penicilin."]}'
```

Expected: the rendered values equal Step 3 (`LLAMA_ARG_MODEL` ends with the decided 4b file name, `MAX_CONCURRENT` is `1`, `TIMEOUT_SECONDS` is `TIMEOUT`); `curl` (use the port from `LLAMA_RERANKER_PORT` if `.env` changes it) returns two results with `relevance_score` in [0, 1], index 0 higher than index 1. Then start the services stopped in Step 1: `docker compose up -d --wait <those services>`.

- [ ] **Step 6: Run the gates and commit**

Run the **Seed gate** in `seed-pipeline/` and the **Backend gate** in `backend/`. Expected: every command exits 0.

```bash
cd /home/andv/personal/thesis
git add compose.yaml .env.example
git diff --cached --name-status
git commit -m "perf(compose): set reranker defaults from the i5-13420H CPU benchmark" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Full 300,000-pair Kaggle scoring of 0.6b, 4b and 8b

The author approved these three runs; no extra confirmation is needed. They run in parallel and Plan B shares the three accounts between them by quota. 8b may need several quota refreshes (spec 8).

**Files:**
- Produces (tracked, committed in Task 15): `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json`, `…/rerank/qwen3_reranker_{0_6b,4b,8b}_fp16/manifest.json`
- Produces (ignored): `seed-pipeline/data/cache/rerank_scores/qwen3_reranker_{0_6b,4b,8b}_fp16.jsonl`, Kaggle profiles of the decided files

**Interfaces:**
- Consumes (Plan A, Plan B): `seed rerank --backend kaggle --kaggle-account auto`; (Task 5): the run without stale variants; (Task 11 or 12): the decided catalog.
- Produces: complete native variants of the three models on `hybrid-qwen4b-p50-k30-rrf2` (Task 15), Kaggle profiles with the selected pairs/s (Task 16).

- [ ] **Step 1: Count the missing pairs**

```bash
cd /home/andv/personal/thesis/seed-pipeline
for model in qwen3-reranker:0.6b-fp16 qwen3-reranker:4b-fp16 qwen3-reranker:8b-fp16; do
  uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --model "$model" --dry-run
done
```

Expected: 4b and 8b `missing_pairs=300000`; 0.6b `missing_pairs=300000` after Task 11, or at most `205050` after Task 12 (recovered pairs plus identical sample pairs are cached). No container starts.

- [ ] **Step 2: Launch the three jobs**

```bash
tmux new-window -t thesis-rerank -n full-0.6b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:full-0.6b 'uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:0.6b-fp16 --kaggle-account auto 2>&1 | tee -a data/work/logs/rollout/full-0.6b.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/full-0.6b.console.log' Enter
tmux new-window -t thesis-rerank -n full-4b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:full-4b 'uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account auto 2>&1 | tee -a data/work/logs/rollout/full-4b.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/full-4b.console.log' Enter
tmux new-window -t thesis-rerank -n full-8b -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:full-8b 'uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:8b-fp16 --kaggle-account auto 2>&1 | tee -a data/work/logs/rollout/full-8b.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/full-8b.console.log' Enter
```

Expected: each model log (`data/work/logs/rerank/qwen3_reranker_{0_6b,4b,8b}_fp16.log`) shows a different account for concurrent sessions, a benchmark before the first session of a model without a valid profile, then scoring progress.

- [ ] **Step 3: Monitor and resume**

```bash
cd /home/andv/personal/thesis/seed-pipeline
tail -n 2 data/work/logs/rollout/full-0.6b.console.log data/work/logs/rollout/full-4b.console.log data/work/logs/rollout/full-8b.console.log
tail -n 20 data/work/logs/rerank/qwen3_reranker_8b_fp16.log
```

- `exit=3` with a quota table: every account has less than 1 hour of GPU quota. Wait until the earliest printed `refreshAt` (the first one after 2026-09-15 is 2026-09-19) and send the same command again in that window. The local cache and checkpoints keep every scored pair.
- `exit=130`, a closed terminal or a WSL restart: follow **Resume after an interruption**.
- `exit=1`: show the last 50 lines of the model log to the author and stop that model (the others keep running).
- `exit=0`: go to Task 15 for that model, without waiting for the others.

- [ ] **Step 4: Verify a finished model**

For the model whose window printed `exit=0`, set `SLUG` to `qwen3_reranker_0_6b_fp16`, `qwen3_reranker_4b_fp16` or `qwen3_reranker_8b_fp16` and `MODEL` to the matching `qwen3-reranker:0.6b-fp16`, `qwen3-reranker:4b-fp16` or `qwen3-reranker:8b-fp16`:

```bash
cd /home/andv/personal/thesis/seed-pipeline
SLUG=qwen3_reranker_4b_fp16 MODEL=qwen3-reranker:4b-fp16 uv run python - <<'EOF'
import json
import os
from pathlib import Path

from seed_pipeline.runtime.catalog import require_model

slug, model = os.environ["SLUG"], os.environ["MODEL"]
manifest = json.loads(
    Path(f"data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/{slug}/manifest.json").read_text(encoding="utf-8")
)
print(manifest["complete"], manifest["total"], manifest["identity"]["model_sha256"] == require_model(model).sha256)
EOF
```

Expected: `300000 300000 True`.

---

### Task 15: Metrics and score-distribution check per model

Run this task once per model, as soon as that model's Task 14 window ends with `exit=0`. Set the pair for the model at hand in every shell block: `SLUG=qwen3_reranker_0_6b_fp16 MODEL=qwen3-reranker:0.6b-fp16`, `SLUG=qwen3_reranker_4b_fp16 MODEL=qwen3-reranker:4b-fp16`, or `SLUG=qwen3_reranker_8b_fp16 MODEL=qwen3-reranker:8b-fp16`.

**Files:**
- Replace (tracked): `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/<slug>/top30-window3/{manifest.json,report.md}`
- Commit: `seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/run.json`, `…/rerank/<slug>/manifest.json`
- Modify: `seed-pipeline/docs/guides/evaluation.md` (results table)
- Create (ignored, first model only): `seed-pipeline/data/work/rollout/report_rows.py`

**Interfaces:**
- Consumes (Task 14): a complete native variant; (Task 10): `score_distribution.py`.
- Produces: the model's report, its table 5.1 cells and table 5.2 Hit@10 cells (Task 16).

- [ ] **Step 1: Replace the completion-era report**

```bash
cd /home/andv/personal/thesis/seed-pipeline
export SLUG=qwen3_reranker_4b_fp16 MODEL=qwen3-reranker:4b-fp16
R=data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2
rm -rf $R/reports/rerank/$SLUG/top30-window3
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --model $MODEL --top-k 30
git -C /home/andv/personal/thesis status --short -- seed-pipeline/$R/reports/baseline
```

Expected: `status=complete` with one `reranked` entry for `$MODEL`; the baseline status is empty (the existing baseline report is reused, not rewritten). Deleting the old folder first avoids `--force`, which would also republish the baseline.

- [ ] **Step 2: Check the score distribution**

```bash
uv run python data/work/rollout/score_distribution.py $R/rerank/$SLUG/rerank_scores.jsonl
```

Expected: `pairs=300000 queries=10000 …` and `distribution=PASS` (minimum ≤ 0.05, maximum ≥ 0.95, all scores in [0, 1], median per-query spread > 0.9). On `distribution=FAIL`, stop and ask the author; do not commit the report or copy its numbers.

- [ ] **Step 3: Format the report cells**

For the first finished model, create `seed-pipeline/data/work/rollout/report_rows.py`:

```python
"""Format one reranker's rrf2 metrics as cells of report tables 5.1 and 5.2. One-off."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path("data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2")
GOLD = Path("data/evaluation/gold/section_retrieval_eval.jsonl")
GROUP_LABELS = (
    ("formulary", "formulary"),
    ("leaflet", "brand_product_qa"),
    ("chunk_risk", "chunk_level_retrieval"),
    ("patient_natural", "patient_natural"),
    ("noisy_confuser", "noisy_confuser"),
    ("multi_intent", "multi_intent"),
)


def percent(values: list[float]) -> str:
    return f"{sum(values) / len(values) * 100:.2f}%".replace(".", ",")


def main(slug: str) -> None:
    group_of: dict[str, str] = {}
    with GOLD.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                group_of[row["query_id"]] = row["eval_group"]
    values: defaultdict[str, list[float]] = defaultdict(list)
    hit10_by_group: defaultdict[str, list[float]] = defaultdict(list)
    path = RUN / "reports" / "rerank" / slug / "top30-window3" / "metrics.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                for key in ("hit@3", "hit@5", "hit@10", "hit@30", "mrr"):
                    values[key].append(float(row[key]))
                if "multi_all_hit@10" in row:
                    values["multi_all_hit@10"].append(float(row["multi_all_hit@10"]))
                hit10_by_group[group_of[row["query_id"]]].append(float(row["hit@10"]))
    mrr = f"{sum(values['mrr']) / len(values['mrr']):.4f}".replace(".", ",")
    cells = [percent(values[key]) for key in ("hit@3", "hit@5", "hit@10", "hit@30")]
    cells += [mrr, percent(values["multi_all_hit@10"])]
    print(f"queries={len(values['mrr'])}")
    print("table 5.1 cells: | " + " | ".join(cells) + " |")
    print(
        "table 5.2 Hit@10: "
        + ", ".join(
            f"{label}={percent(hit10_by_group[group])}" for group, label in GROUP_LABELS
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
```

On the completion-era 4b report of 2026-09-15 it printed `| 88,51% | 92,97% | 96,98% | 99,09% | 0,8060 | 77,20% |` and `formulary=97,38%, brand_product_qa=99,04%, chunk_level_retrieval=96,20%, patient_natural=100,00%, noisy_confuser=81,40%, multi_intent=96,80%`, matching tables 5.1 and 5.2 of the report.

```bash
uv run python data/work/rollout/report_rows.py $SLUG
grep -E "^\| (Hit@10|MRR) \|" $R/reports/rerank/$SLUG/top30-window3/report.md
```

Expected: `queries=10000`; the Hit@10 and MRR cells equal the two `report.md` rows (with a decimal comma). Keep both printed lines for Task 16.

- [ ] **Step 4: Update the guide's results table**

In the results table at the end of `seed-pipeline/docs/guides/evaluation.md` (section `Kết quả tham chiếu`), replace the Hit@10 and MRR of the row `| rrf2 + \`$MODEL\` | … | … |` with the Step 3 values. The rows before this rollout read `| rrf2 + \`qwen3-reranker:0.6b-fp16\` | 96,18% | 0,7823 |`, `| rrf2 + \`qwen3-reranker:4b-fp16\` | 96,98% | 0,8060 |` and `| rrf2 + \`qwen3-reranker:8b-fp16\` | 89,75% | 0,4797 |`.

- [ ] **Step 5: Commit**

```bash
cd /home/andv/personal/thesis
R=seed-pipeline/data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2
git add $R/run.json $R/rerank/qwen3_reranker_*_fp16/manifest.json \
  $R/reports/rerank/$SLUG/top30-window3/manifest.json \
  $R/reports/rerank/$SLUG/top30-window3/report.md \
  seed-pipeline/docs/guides/evaluation.md
git diff --cached --name-status
git commit -m "chore(seed-data): add native $MODEL scores to the rrf2 run" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Expected staged paths: `run.json`, the rerank manifests of every Qwen3 variant registered so far, the two report files of `$SLUG`, and the guide.

---

### Task 16: Update the progress report

Run after Task 15 has been done for all three models and after Task 13.

**Files:**
- Modify: `report/report.md` (tracked)

**Interfaces:**
- Consumes: Task 15 Step 3 output for 0.6b, 4b and 8b; `profile_summary.py` (Task 13); Task 13 Step 3 values; Task 10 Step 3 values; Task 15 Step 2 output for 8b.
- Produces: tables 5.1–5.3 and remarks 5.4 with native numbers.

- [ ] **Step 1: Collect the Kaggle speed numbers**

```bash
cd /home/andv/personal/thesis/seed-pipeline
for slug in qwen3_reranker_0_6b_fp16 qwen3_reranker_4b_fp16 qwen3_reranker_8b_fp16; do
  echo "== $slug"
  uv run python data/work/rollout/profile_summary.py data/cache/kaggle_profiles/rerank/$slug.json
done
```

Expected: for each model a `selected` level (`server_slots`, `physical_batch_size`, `request_batch_size` 30) and its `measurement` with status `ok` and pairs per second.

- [ ] **Step 2: Table 5.1**

In `report/report.md`, replace the cells of the rows `Hybrid k=2 + \`qwen3-reranker:0.6b\``, `Hybrid k=2 + \`qwen3-reranker:4b\`` and `Hybrid k=2 + \`qwen3-reranker:8b\`` with each model's `table 5.1 cells` line. In the Hit@3, Hit@5, Hit@10, MRR and Multi-all-hit@10 columns, bold exactly the largest value of the column (move the bold if a new value is larger, bold both on a tie); leave the Hit@30 column as it is. Make the row label bold only for the reranker with the highest MRR, and remove the bold from the other labels.

- [ ] **Step 3: Table 5.2**

Rename the last column header to `Hybrid k=2 + qwen3-reranker-<size>` for the reranker with the highest MRR in table 5.1 (`0.6b`, `4b` or `8b`) and replace the six cells of that column with its `table 5.2 Hit@10` values, in the order `formulary`, `brand_product_qa`, `chunk_level_retrieval`, `patient_natural`, `noisy_confuser`, `multi_intent`.

- [ ] **Step 4: Table 5.3**

Replace the whole subsection from `### 5.3 Tốc độ reranker (Kaggle T4, mẫu 512 cặp)` up to (not including) `### 5.4 Nhận xét` with the text below, filling each brace from Step 1 (Kaggle) and Task 13 Step 3 (CPU), with decimal commas and pairs per second rounded to one decimal:

```markdown
### 5.3 Tốc độ reranker (`/v1/rerank`, gom khối)

Kaggle, llama.cpp `b9637`, 30 tài liệu mỗi request; cấu hình do benchmark đầu-cuối chọn (số cặp/giây cao nhất):

| Reranker | GPU | `-np` | `-ub` | Cặp / giây |
| --- | --- | ---: | ---: | ---: |
| `qwen3-reranker:0.6b` | 2 × T4, 2 server | {NP_06B} | {UB_06B} | {PAIRS_06B} |
| `qwen3-reranker:4b` | 2 × T4, 2 server | {NP_4B} | {UB_4B} | {PAIRS_4B} |
| `qwen3-reranker:8b` | 2 × T4, 1 server chia 2 GPU | {NP_8B} | {UB_8B} | {PAIRS_8B} |

CPU production (Intel i5-13420H, llama.cpp `b10920`, `qwen3-reranker:4b`, 15 tài liệu mỗi request): `-np` {NP_CPU}, `-ub` {UB_CPU}, {THREADS_CPU} luồng, {PAIRS_CPU} cặp/giây, p95 {P95_CPU} giây mỗi request. Số Kaggle và CPU đo trên hai phiên bản llama.cpp khác nhau nên không so với nhau.

`bge-reranker-v2-m3` (31,2 cặp/giây) và `bge-reranker-v2-gemma` (8,2 cặp/giây) giữ số đo cũ trên mẫu 512 cặp, không gom khối, nên không so trực tiếp với bảng trên.
```

- [ ] **Step 5: Remarks 5.4**

In `### 5.4 Nhận xét`:
- Bullet `**Reranker sửa thứ hạng:**`: rewrite as `**Reranker sửa thứ hạng:** MRR 0,7242 → {MRR của reranker tốt nhất} với \`qwen3-reranker:<size>\`, Hit@3 {Hit@3 của nó}.` followed by `Đây là cấu hình production.` when the best reranker is 4b, otherwise by `Production dùng \`qwen3-reranker:4b\` (p95 CPU {P95_CPU} giây mỗi request).`
- Bullet `**Đánh đổi:**`: replace `87,00% → 77,20%` with `87,00% → {Multi-all-hit@10 của reranker tốt nhất}`; delete the bullet if that value is 87,00% or higher.
- Replace the bullet `**Kết quả 8b không hợp lệ:** …` (all its lines) with: `**Chấm native:** ba model \`qwen3-reranker\` được chấm lại qua \`/v1/rerank\` bằng classifier head (\`cls.output.weight\`), đúng phép softmax trên logit "yes"/"no" của model card. Điểm 8b trải từ {min} tới {max}, trung vị chênh lệch điểm trong một câu hỏi {median_query_spread}. Kết quả 8b cũ (MRR 0,4797, điểm nén trong 0,31–0,91) là lỗi của cách chấm \`completion_logprobs\`.` with the braces from Task 15 Step 2 output for 8b (4 decimals, decimal comma).
- Add at the end: `**Instruction:** trên tập con 1.000 câu phân tầng, instruction tiếng Việt y tế đổi MRR@30 của \`qwen3-reranker:0.6b\` {DIFFERENCE} (khoảng tin cậy 95% [{CI_LOW}; {CI_HIGH}]), nên {DECISION_TEXT}.` with the Task 10 values and `{DECISION_TEXT}` = `cả ba model dùng instruction này` or `giữ template gốc`.

- [ ] **Step 6: Update other mentions and check**

```bash
cd /home/andv/personal/thesis
grep -n -E "0,8060|96,98%|0,7823|96,18%|0,4797|89,75%|88,51%|18,5|chưa đo|completion_logprobs|logprobs" report/report.md
grep -n "{" report/report.md
git status --short report
```

Expected: the first grep only shows lines where the old numbers are quoted on purpose (the `**Chấm native:**` bullet); update any other hit, for example in `## 1. Tóm tắt`, to the new numbers. The brace grep prints nothing. `git status` shows ` M report/report.md`.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add report/report.md
git diff --cached --name-status
git commit -m "docs(report): update reranker results with native batched scoring" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Expected: `git diff --cached --name-status` lists only `M report/report.md`; pre-commit passes.

---

### Task 17: Archive the new data (ask first) and clean up

**Files:** none tracked.

**Interfaces:**
- Consumes: `seed data push --kaggle-account acc1 --message TEXT` (archive of every Git-ignored file under `data/` except `data/work/`).
- Produces: a new version of the private dataset `doanvanan0209/seed-pipeline-data` containing the native score caches, run score files and the sample run's `metrics.jsonl`, so `seed data pull` on a fresh clone reproduces the guide.

- [ ] **Step 1: Ask the author**

Ask: "Push the new score caches, rerank artifacts and the sample run to the private Kaggle archive `doanvanan0209/seed-pipeline-data` now (`seed data push --kaggle-account acc1`)? It uploads the changed archive parts." Stop until the author says yes. If the author says no, skip to Step 3.

- [ ] **Step 2: Push**

```bash
tmux new-window -t thesis-rerank -n data-push -c /home/andv/personal/thesis/seed-pipeline
tmux send-keys -t thesis-rerank:data-push 'uv run seed data push --kaggle-account acc1 --message "Native Qwen3 rerank scores and instruction experiment" 2>&1 | tee -a data/work/logs/rollout/data-push.console.log; echo "exit=${PIPESTATUS[0]}" | tee -a data/work/logs/rollout/data-push.console.log' Enter
```

Expected: `status=complete` and `exit=0`. After an interruption, send the same command again.

- [ ] **Step 3: Clean up**

```bash
cd /home/andv/personal/thesis/seed-pipeline
tail -qn 1 data/work/logs/rollout/*.console.log
rm -rf data/work/rollout data/work/vimed
tmux kill-session -t thesis-rerank
git -C /home/andv/personal/thesis status --short
```

Expected: every console log ends with `exit=0` (check before deleting anything; a job without `exit=0` means stop here). The one-off scripts are deleted, `data/work/logs/` is kept, and `git status` shows no file from this plan left unstaged apart from `?? report/`.
