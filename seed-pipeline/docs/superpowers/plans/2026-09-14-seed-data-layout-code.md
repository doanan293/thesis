# Seed Data Layout: Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make seed-pipeline read and write the five-folder data layout with readable, hash-free names, delete backward-compatibility code, name leaflet data after its content across seed-pipeline and backend, and let `pharma-agent corpus gc` delete orphaned documents and sections.

**Architecture:** Expand, migrate, contract on `seed_pipeline.config.paths`. Task 1 moves evaluation, cache, corpus and work constants to the new folders while build-input constants keep pointing at the pre-migration folders; Tasks 8–9 move the build inputs and delete the old constants. Every other task changes one subsystem end to end (module, CLI, tests) so both test suites are green after each commit. This plan changes code only: moving the data is plan `2026-09-14-seed-data-migration.md`, so `pytest -m data` is expected to fail between the two plans.

**Tech Stack:** Python 3.12, uv workspace (root `pyproject.toml`), Typer, pytest (`asyncio_mode = "auto"`, warnings are errors), ruff, pyrefly, SQLAlchemy + Postgres, Qdrant, testcontainers (backend integration tests).

**Spec:** `seed-pipeline/docs/superpowers/specs/2026-09-14-seed-data-layout-design.md` (sections §4–§6 and §12 are what this plan implements).

## Global Constraints

- No artifact path that seed-pipeline code creates may contain 12 or more consecutive lowercase hexadecimal characters (spec §5). Hashes, digests and identities live inside `manifest.json` or inside records.
- After Task 11, no tracked file under `backend/src`, `backend/skills`, `seed-pipeline/src` or `frontend/app` contains `ankhang` or `an khang` (case-insensitive) (spec §2.5, §12).
- Leaflet identifiers: document and section key `leaflet:<category>:<slug>`; bundle document source `{"title": "Tờ hướng dẫn sử dụng", "url": null}`; collection title `Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng thuốc`; gold `query_id` prefixes `leaflet-`, `leaflet-alias-`, `leaflet-alias-any-`; `eval_group` `leaflet`; `source_family` `leaflet_brand`; `source_subcategory` and tags `leaflet_<aspect>`; tag `leaflet` (spec §5).
- Kaggle worker modules must not import `seed_pipeline.config.paths` or `pharma_agent` (`seed-pipeline/tests/integrations/kaggle/test_worker_bundle_imports.py`). `runtime/runtime_profiles.py` and `cache/jsonl_records.py` stay free of `config.paths`.
- Before Task 1, let running Kaggle jobs finish: their resume state under `data/heavy/.work/` is not read after `WORK_DIR` moves to `data/work/`.
- Test folders have no `__init__.py` (pytest runs with `--import-mode=importlib`); do not add one.
- Run `uv run ruff format src tests` before each gate; the snippets below are close to, but not guaranteed to be, formatter output.
- `CHUNKER_VERSION` stays `chunker-v1`; bundle schema stays `knowledge-bundle/v1`.
- No `noqa`, `type: ignore`, `pyrefly: ignore`, rule ignores or relaxed warnings filters. Fix findings in code.
- Tests never write into the real `seed-pipeline/data/`: locks go to a per-test temp dir (Task 2 root conftest) and tests that reach the default cache paths patch them.
- `seed-pipeline/.gitignore` keeps the `data/heavy/` rule; the data migration plan removes it after the old data is deleted.
- `seed-pipeline/docs/guides/migration-2026-09.md`, `seed bundle parity`, `bundle/parity.py`, `tests/bundle/test_parity*.py` and `MIGRATION_DIR` stay until the data migration plan finishes (it still needs them).
- **Seed gate** (run in `seed-pipeline/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
- **Backend gate** (run in `backend/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`, plus `uv run pytest -q -m integration` in tasks that touch Postgres or Qdrant code.
- **Commit** from the repository root after each task: `git add` only the task's files, review `git diff --cached --name-status`, then `git commit` with a conventional message that ends with these two lines (pre-commit runs on commit; fix and re-stage on failure, never `--no-verify`):
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N4DN578zgtYyAbaKdYXRX4
  ```

## File Structure

| Area | Files | Responsibility after this plan |
| --- | --- | --- |
| Paths | `seed-pipeline/src/seed_pipeline/config/paths.py`, `seed-pipeline/.gitignore` | One place that names every data folder; `run_dir(run)` |
| Locks | `seed-pipeline/src/seed_pipeline/integrations/kaggle/job_lock.py`, `seed-pipeline/tests/conftest.py` (new) | Readable `<target>.<job|cache>.lock` files under `data/work/locks` |
| Runs | `evaluation/run_workspace.py`, `evaluation/backend_retrieval.py`, `cli/commands/retrieve.py` | `run.json` schema 3, one tree per run, conflicts and `--force` |
| Rerank | `evaluation/rerank_artifacts.py`, `evaluation/rerank_service.py`, `evaluation/rerank_score_cache.py`, `cli/commands/rerank.py` | `rerank/<model>/`, `cache/rerank_scores/<model>.jsonl`, dry-run pair count |
| Metrics | `evaluation/metrics_artifacts.py`, `evaluation/metrics_service.py`, `cli/commands/metrics.py` | `reports/baseline/top<K>-window<N>/`, `reports/rerank/<model>/top<K>-window<N>/` |
| Caches | `cache/jsonl_records.py`, `evaluation/query_embedding_{cache,artifact,service}.py`, `integrations/kaggle/workers/query_embed.py`, `evaluation/preload_query_embeddings.py` | Sealed records only |
| Kaggle | `runtime/runtime_profiles.py`, `integrations/kaggle/{auto_profile,kernels}.py` | `cache/kaggle_profiles/<workload>/<model>.json`, one kernel reference |
| Build | `artifacts/paths.py`, `artifacts/contract.py`, `orchestration/{build_corpus,preflight}.py`, `cli/commands/{build,source}.py` | `work/build/in-progress`, leaflet HTML read from `sources/leaflets` |
| Leaflet source | `corpus/sources/leaflet_source.py` (new), `corpus/sources/crawl.py` | Crawl to `html/<category>/<slug>.html`, `urls/`, `manifest.json`; verify before build |
| Neutral names | `corpus/crawling/integrate_leaflets.py` (renamed), `corpus/crawling/parse_html.py`, `corpus/validation/validate_final_rag.py`, `bundle/export.py`, `evaluation/{section_eval_schema,build_section_retrieval_eval,patient_query_generation,metrics_service}.py` | `leaflet` identifiers and text |
| Backend | `backend/src/pharma_agent/domain/agent/prompts.py`, `backend/skills/brand-to-generic/SKILL.md`, `backend/src/pharma_agent/domain/corpus/{bundle,models}.py`, `backend/src/pharma_agent/infrastructure/persistence/postgres/corpus_repository.py`, `backend/src/pharma_agent/cli.py`, backend test fixtures | Neutral prompt and fixtures; gc deletes orphan identities |
| Policy | `seed-pipeline/tests/test_repository_data_policy.py`, `seed-pipeline/tests/test_removed_contracts.py`, READMEs and guides | Guards and documentation |

Deleted: `evaluation/rejudge_service.py`, `evaluation/rejudging.py`, `evaluation/dump_retrieval_candidates.py`, `artifacts/snapshot.py`, `corpus/crawling/collect_urls.py`, `corpus/crawling/download_html.py`, `corpus/crawling/integrate_ankhang.py` (renamed), and their tests.

---

### Task 1: Evaluation, cache, corpus and work paths

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py`
- Modify (rename constants): every file listed by the grep in Step 5
- Delete: `seed-pipeline/src/seed_pipeline/evaluation/dump_retrieval_candidates.py` (move `load_query_rows` into `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py`)
- Modify: `seed-pipeline/.gitignore`
- Test: `seed-pipeline/tests/config/test_paths.py`, `seed-pipeline/tests/test_repository_data_policy.py`, `seed-pipeline/tests/test_removed_contracts.py`, `seed-pipeline/tests/cli/test_retrieve_command.py`, `seed-pipeline/tests/cli/test_metrics_command.py`, `seed-pipeline/tests/cli/test_rerank_command.py`, `seed-pipeline/tests/evaluation/test_metrics_artifacts.py`, `seed-pipeline/tests/bundle/test_parity_data.py`

**Interfaces:**
- Produces (`seed_pipeline.config.paths`): `CORPUS_DIR`, `RAG_FINAL_DIR`, `RAG_FINAL_SECTIONS_PATH`, `BUNDLE_DIR`, `EVALUATION_DIR`, `GOLD_DIR`, `RUNS_DIR`, `run_dir(run: str) -> Path`, `CACHE_DIR`, `TEXT_EMBEDDING_CACHE_DIR`, `QUERY_EMBEDDING_CACHE_DIR`, `RERANK_SCORE_CACHE_DIR`, `KAGGLE_PROFILE_DIR`, `WORK_DIR`, `BUILD_WORK_DIR`, `LOCK_DIR`, `TEXT_INTERIM_DIR`, `RAG_INTERIM_DIR`, `DOCLING_INTERIM_DIR`, `CANONICAL_INTERIM_DIR`, `ANKHANG_MARKDOWN_INTERIM_DIR`, `BUNDLE_EMBED_WORK_DIR`, `EVALUATION_CHUNKS_PATH`; temporary `retrieval_run_roots(run) -> tuple[Path, Path]` returning `(run_dir(run), run_dir(run))` (deleted in Task 4).
- Produces (`seed_pipeline.evaluation.backend_retrieval`): `load_query_rows(path, limit)` (moved, same signature and body).
- Removed: `PROCESSED_DIR`, `PROCESSED_EVALUATION_DIR`, `DATA_CACHE_DIR`, `RUNTIME_PROFILE_DIR`, `BUNDLES_DIR`, `DEFAULT_BUNDLE_DIR`, `INTERIM_DIR`, `RETRIEVAL_EVAL_DIR`, `HEAVY_RETRIEVAL_EVAL_DIR`, `DATA_RUNS_DIR`, `RETRIEVAL_EVAL_RUNS_DIR`, `RAG_FINAL_MANIFEST_PATH`, `RAG_FINAL_VALIDATION_PATH`.
- Unchanged for now: `HEAVY_DATA_DIR`, `RESOURCES_*`, `MANIFESTS_DIR`, `HEAVY_RAW_DIR`, `RAW_*`, `MIGRATION_DIR`, `query_embedding_bundle_dir`, `query_embedding_cache_path`, `text_embedding_cache_path`, `rerank_score_cache_path`, `COMPOSE_FILE`, `GGUF_ROOT`, `BACKEND_ENV_FILE`.

- [ ] **Step 1: Write the failing path test**

Replace the whole of `seed-pipeline/tests/config/test_paths.py` with:

```python
from seed_pipeline.config import paths


def test_evaluation_cache_corpus_and_work_folders_follow_the_layout() -> None:
    assert paths.RAG_FINAL_DIR == paths.DATA_DIR / "corpus" / "rag-final"
    assert paths.RAG_FINAL_SECTIONS_PATH == paths.RAG_FINAL_DIR / "sections.jsonl"
    assert paths.BUNDLE_DIR == paths.DATA_DIR / "corpus" / "formulary"
    assert paths.GOLD_DIR == paths.DATA_DIR / "evaluation" / "gold"
    assert paths.RUNS_DIR == paths.DATA_DIR / "evaluation" / "runs"
    assert paths.run_dir("trial") == paths.RUNS_DIR / "trial"
    assert paths.TEXT_EMBEDDING_CACHE_DIR == paths.CACHE_DIR / "text_embeddings"
    assert paths.QUERY_EMBEDDING_CACHE_DIR == paths.CACHE_DIR / "query_embeddings"
    assert paths.RERANK_SCORE_CACHE_DIR == paths.CACHE_DIR / "rerank_scores"
    assert paths.KAGGLE_PROFILE_DIR == paths.CACHE_DIR / "kaggle_profiles"
    assert paths.CACHE_DIR == paths.DATA_DIR / "cache"
    assert paths.WORK_DIR == paths.DATA_DIR / "work"
    assert paths.BUILD_WORK_DIR == paths.WORK_DIR / "build"
    assert paths.LOCK_DIR == paths.WORK_DIR / "locks"
    assert paths.TEXT_INTERIM_DIR == paths.BUILD_WORK_DIR / "text"
    assert paths.BUNDLE_EMBED_WORK_DIR == paths.WORK_DIR / "bundle-embed"
    assert paths.EVALUATION_CHUNKS_PATH == (
        paths.WORK_DIR / "evaluation-chunks" / "chunks.jsonl"
    )
```

Add to `seed-pipeline/tests/test_removed_contracts.py`:

```python
REMOVED_PATH_NAMES = (
    "PROCESSED_DIR",
    "PROCESSED_EVALUATION_DIR",
    "DATA_CACHE_DIR",
    "RUNTIME_PROFILE_DIR",
    "BUNDLES_DIR",
    "DEFAULT_BUNDLE_DIR",
    "INTERIM_DIR",
    "RETRIEVAL_EVAL_DIR",
    "HEAVY_RETRIEVAL_EVAL_DIR",
    "DATA_RUNS_DIR",
    "RETRIEVAL_EVAL_RUNS_DIR",
    "RAG_FINAL_MANIFEST_PATH",
    "RAG_FINAL_VALIDATION_PATH",
)


@pytest.mark.parametrize("name", REMOVED_PATH_NAMES)
def test_pre_layout_path_names_are_gone(name: str) -> None:
    assert not hasattr(paths, name)
```

and append `"seed_pipeline.evaluation.dump_retrieval_candidates",` to `REMOVED_MODULES`.

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/config/test_paths.py tests/test_removed_contracts.py`
Expected: FAIL with an `AssertionError` on `RAG_FINAL_DIR` and the `hasattr` assertions failing.

- [ ] **Step 3: Rewrite the constants in `paths.py`**

Replace everything from `PROJECT_ROOT = resolve_workspace_root()` (line 37) to the end of `def retrieval_run_roots` (line 91) with:

```python
PROJECT_ROOT = resolve_workspace_root()
DATA_DIR = PROJECT_ROOT / "data"

# Build inputs stay in the pre-migration folders until the leaflet source tasks move them.
HEAVY_DATA_DIR = DATA_DIR / "heavy"
RESOURCES_DIR = DATA_DIR / "resources"
RESOURCES_ANKHANG_DIR = RESOURCES_DIR / "ankhang"
RESOURCES_CURATION_DIR = RESOURCES_DIR / "curation"
MANIFESTS_DIR = DATA_DIR / "manifests"
HEAVY_RAW_DIR = HEAVY_DATA_DIR / "raw"
RAW_DIR = HEAVY_RAW_DIR
RAW_ANKHANG_DIR = RAW_DIR / "ankhang"
RAW_ANKHANG_HTML_DIR = RAW_ANKHANG_DIR / "html"
RAW_CURATION_DIR = RESOURCES_CURATION_DIR
RAW_ANKHANG_SNAPSHOTS_DIR = RAW_ANKHANG_DIR / "snapshots"
MIGRATION_DIR = HEAVY_DATA_DIR / "migration"

CORPUS_DIR = DATA_DIR / "corpus"
RAG_FINAL_DIR = CORPUS_DIR / "rag-final"
RAG_FINAL_SECTIONS_PATH = RAG_FINAL_DIR / "sections.jsonl"
BUNDLE_DIR = CORPUS_DIR / "formulary"

EVALUATION_DIR = DATA_DIR / "evaluation"
GOLD_DIR = EVALUATION_DIR / "gold"
RUNS_DIR = EVALUATION_DIR / "runs"

CACHE_DIR = DATA_DIR / "cache"
TEXT_EMBEDDING_CACHE_DIR = CACHE_DIR / "text_embeddings"
QUERY_EMBEDDING_CACHE_DIR = CACHE_DIR / "query_embeddings"
RERANK_SCORE_CACHE_DIR = CACHE_DIR / "rerank_scores"
KAGGLE_PROFILE_DIR = CACHE_DIR / "kaggle_profiles"

WORK_DIR = DATA_DIR / "work"
BUILD_WORK_DIR = WORK_DIR / "build"
LOCK_DIR = WORK_DIR / "locks"
TEXT_INTERIM_DIR = BUILD_WORK_DIR / "text"
RAG_INTERIM_DIR = BUILD_WORK_DIR / "rag"
DOCLING_INTERIM_DIR = BUILD_WORK_DIR / "docling"
CANONICAL_INTERIM_DIR = BUILD_WORK_DIR / "canonical"
ANKHANG_MARKDOWN_INTERIM_DIR = BUILD_WORK_DIR / "ankhang_markdown"
BUNDLE_EMBED_WORK_DIR = WORK_DIR / "bundle-embed"
EVALUATION_CHUNKS_PATH = WORK_DIR / "evaluation-chunks" / "chunks.jsonl"

COMPOSE_FILE = PROJECT_ROOT.parent / "compose.yaml"
GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"
BACKEND_ENV_FILE = PROJECT_ROOT.parent / "backend" / ".env"


def run_dir(run: str) -> Path:
    return RUNS_DIR / run


def retrieval_run_roots(run: str) -> tuple[Path, Path]:
    """Both roots are the single run tree; Task 4 removes this helper."""
    return run_dir(run), run_dir(run)
```

- [ ] **Step 4: Move `load_query_rows` and delete the dead dump module**

Copy the function `load_query_rows` (lines 124–153 of `evaluation/dump_retrieval_candidates.py`) verbatim into `evaluation/backend_retrieval.py` above `class BackendCandidateRetriever`, add `import json` (the only import it lacks there), delete the line `from seed_pipeline.evaluation.dump_retrieval_candidates import load_query_rows`, then `git rm seed-pipeline/src/seed_pipeline/evaluation/dump_retrieval_candidates.py`.

- [ ] **Step 5: Rename the removed constants in their users**

Run (in `seed-pipeline/`):

```bash
grep -rlwE 'PROCESSED_EVALUATION_DIR|DEFAULT_BUNDLE_DIR|RUNTIME_PROFILE_DIR|DATA_CACHE_DIR|INTERIM_DIR|PROCESSED_DIR|BUNDLES_DIR|RETRIEVAL_EVAL_DIR|HEAVY_RETRIEVAL_EVAL_DIR|DATA_RUNS_DIR|RETRIEVAL_EVAL_RUNS_DIR|RAG_FINAL_MANIFEST_PATH|RAG_FINAL_VALIDATION_PATH' src tests \
  | grep -v -e 'config/paths.py' -e 'tests/test_removed_contracts.py' | xargs sed -i -E \
  -e 's/\bPROCESSED_EVALUATION_DIR\b/GOLD_DIR/g' \
  -e 's/\bDEFAULT_BUNDLE_DIR\b/BUNDLE_DIR/g' \
  -e 's/\bRUNTIME_PROFILE_DIR\b/KAGGLE_PROFILE_DIR/g'
```

Then fix by hand the remaining references the command prints with `grep -rnwE 'DATA_CACHE_DIR|INTERIM_DIR|PROCESSED_DIR|BUNDLES_DIR|RETRIEVAL_EVAL_DIR|HEAVY_RETRIEVAL_EVAL_DIR|DATA_RUNS_DIR|RETRIEVAL_EVAL_RUNS_DIR|RAG_FINAL_MANIFEST_PATH|RAG_FINAL_VALIDATION_PATH' src tests | grep -v test_removed_contracts.py`:
- `tests/cli/test_retrieve_command.py`: import `run_dir` instead of `RETRIEVAL_EVAL_DIR`, `HEAVY_RETRIEVAL_EVAL_DIR`; assert `request.run_root == run_dir("experiment")` and `request.artifact_root == run_dir("experiment")`; rename the test to `test_retrieve_passes_the_run_tree_and_backend_defaults`.
- `tests/cli/test_metrics_command.py` and `tests/cli/test_rerank_command.py`: import `run_dir`; assert `captured["request"].artifact_root == run_dir("experiment")`.
- The expected remaining output of the grep is empty.

- [ ] **Step 6: Replace the data ignore rules**

In `seed-pipeline/.gitignore` replace the block

```gitignore
# Large generated data, archived outside git
data/heavy/
```

with:

```gitignore
# Pre-migration data folder; the data migration removes this rule with the folder.
data/heavy/

# Large or generated data lives in the Kaggle archive (`seed data push` / `seed data pull`).
data/sources/**/*.pdf
data/sources/leaflets/html/
data/corpus/**
!data/corpus/**/
!data/corpus/**/manifest.json
!data/corpus/**/validation_report.json
data/evaluation/gold/
data/evaluation/runs/**
!data/evaluation/runs/**/
!data/evaluation/runs/**/run.json
!data/evaluation/runs/**/manifest.json
!data/evaluation/runs/**/report.md
data/cache/
data/work/
```

Replace `test_data_and_local_tool_ignore_boundaries` in `seed-pipeline/tests/test_repository_data_policy.py` with:

```python
@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("data/heavy/probe.bin", True),
        ("data/sources/duoc-thu-quoc-gia-viet-nam.pdf", True),
        ("data/sources/leaflets/html/thuoc/panadol.html", True),
        ("data/sources/leaflets/manifest.json", False),
        ("data/sources/leaflets/urls/drug_urls.txt", False),
        ("data/sources/term_glossary.json", False),
        ("data/corpus/rag-final/sections.jsonl", True),
        ("data/corpus/rag-final/manifest.json", False),
        ("data/corpus/rag-final/validation_report.json", False),
        ("data/corpus/formulary/embeddings/model.jsonl", True),
        ("data/corpus/formulary/manifest.json", False),
        ("data/evaluation/gold/section_retrieval_eval.jsonl", True),
        ("data/evaluation/runs/probe/run.json", False),
        ("data/evaluation/runs/probe/candidates/candidates.jsonl", True),
        ("data/evaluation/runs/probe/reports/baseline/top30-window3/report.md", False),
        (
            "data/evaluation/runs/probe/reports/baseline/top30-window3/manifest.json",
            False,
        ),
        (
            "data/evaluation/runs/probe/reports/baseline/top30-window3/metrics.jsonl",
            True,
        ),
        ("data/cache/rerank_scores/model.jsonl", True),
        ("data/work/locks/probe.job.lock", True),
        ("docs/superpowers/specs/probe.md", False),
    ],
)
def test_data_ignore_rules_follow_the_layout(path: str, ignored: bool) -> None:
    assert _is_ignored(path) is ignored
```

(add `import pytest` at the top of the file; `subprocess` and `Path` are already imported).

- [ ] **Step 7: Run the seed gate**

Run the **Seed gate**. Expected: PASS (the `data` marker is deselected by default).

- [ ] **Step 8: Commit**

```bash
git add seed-pipeline/src seed-pipeline/tests seed-pipeline/.gitignore
git commit -m "refactor(seed): point evaluation, cache, corpus and work paths at the new layout"
```

---

### Task 2: Readable lock names with a lock kind

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/job_lock.py`
- Create: `seed-pipeline/tests/conftest.py`
- Test: `seed-pipeline/tests/integrations/kaggle/test_job_lock.py`, `seed-pipeline/tests/evaluation/test_rerank_service.py`

**Interfaces:**
- Consumes: `seed_pipeline.config.paths.DATA_DIR`, `LOCK_DIR` (Task 1).
- Produces: `LockKind = Literal["job", "cache"]`; `lock_file_name(target: Path, kind: LockKind, *, data_dir: Path | None = None) -> str`; `kaggle_job_lock(target: Path, *, lock_root: Path | None = None)`; `kaggle_cache_lock(target: Path, *, lock_root: Path | None = None)`. Default lock root is the module global `LOCK_DIR`, read at call time so tests can patch it.

- [ ] **Step 1: Write the failing lock tests**

Change the imports at the top of `seed-pipeline/tests/integrations/kaggle/test_job_lock.py` to:

```python
import threading
from pathlib import Path

import pytest

from seed_pipeline.integrations.kaggle.job_lock import (
    kaggle_cache_lock,
    kaggle_job_lock,
    lock_file_name,
)
```

and append:

```python
def test_lock_file_is_named_after_the_target_inside_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    target = data / "cache" / "text_embeddings" / "qwen3_embedding_4b_fp16.jsonl"

    assert (
        lock_file_name(target, "job", data_dir=data)
        == "cache__text_embeddings__qwen3_embedding_4b_fp16.jsonl.job.lock"
    )


def test_lock_file_for_a_target_outside_data_is_marked_external(tmp_path: Path) -> None:
    data = tmp_path / "data"
    target = tmp_path / "elsewhere" / "run.json"
    expected = "_external/" + target.resolve().as_posix().lstrip("/")

    assert lock_file_name(target, "cache", data_dir=data) == (
        expected.replace("/", "__") + ".cache.lock"
    )


def test_job_and_cache_locks_on_one_target_can_nest(tmp_path: Path) -> None:
    target = tmp_path / "cache.jsonl"
    lock_root = tmp_path / "locks"

    with (
        kaggle_job_lock(target, lock_root=lock_root),
        kaggle_cache_lock(target, lock_root=lock_root),
    ):
        kinds = sorted(path.name.rsplit(".", 2)[-2] for path in lock_root.iterdir())

    assert kinds == ["cache", "job"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_job_lock.py`
Expected: FAIL with `ImportError: cannot import name 'lock_file_name'`.

- [ ] **Step 3: Rewrite `job_lock.py`**

Replace the whole file with:

```python
from __future__ import annotations

import fcntl
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Literal

from seed_pipeline.config.paths import DATA_DIR, LOCK_DIR

LockKind = Literal["job", "cache"]
MAX_LOCK_FILE_NAME_BYTES = 255


def lock_file_name(
    target: Path, kind: LockKind, *, data_dir: Path | None = None
) -> str:
    """Name a lock after its target: the path under data/ with "/" replaced by "__".

    The kind is part of the name because one command holds a job lock and a cache lock on
    the same target, and flock locks belong to the open handle, not to the process.
    """
    resolved = Path(target).resolve(strict=False)
    base = Path(DATA_DIR if data_dir is None else data_dir).resolve(strict=False)
    try:
        relative = resolved.relative_to(base).as_posix()
    except ValueError:
        relative = "_external/" + resolved.as_posix().lstrip("/")
    name = f"{relative.replace('/', '__')}.{kind}.lock"
    if len(name.encode("utf-8")) > MAX_LOCK_FILE_NAME_BYTES:
        raise ValueError(f"Lock target path is too long for a lock file name: {target}")
    return name


@contextmanager
def _acquire(
    target: Path,
    *,
    kind: LockKind,
    lock_root: Path,
    non_blocking: bool,
) -> Generator[Path, None, None]:
    canonical = str(Path(target).resolve(strict=False))
    lock_path = Path(lock_root) / lock_file_name(target, kind)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Local target {canonical} already has an active Kaggle job; "
                "wait for it to finish or choose a different run/job"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(canonical)
        handle.flush()
        try:
            yield Path(target)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def kaggle_job_lock(
    target: Path,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[Path]:
    root = LOCK_DIR if lock_root is None else lock_root
    return _acquire(target, kind="job", lock_root=root, non_blocking=True)


def kaggle_cache_lock(
    target: Path,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[Path]:
    root = LOCK_DIR if lock_root is None else lock_root
    return _acquire(target, kind="cache", lock_root=root, non_blocking=False)
```

- [ ] **Step 4: Isolate locks written by tests**

Create `seed-pipeline/tests/conftest.py`:

```python
from pathlib import Path

import pytest

from seed_pipeline.integrations.kaggle import job_lock


@pytest.fixture(autouse=True)
def isolated_lock_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Locks taken by tests never land in the real data/work/locks."""
    root = tmp_path_factory.mktemp("locks")
    monkeypatch.setattr(job_lock, "LOCK_DIR", root)
    return root
```

In `seed-pipeline/tests/evaluation/test_rerank_service.py` delete the two lines `monkeypatch.setattr(job_lock, "WORK_DIR", tmp_path / "work")` (lines 221 and 258) and the now-unused `from seed_pipeline.integrations.kaggle import job_lock` imports inside those two tests.

- [ ] **Step 5: Run the seed gate**

Run the **Seed gate**. Expected: PASS, including `tests/embeddings/test_text_cache.py::test_kaggle_backend_propagates_account_and_merges_remote_cache`, which nests both locks on one target.

- [ ] **Step 6: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/job_lock.py seed-pipeline/tests/conftest.py seed-pipeline/tests/integrations/kaggle/test_job_lock.py seed-pipeline/tests/evaluation/test_rerank_service.py
git commit -m "refactor(seed): name Kaggle locks after their target and kind"
```

---

### Task 3: Remove `seed evaluation rejudge-current`

**Files:**
- Delete: `seed-pipeline/src/seed_pipeline/evaluation/rejudge_service.py`, `seed-pipeline/src/seed_pipeline/evaluation/rejudging.py`, `seed-pipeline/tests/evaluation/test_rejudge_service.py`, `seed-pipeline/tests/evaluation/test_rejudging.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/evaluation.py`, `seed-pipeline/tests/cli/test_evaluation_command.py`, `seed-pipeline/README.md`, `seed-pipeline/docs/guides/cli-reference.md`
- Test: `seed-pipeline/tests/test_removed_contracts.py`

**Interfaces:** removes `RejudgeRequest`, `RejudgeResult`, `run_rejudging`, `rejudge_rows` and the `evaluation rejudge-current` command. Nothing else consumes them.

- [ ] **Step 1: Write the failing removed-contract tests**

In `seed-pipeline/tests/test_removed_contracts.py` append `"seed_pipeline.evaluation.rejudge_service",` and `"seed_pipeline.evaluation.rejudging",` to `REMOVED_MODULES`, and replace the command parametrization and its test with:

```python
@pytest.mark.parametrize(
    "args",
    [["vectors", "upload"], ["embed", "chunks"], ["evaluation", "rejudge-current"]],
)
def test_old_commands_are_gone(args: list[str]) -> None:
    assert CliRunner().invoke(app, [*args, "--help"]).exit_code == 2
```

(`--help` matters: without it `rejudge-current` already exits 2 today because its required options are missing, so the test would not fail first.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_removed_contracts.py`
Expected: FAIL for the two modules (spec found) and for `["evaluation", "rejudge-current"]` (exit code is not 2).

- [ ] **Step 3: Delete the command and its modules**

- In `cli/commands/evaluation.py` delete lines 101–155 (the two blank lines before `@evaluation_app.command("rejudge-current")` through the end of the file: `rejudge_current`, `_rejudge_current`), the import of `RejudgeRequest, run_rejudging` (lines 28–31) and `retrieval_run_roots` from the paths import.
- In `tests/cli/test_evaluation_command.py` delete `test_rejudge_current_defaults_to_dry_run`, `test_rejudge_current_apply_forwards_apply_flag` and the `RejudgeResult` import.
- `git rm` the four deleted files listed above.
- In `seed-pipeline/README.md` delete the subsection `### Re-judge metrics trên artifacts hiện có` and its code block; in `docs/guides/cli-reference.md` delete the lines that mention `rejudge-current`.

- [ ] **Step 4: Run the seed gate**

Run the **Seed gate**. Expected: PASS (`tests/test_docs.py` no longer parses a `rejudge-current` command).

- [ ] **Step 5: Commit**

```bash
git add -A seed-pipeline/src/seed_pipeline/evaluation seed-pipeline/src/seed_pipeline/cli/commands/evaluation.py seed-pipeline/tests seed-pipeline/README.md seed-pipeline/docs/guides/cli-reference.md
git commit -m "refactor(seed): remove the one-off rejudge-current command"
```

---

### Task 4: One run tree with named rerank variants and metrics reports

This task changes the run registry, rerank variants and metrics reports together because all three read the same `run.json` record; splitting them would leave the suite red between commits. Task 3 has already deleted the rejudge modules, which were the only other users of these APIs.

**Files:**
- Modify (rewrite): `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py`, `seed-pipeline/src/seed_pipeline/evaluation/rerank_artifacts.py`, `seed-pipeline/src/seed_pipeline/evaluation/metrics_artifacts.py`, `seed-pipeline/src/seed_pipeline/cli/commands/metrics.py`, `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py`
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/metrics_service.py`, `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py`, `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py`, `seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py`, `seed-pipeline/src/seed_pipeline/config/paths.py`, `seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py`
- Test (rewrite): `seed-pipeline/tests/evaluation/test_run_workspace.py`, `seed-pipeline/tests/evaluation/test_rerank_artifacts.py`, `seed-pipeline/tests/evaluation/test_metrics_artifacts.py`, `seed-pipeline/tests/cli/test_metrics_command.py`, `seed-pipeline/tests/cli/test_rerank_command.py`
- Test (modify): `seed-pipeline/tests/evaluation/test_rerank_service.py`, `seed-pipeline/tests/evaluation/test_backend_retrieval.py`, `seed-pipeline/tests/cli/test_retrieve_command.py`, `seed-pipeline/tests/test_removed_contracts.py`
- Unchanged but relied on: `seed-pipeline/tests/evaluation/conftest.py` (`complete_run`, `candidate_bundle`, `complete_rerank_cache`; `complete_run` stores an absolute evaluation path under `tmp_path`, which stays valid)

**Interfaces:**
- Consumes: `paths.DATA_DIR`, `paths.GOLD_DIR`, `paths.RERANK_SCORE_CACHE_DIR`, `paths.run_dir` (Task 1); `ModelSpec.slug` (`"qwen3-reranker:0.6b-fp16"` → `"qwen3_reranker_0_6b_fp16"`); `RerankVariantIdentity.create(candidate_data_sha256, model)` and `.from_values(*, candidate_data_sha256, model, model_sha256, protocol, request_contract_sha256)`; `MetricsArtifactIdentity.create(*, evaluation_sha256, candidate_data_sha256, top_k, window_size, rerank_variant_sha256=None)`; `RerankScoreCache.validate_subset(candidates_path, model)` (result has `.is_complete`, `.missing`, `.sha256`).
- Produces (`run_workspace`): `RUN_SCHEMA_VERSION = 3`; `RunOrigin = Literal["backend", "imported"]`; `RunConflictError`; `RunIdentity` (fields unchanged); `RerankVariantRecord(model: str, variant_sha256: str, artifact_dir: str)`; `RunRecord(identity, origin, candidates_dir: str | None = None, rerank_variants: dict[str, RerankVariantRecord])` keyed by model slug; `identity_differences(current, expected) -> list[str]`; `load_run_record(path) -> RunRecord`; `RunWorkspace(root, identity, origin="backend")` with `record_path`, `candidates_dir`, `reports_dir`, `rerank_dir(model_slug)`, `open_or_create(root, identity, *, origin="backend", force=False)`, `read_record()`, `write_record(record)`, `record_candidates(artifact)`, `resolve(relative)`, `register_rerank_variant(model_slug, variant)`, `variant_records()`; `replace_directory(source, target)`; `evaluation_reference(path, data_dir) -> str`; `resolve_evaluation_reference(reference, data_dir) -> Path`.
- Produces (`rerank_artifacts`): `finalize_run_rerank_bundle(*, workspace, candidate_bundle, cache_path, identity, force=False) -> ArtifactBundle`; `load_registered_rerank_bundle(workspace, record) -> ArtifactBundle`.
- Produces (`metrics_artifacts`): `MetricsArtifactResult` (fields unchanged); `report_dir(run_root, *, top_k, window_size, model=None) -> Path`; `publish_metrics_artifact(run_root, identity, metrics, breakdowns, results, *, top_k, window_size, model=None, variant_sha256=None, force=False) -> MetricsArtifactResult`.
- Produces (`metrics_service`): `MetricsRequest(run_root, top_k=DEFAULT_TOP_K, window_size=3, model=None, force=False)`; `select_rerank_variants(variants, *, model) -> dict[str, RerankVariantRecord]`.
- Produces (`rerank_service`): `RerankRequest(run_root, model, force, dry_run, budget_seconds, request_timeout_seconds, benchmark=False, benchmark_pairs=512, kaggle_account=None)`; dry runs report `target=<dir>` and `missing_pairs=<N>` in `actions`.
- Produces (`paths`): `rerank_score_cache_path(model: str) -> Path` = `RERANK_SCORE_CACHE_DIR / f"{slug}.jsonl"`.
- Produces (`backend_retrieval`): `RetrieveRequest` without `artifact_root`.
- Removed: `LegacyRerankReference`, `migrate_legacy_rerank`, `variant_artifact_dir`, `RunRecord.schema_version/status/rerank_scores_dir/reports_dir/reranker/legacy_*`, `RunWorkspace.artifact_root/artifact_base/rerank_scores_dir/resolve_relative_path/record_rerank_scores/record_reports`, `rerank_checkpoint_path`, `default_rerank_score_cache_path`, `paths.retrieval_run_roots`, `metrics_artifacts.select_rerank_variants`, CLI options `metrics --variant`, `metrics --output-dir`, `rerank --output-dir`, `rerank --candidates`, the `candidates-manifest.json` and `metrics-manifest.json` copies.

- [ ] **Step 1: Write the failing run workspace tests**

Replace `seed-pipeline/tests/evaluation/test_run_workspace.py` with:

```python
import json
from pathlib import Path

import pytest

from seed_pipeline.evaluation.run_workspace import (
    RUN_SCHEMA_VERSION,
    RerankVariantRecord,
    RunConflictError,
    RunIdentity,
    RunWorkspace,
    evaluation_reference,
    load_run_record,
    resolve_evaluation_reference,
)


def identity(*, candidate_k: int = 30, rrf_k: int = 2) -> RunIdentity:
    return RunIdentity(
        evaluation_path="evaluation/gold/section_retrieval_eval.jsonl",
        evaluation_sha256="e" * 64,
        collection_name="chunks_current",
        embedding_model="qwen3-embedding:4b-fp16",
        query_embeddings_sha256="q" * 64,
        retriever="hybrid",
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        limit=None,
        prefetch_k=50,
    )


def test_new_run_writes_schema_three_with_origin(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(
        tmp_path / "run", identity(), origin="imported"
    )

    payload = json.loads(workspace.record_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RUN_SCHEMA_VERSION == 3
    assert payload["origin"] == "imported"
    assert load_run_record(workspace.record_path).identity == identity()


def test_reopening_a_run_with_the_same_identity_keeps_its_files(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    marker = workspace.candidates_dir / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    RunWorkspace.open_or_create(tmp_path / "run", identity())

    assert marker.read_text(encoding="utf-8") == "keep"


def test_a_different_identity_names_the_run_and_the_fields(tmp_path: Path) -> None:
    RunWorkspace.open_or_create(tmp_path / "run", identity())

    with pytest.raises(RunConflictError, match=r"run .*candidate_k, rrf_k"):
        RunWorkspace.open_or_create(
            tmp_path / "run", identity(candidate_k=10, rrf_k=60)
        )


def test_force_replaces_the_whole_run_tree(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    stale = workspace.reports_dir / "old.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    RunWorkspace.open_or_create(tmp_path / "run", identity(rrf_k=60), force=True)

    assert not stale.exists()
    assert load_run_record(tmp_path / "run" / "run.json").identity.rrf_k == 60


def test_record_candidates_stores_a_relative_directory_only(complete_run: Path) -> None:
    record = load_run_record(complete_run / "run.json")

    assert record.candidates_dir == "candidates"
    assert not (complete_run / "candidates-manifest.json").exists()


def test_registering_a_variant_keys_it_by_model_slug(tmp_path: Path) -> None:
    workspace = RunWorkspace.open_or_create(tmp_path / "run", identity())
    variant = RerankVariantRecord(
        "qwen3-reranker:0.6b-fp16", "v" * 64, "rerank/qwen3_reranker_0_6b_fp16"
    )

    workspace.register_rerank_variant("qwen3_reranker_0_6b_fp16", variant)

    assert workspace.variant_records() == {"qwen3_reranker_0_6b_fp16": variant}


def test_older_run_schemas_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 2, "identity": {}}), encoding="utf-8")

    with pytest.raises(RunConflictError, match="Unsupported run schema 2"):
        load_run_record(path)


def test_evaluation_paths_inside_data_are_stored_relative(tmp_path: Path) -> None:
    data = tmp_path / "data"
    gold = data / "evaluation" / "gold" / "section_retrieval_eval.jsonl"

    reference = evaluation_reference(gold, data)

    assert reference == "evaluation/gold/section_retrieval_eval.jsonl"
    assert resolve_evaluation_reference(reference, data) == gold
    outside = tmp_path / "elsewhere.jsonl"
    assert (
        resolve_evaluation_reference(evaluation_reference(outside, data), data)
        == outside.resolve()
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/evaluation/test_run_workspace.py`
Expected: FAIL with `ImportError: cannot import name 'RUN_SCHEMA_VERSION'`.

- [ ] **Step 3: Rewrite `run_workspace.py`**

Replace the whole file with:

```python
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

RUN_SCHEMA_VERSION = 3
RunOrigin = Literal["backend", "imported"]


class RunConflictError(RuntimeError):
    """Raised when a named run or one of its artifacts is reused with different inputs."""


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


@dataclass(frozen=True)
class RerankVariantRecord:
    model: str
    variant_sha256: str
    artifact_dir: str


@dataclass(frozen=True)
class RunRecord:
    identity: RunIdentity
    origin: RunOrigin
    candidates_dir: str | None = None
    rerank_variants: dict[str, RerankVariantRecord] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "origin": self.origin,
            "identity": asdict(self.identity),
            "candidates_dir": self.candidates_dir,
            "rerank_variants": {
                slug: asdict(variant)
                for slug, variant in sorted(self.rerank_variants.items())
            },
        }


def identity_differences(current: RunIdentity, expected: RunIdentity) -> list[str]:
    return [
        item.name
        for item in fields(RunIdentity)
        if getattr(current, item.name) != getattr(expected, item.name)
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_tree(path: Path) -> None:
    """Rename a tree aside first so a crash never leaves a half-deleted target."""
    stale = path.with_name(f".{path.name}.replaced")
    if stale.exists():
        shutil.rmtree(stale)
    os.replace(path, stale)
    shutil.rmtree(stale)


def replace_directory(source: Path, target: Path) -> None:
    source, target = Path(source), Path(target)
    if target.exists():
        _remove_tree(target)
    os.replace(source, target)


def evaluation_reference(path: Path, data_dir: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(data_dir).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_evaluation_reference(reference: str, data_dir: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else Path(data_dir) / path


def _origin(value: object, path: Path) -> RunOrigin:
    if value == "backend":
        return "backend"
    if value == "imported":
        return "imported"
    raise RunConflictError(f"Unknown run origin {value!r} in {path}")


def load_run_record(path: Path) -> RunRecord:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != RUN_SCHEMA_VERSION:
        raise RunConflictError(
            f"Unsupported run schema {schema_version} in {path}; rebuild the run"
        )
    raw_variants = payload.get("rerank_variants") or {}
    if not isinstance(raw_variants, dict):
        raise RunConflictError(f"Run rerank_variants must be an object in {path}")
    variants = {
        str(slug): RerankVariantRecord(
            str(item["model"]), str(item["variant_sha256"]), str(item["artifact_dir"])
        )
        for slug, item in raw_variants.items()
    }
    return RunRecord(
        identity=RunIdentity(**payload["identity"]),
        origin=_origin(payload.get("origin"), path),
        candidates_dir=payload.get("candidates_dir"),
        rerank_variants=variants,
    )


@dataclass(frozen=True)
class RunWorkspace:
    root: Path
    identity: RunIdentity
    origin: RunOrigin = "backend"

    @property
    def record_path(self) -> Path:
        return self.root / "run.json"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def rerank_dir(self, model_slug: str) -> Path:
        return self.root / "rerank" / model_slug

    @classmethod
    def open_or_create(
        cls,
        root: Path,
        identity: RunIdentity,
        *,
        origin: RunOrigin = "backend",
        force: bool = False,
    ) -> RunWorkspace:
        workspace = cls(Path(root), identity, origin)
        if workspace.record_path.exists():
            current = workspace.read_record()
            differences = identity_differences(current.identity, identity)
            if current.origin != origin:
                differences.append("origin")
            if not differences:
                return workspace
            if not force:
                raise RunConflictError(
                    f"run {workspace.root} already exists with different "
                    f"{', '.join(differences)}; use another --run name or --force"
                )
            _remove_tree(workspace.root)
        workspace.root.mkdir(parents=True, exist_ok=True)
        workspace.write_record(RunRecord(identity, origin))
        return workspace

    def read_record(self) -> RunRecord:
        return load_run_record(self.record_path)

    def write_record(self, record: RunRecord) -> None:
        atomic_write_json(self.record_path, record.to_payload())

    def record_candidates(self, artifact: Any) -> None:
        candidate_dir = Path(artifact.data_path).parent
        try:
            relative = candidate_dir.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise RunConflictError(
                f"Candidates {candidate_dir} are outside run {self.root}"
            ) from exc
        current = self.read_record()
        self.write_record(
            RunRecord(
                current.identity, current.origin, relative, current.rerank_variants
            )
        )

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            raise RunConflictError(f"Run artifact paths must be relative: {relative}")
        return self.root / path

    def register_rerank_variant(
        self, model_slug: str, variant: RerankVariantRecord
    ) -> None:
        current = self.read_record()
        variants = {**current.rerank_variants, model_slug: variant}
        self.write_record(
            RunRecord(
                current.identity, current.origin, current.candidates_dir, variants
            )
        )

    def variant_records(self) -> dict[str, RerankVariantRecord]:
        return dict(self.read_record().rerank_variants)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/evaluation/test_run_workspace.py`
Expected: PASS. (Other evaluation tests fail until Step 11.)

- [ ] **Step 5: Point retrieval at the single tree**

In `evaluation/backend_retrieval.py`:
- Delete the field `artifact_root: Path | None = None` from `RetrieveRequest`.
- Import `DATA_DIR` next to `query_embedding_cache_path` from `seed_pipeline.config.paths`, and change the run_workspace import to `from seed_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace, evaluation_reference`.
- In `run_retrieval`, set `evaluation_path=evaluation_reference(request.evaluation_path, DATA_DIR),` in the `RunIdentity(...)` call and open the workspace with:

<!-- fmt: off -->
```python
            workspace = RunWorkspace.open_or_create(
                request.run_root, identity, force=request.force
            )
```
<!-- fmt: on -->

In `cli/commands/retrieve.py` change the paths import to `from seed_pipeline.config.paths import BACKEND_ENV_FILE, GOLD_DIR, run_dir` and replace the first lines of the command body with:

<!-- fmt: off -->
```python
    request = RetrieveRequest(
        evaluation_path=evaluation,
        run_root=run_dir(run),
        retriever=retriever,
```
<!-- fmt: on -->

(the remaining keyword arguments stay; `artifact_root=` is gone).

In `tests/evaluation/test_backend_retrieval.py` delete `artifact_root=tmp_path / "heavy-run",` from `_request`, and append to `test_hybrid_retrieval_uses_cached_query_vectors`:

<!-- fmt: off -->
```python
    record = load_run_record(tmp_path / "run" / "run.json")
    assert record.origin == "backend"
    assert record.identity.evaluation_path == str(
        (tmp_path / "evaluation.jsonl").resolve()
    )
```
<!-- fmt: on -->

In `tests/cli/test_retrieve_command.py` replace the line `assert request.artifact_root == run_dir("experiment")` (written in Task 1) with `assert not hasattr(request, "artifact_root")`.

- [ ] **Step 6: Write the failing rerank artifact tests**

Replace `seed-pipeline/tests/evaluation/test_rerank_artifacts.py` with:

```python
from pathlib import Path

import pytest

from seed_pipeline.artifacts.bundle import ArtifactBundle
from seed_pipeline.evaluation.rerank_artifacts import (
    finalize_run_rerank_bundle,
    load_registered_rerank_bundle,
)
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.run_workspace import (
    RunConflictError,
    RunWorkspace,
    load_run_record,
)
from seed_pipeline.evaluation.variant_identity import RerankVariantIdentity

MODEL = "qwen3-reranker:0.6b-fp16"
SLUG = "qwen3_reranker_0_6b_fp16"


def workspace_for(run: Path) -> RunWorkspace:
    record = load_run_record(run / "run.json")
    return RunWorkspace(run, record.identity, record.origin)


def test_variant_is_published_under_the_model_slug_and_reused(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    workspace = workspace_for(complete_run)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256, MODEL
    )

    first = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    second = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )

    assert first.root == second.root == complete_run / "rerank" / SLUG
    assert first.completion.is_complete
    record = workspace.variant_records()[SLUG]
    assert (record.model, record.variant_sha256, record.artifact_dir) == (
        MODEL,
        identity.sha256,
        f"rerank/{SLUG}",
    )
    assert load_registered_rerank_bundle(workspace, record).root == first.root


def test_a_different_variant_conflicts_unless_forced(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    workspace = workspace_for(complete_run)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256, MODEL
    )
    finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    changed = RerankVariantIdentity.from_values(
        candidate_data_sha256=candidate_bundle.manifest.data_sha256,
        model=MODEL,
        model_sha256=str(identity.payload["model_sha256"]),
        protocol=str(identity.payload["protocol"]),
        request_contract_sha256="c" * 64,
    )

    with pytest.raises(RunConflictError, match=f"rerank/{SLUG}"):
        finalize_run_rerank_bundle(
            workspace=workspace,
            candidate_bundle=candidate_bundle,
            cache_path=complete_rerank_cache.path,
            identity=changed,
        )


def test_force_rebuilds_a_variant_with_the_same_identity(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    workspace = workspace_for(complete_run)
    identity = RerankVariantIdentity.create(
        candidate_bundle.manifest.data_sha256, MODEL
    )
    first = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
    )
    (first.root / "stale.txt").write_text("stale", encoding="utf-8")

    second = finalize_run_rerank_bundle(
        workspace=workspace,
        candidate_bundle=candidate_bundle,
        cache_path=complete_rerank_cache.path,
        identity=identity,
        force=True,
    )

    assert second.root == first.root
    assert not (second.root / "stale.txt").exists()
```

A forced replacement with a changed contract needs cache records under that contract; `LocalRerankBackend` covers it in Step 10 because it re-scores into a matching cache.

- [ ] **Step 7: Rewrite `rerank_artifacts.py`**

Replace the whole file with:

```python
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from seed_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from seed_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    write_json,
)
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunConflictError,
    RunWorkspace,
    replace_directory,
)
from seed_pipeline.evaluation.variant_identity import RerankVariantIdentity
from seed_pipeline.runtime.catalog import require_model


def _load_complete_variant(path: Path) -> ArtifactBundle:
    return load_bundle(path, expected_type="rerank_score_cache", require_complete=True)


def finalize_run_rerank_bundle(
    *,
    workspace: RunWorkspace,
    candidate_bundle: ArtifactBundle,
    cache_path: Path,
    identity: RerankVariantIdentity,
    force: bool = False,
) -> ArtifactBundle:
    model = str(identity.payload["reranker"])
    spec = require_model(model)
    if str(identity.payload["model_sha256"]) != spec.sha256:
        raise ArtifactContractError(
            f"Rerank model digest does not match catalog for {model}"
        )
    target = workspace.rerank_dir(spec.slug)
    record = RerankVariantRecord(
        model, identity.sha256, target.relative_to(workspace.root).as_posix()
    )
    if target.exists() and not force:
        existing = _load_complete_variant(target)
        existing_sha256 = existing.manifest.identity.get("variant_sha256")
        if existing_sha256 != identity.sha256:
            raise RunConflictError(
                f"Rerank variant {target} already exists for different inputs "
                f"(variant_sha256 {existing_sha256} != {identity.sha256}); use --force"
            )
        workspace.register_rerank_variant(spec.slug, record)
        return existing
    cache = RerankScoreCache(
        cache_path,
        model_sha256=spec.sha256,
        request_contract_sha256=str(identity.payload["request_contract_sha256"]),
        rewrite_legacy=False,
    )
    records = cache.subset_records(candidate_bundle.data_path, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        data_path = temporary / "rerank_scores.jsonl"
        with data_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
        manifest = ArtifactManifest.create(
            artifact_type="rerank_score_cache",
            data_path=data_path,
            record_count=len(records),
            identity={
                **identity.payload,
                "variant_sha256": identity.sha256,
                "pair_count": len(records),
            },
        )
        payload = manifest.to_dict()
        payload.update(total=len(records), complete=len(records), missing=0)
        write_json(temporary / "manifest.json", payload)
        _load_complete_variant(temporary)
        replace_directory(temporary, target)
        temporary = Path()
    finally:
        if str(temporary) != ".":
            shutil.rmtree(temporary, ignore_errors=True)
    promoted = _load_complete_variant(target)
    if promoted.manifest.identity.get("variant_sha256") != identity.sha256:
        raise ArtifactContractError(f"Rerank variant identity mismatch at {target}")
    workspace.register_rerank_variant(spec.slug, record)
    return promoted


def load_registered_rerank_bundle(
    workspace: RunWorkspace, record: RerankVariantRecord
) -> ArtifactBundle:
    bundle = _load_complete_variant(workspace.resolve(record.artifact_dir))
    identity = bundle.manifest.identity
    if identity.get("variant_sha256") != record.variant_sha256:
        raise ArtifactContractError(
            f"Rerank variant at {bundle.root} does not match run.json"
        )
    if identity.get("reranker") != record.model:
        raise ArtifactContractError(f"Rerank variant model mismatch at {bundle.root}")
    return bundle
```

(`rewrite_legacy=False` is deleted together with the parameter in Task 5.)

- [ ] **Step 8: Write the failing metrics tests**

Replace `seed-pipeline/tests/evaluation/test_metrics_artifacts.py` with:

```python
import re
from pathlib import Path

import pytest

from seed_pipeline.artifacts.bundle import ArtifactBundle
from seed_pipeline.evaluation.metrics_artifacts import (
    MetricsArtifactResult,
    publish_metrics_artifact,
    report_dir,
)
from seed_pipeline.evaluation.metrics_service import (
    MetricsRequest,
    run_metrics,
    select_rerank_variants,
)
from seed_pipeline.evaluation.rerank_artifacts import finalize_run_rerank_bundle
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunConflictError,
    RunWorkspace,
    load_run_record,
)
from seed_pipeline.evaluation.variant_identity import (
    MetricsArtifactIdentity,
    RerankVariantIdentity,
)

MODEL = "qwen3-reranker:0.6b-fp16"
SLUG = "qwen3_reranker_0_6b_fp16"


def register_variant(
    run: Path, candidate_bundle: ArtifactBundle, cache: RerankScoreCache
) -> None:
    record = load_run_record(run / "run.json")
    finalize_run_rerank_bundle(
        workspace=RunWorkspace(run, record.identity, record.origin),
        candidate_bundle=candidate_bundle,
        cache_path=cache.path,
        identity=RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, MODEL
        ),
    )


def metrics_identity(evaluation_sha256: str) -> MetricsArtifactIdentity:
    return MetricsArtifactIdentity.create(
        evaluation_sha256=evaluation_sha256,
        candidate_data_sha256="candidates",
        top_k=1,
        window_size=3,
    )


def publish_empty(
    root: Path, identity: MetricsArtifactIdentity, *, force: bool = False
) -> MetricsArtifactResult:
    return publish_metrics_artifact(
        root,
        identity,
        {"count": 0, "mrr": 0.0},
        {"eval_group": {}, "difficulty": {}},
        [],
        top_k=1,
        window_size=3,
        force=force,
    )


def test_report_directories_are_named_after_cutoff_and_model(tmp_path: Path) -> None:
    assert report_dir(tmp_path, top_k=30, window_size=3) == (
        tmp_path / "reports" / "baseline" / "top30-window3"
    )
    assert report_dir(tmp_path, top_k=10, window_size=3, model=MODEL) == (
        tmp_path / "reports" / "rerank" / SLUG / "top10-window3"
    )


def test_metrics_publish_baseline_and_rerank_reports_in_the_run(
    complete_run: Path,
    candidate_bundle: ArtifactBundle,
    complete_rerank_cache: RerankScoreCache,
) -> None:
    register_variant(complete_run, candidate_bundle, complete_rerank_cache)

    result = run_metrics(MetricsRequest(complete_run, top_k=1))

    assert result.baseline.artifact_dir == report_dir(
        complete_run, top_k=1, window_size=3
    )
    assert [item.artifact_dir for item in result.reranked] == [
        report_dir(complete_run, top_k=1, window_size=3, model=MODEL)
    ]
    for artifact in (result.baseline, *result.reranked):
        assert {path.name for path in artifact.artifact_dir.iterdir()} == {
            "manifest.json",
            "metrics.jsonl",
            "report.md",
        }
    assert (
        run_metrics(MetricsRequest(complete_run, top_k=1)).baseline == result.baseline
    )
    produced = [
        path.relative_to(complete_run).as_posix() for path in complete_run.rglob("*")
    ]
    assert [path for path in produced if re.search(r"[0-9a-f]{12,}", path)] == []


def test_a_report_for_different_inputs_conflicts_unless_forced(tmp_path: Path) -> None:
    first = publish_empty(tmp_path, metrics_identity("a"))

    with pytest.raises(RunConflictError, match="top1-window3"):
        publish_empty(tmp_path, metrics_identity("b"))
    replaced = publish_empty(tmp_path, metrics_identity("b"), force=True)

    assert replaced.artifact_dir == first.artifact_dir
    assert replaced.metrics_sha256 == metrics_identity("b").sha256


def test_select_variants_returns_all_or_the_named_model() -> None:
    qwen = RerankVariantRecord(MODEL, "a" * 64, f"rerank/{SLUG}")
    bge = RerankVariantRecord(
        "bge-reranker-v2-m3:f16", "b" * 64, "rerank/bge_reranker_v2_m3_f16"
    )
    variants = {SLUG: qwen, "bge_reranker_v2_m3_f16": bge}

    assert list(select_rerank_variants(variants, model=None)) == [
        "bge_reranker_v2_m3_f16",
        SLUG,
    ]
    assert select_rerank_variants(variants, model=MODEL) == {SLUG: qwen}
    with pytest.raises(ValueError, match="available: qwen3-reranker"):
        select_rerank_variants({SLUG: qwen}, model="bge-reranker-v2-m3:f16")
```

- [ ] **Step 9: Rewrite `metrics_artifacts.py` and update `metrics_service.py`**

Replace `metrics_artifacts.py` with:

```python
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from seed_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    ArtifactManifest,
    write_json,
)
from seed_pipeline.evaluation.run_workspace import RunConflictError, replace_directory
from seed_pipeline.evaluation.variant_identity import MetricsArtifactIdentity
from seed_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class MetricsArtifactResult:
    artifact_dir: Path
    report_path: Path
    results_path: Path
    metrics_sha256: str
    model: str | None = None
    variant_sha256: str | None = None


def report_dir(
    run_root: Path, *, top_k: int, window_size: int, model: str | None = None
) -> Path:
    name = f"top{top_k}-window{window_size}"
    if model is None:
        return Path(run_root) / "reports" / "baseline" / name
    return Path(run_root) / "reports" / "rerank" / require_model(model).slug / name


def _load_metrics_bundle(path: Path) -> ArtifactBundle:
    return load_bundle(path, expected_type="metrics_report", require_complete=True)


def publish_metrics_artifact(
    run_root: Path,
    identity: MetricsArtifactIdentity,
    metrics: dict,
    breakdowns: dict,
    results: list[dict],
    *,
    top_k: int,
    window_size: int,
    model: str | None = None,
    variant_sha256: str | None = None,
    force: bool = False,
) -> MetricsArtifactResult:
    from seed_pipeline.evaluation.metrics_service import (
        markdown_breakdown_tables,
        markdown_metric_table,
    )

    target = report_dir(run_root, top_k=top_k, window_size=window_size, model=model)
    if target.exists() and not force:
        existing = _load_metrics_bundle(target)
        existing_sha256 = existing.manifest.identity.get("metrics_sha256")
        if existing_sha256 != identity.sha256:
            raise RunConflictError(
                f"Metrics report {target} already exists for different inputs "
                f"(metrics_sha256 {existing_sha256} != {identity.sha256}); use --force"
            )
        return MetricsArtifactResult(
            target,
            target / "report.md",
            existing.data_path,
            identity.sha256,
            model,
            variant_sha256,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        results_path = temporary / "metrics.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
        (temporary / "report.md").write_text(
            "# Retrieval Metrics\n\n"
            + markdown_metric_table(metrics)
            + "\n\n"
            + markdown_breakdown_tables(breakdowns)
            + "\n",
            encoding="utf-8",
        )
        manifest = ArtifactManifest.create(
            artifact_type="metrics_report",
            data_path=results_path,
            record_count=len(results),
            identity={
                **identity.payload,
                "metrics_sha256": identity.sha256,
                "model": model,
                "variant_sha256": variant_sha256,
            },
        )
        payload = manifest.to_dict()
        payload.update(total=len(results), complete=len(results), missing=0)
        write_json(temporary / "manifest.json", payload)
        _load_metrics_bundle(temporary)
        replace_directory(temporary, target)
        temporary = Path()
    finally:
        if str(temporary) != ".":
            shutil.rmtree(temporary, ignore_errors=True)
    promoted = _load_metrics_bundle(target)
    if promoted.manifest.identity.get("metrics_sha256") != identity.sha256:
        raise ArtifactContractError(f"Metrics identity mismatch at {target}")
    return MetricsArtifactResult(
        target,
        target / "report.md",
        promoted.data_path,
        identity.sha256,
        model,
        variant_sha256,
    )
```

In `metrics_service.py`:
- Replace the imports from `seed_pipeline.config.paths`, `artifact_contracts`, `metrics_artifacts`, `rerank_artifacts` and `run_workspace` with:

```python
from seed_pipeline.config.paths import DATA_DIR
from seed_pipeline.evaluation.artifact_contracts import (
    ArtifactContractError,
    sha256_file,
)
from seed_pipeline.evaluation.metrics_artifacts import (
    MetricsArtifactResult,
    publish_metrics_artifact,
)
from seed_pipeline.evaluation.rerank_artifacts import load_registered_rerank_bundle
from seed_pipeline.evaluation.run_workspace import (
    RerankVariantRecord,
    RunWorkspace,
    load_run_record,
    resolve_evaluation_reference,
)
```

  and drop `ArtifactBundle` from the `seed_pipeline.artifacts.bundle` import (keep `load_bundle`).
- Replace `MetricsRequest` with:

```python
@dataclass(frozen=True)
class MetricsRequest:
    run_root: Path
    top_k: int = DEFAULT_TOP_K
    window_size: int = 3
    model: str | None = None
    force: bool = False
```

- Delete `_load_metrics_candidate_bundle` and `_resolve_metrics_evaluation_path`, and replace `load_and_validate_metric_inputs` with:

```python
def select_rerank_variants(
    variants: dict[str, RerankVariantRecord], *, model: str | None
) -> dict[str, RerankVariantRecord]:
    if model is None:
        return dict(sorted(variants.items()))
    slug = require_model(model).slug
    if slug not in variants:
        available = ", ".join(sorted(item.model for item in variants.values()))
        raise ValueError(f"Unknown reranker model {model}; available: {available}")
    return {slug: variants[slug]}


def load_and_validate_metric_inputs(request: MetricsRequest) -> MetricInputs:
    record = load_run_record(request.run_root / "run.json")
    validate_metrics_cutoff(request.top_k, record.identity.candidate_k)
    workspace = RunWorkspace(request.run_root, record.identity, record.origin)
    if record.candidates_dir is None:
        raise ArtifactContractError(
            f"Run {request.run_root} has no candidates; run seed retrieve first"
        )
    candidate_bundle = load_bundle(
        workspace.resolve(record.candidates_dir),
        expected_type="retrieval_candidates",
        require_complete=True,
    )
    evaluation = resolve_evaluation_reference(record.identity.evaluation_path, DATA_DIR)
    if sha256_file(evaluation) != record.identity.evaluation_sha256:
        raise ArtifactContractError(
            "Run evaluation input changed after retrieval; create a new --run"
        )
    rows = {str(row["query_id"]): row for row in iter_jsonl_objects(evaluation)}
    rerank_inputs: list[RerankMetricInput] = []
    for variant in select_rerank_variants(
        record.rerank_variants, model=request.model
    ).values():
        score_bundle = load_registered_rerank_bundle(workspace, variant)
        spec = require_model(variant.model)
        if spec.rerank_contract is None:
            raise RerankScoreCacheError(
                f"Reranker {variant.model} has no scoring contract"
            )
        score_cache = RerankScoreCache(
            score_bundle.data_path,
            model_sha256=spec.sha256,
            request_contract_sha256=spec.rerank_contract.sha256,
            rewrite_legacy=False,
        )
        subset = score_cache.validate_subset(candidate_bundle.data_path, variant.model)
        if not subset.is_complete:
            raise RerankScoreCacheError(
                f"Rerank score cache is missing {subset.missing} records"
            )
        rerank_inputs.append(
            RerankMetricInput(variant.variant_sha256, variant.model, score_cache)
        )
    return MetricInputs(
        rows,
        CandidateArtifactReader.from_data_path(candidate_bundle.data_path),
        candidate_bundle.manifest.data_sha256,
        record.identity.evaluation_sha256,
        tuple(rerank_inputs),
        request.run_root,
    )
```

- In `run_metrics` replace the two `publish_metrics_artifact(...)` calls with:

<!-- fmt: off -->
```python
    baseline_artifact = publish_metrics_artifact(
        inputs.run_root,
        baseline_identity,
        baseline,
        baseline_breakdowns,
        baseline_rows,
        top_k=request.top_k,
        window_size=request.window_size,
        force=request.force,
    )
```
<!-- fmt: on -->

  and

<!-- fmt: off -->
```python
        reranked_artifacts.append(
            publish_metrics_artifact(
                inputs.run_root,
                identity,
                reranked,
                breakdowns,
                rows,
                top_k=request.top_k,
                window_size=request.window_size,
                model=rerank_input.model,
                variant_sha256=rerank_input.variant_sha256,
                force=request.force,
            )
        )
```
<!-- fmt: on -->

Replace `cli/commands/metrics.py` with:

```python
from __future__ import annotations

from typing import Annotated, cast

import typer

from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import DEFAULT_TOP_K
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.metrics_service import MetricsRequest, run_metrics


def metrics(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    top_k: Annotated[int, typer.Option("--top-k")] = DEFAULT_TOP_K,
    window_size: Annotated[int, typer.Option("--window-size")] = 3,
    model: Annotated[str | None, typer.Option("--model")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
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
```

Replace `tests/cli/test_metrics_command.py` with:

```python
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import seed_pipeline.cli.commands.metrics as metrics_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.metrics_artifacts import MetricsArtifactResult
from seed_pipeline.evaluation.metrics_service import MetricsRequest, MetricsResult

runner = CliRunner()


def fake_metrics_result() -> MetricsResult:
    baseline = Path("reports/baseline/top10-window3")
    rerank = Path("reports/rerank/model_a/top10-window3")
    return MetricsResult(
        baseline=MetricsArtifactResult(
            baseline, baseline / "report.md", baseline / "metrics.jsonl", "base"
        ),
        reranked=(
            MetricsArtifactResult(
                rerank,
                rerank / "report.md",
                rerank / "metrics.jsonl",
                "reranked",
                model="model-a",
                variant_sha256="variant",
            ),
        ),
    )


def capture_request(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, MetricsRequest]
) -> None:
    def fake_run(request: MetricsRequest) -> MetricsResult:
        captured["request"] = request
        return fake_metrics_result()

    monkeypatch.setattr(metrics_command, "run_metrics", fake_run)


def test_metrics_reads_the_run_tree_and_lists_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, MetricsRequest] = {}
    capture_request(monkeypatch, captured)

    result = runner.invoke(
        app, ["--json", "metrics", "--run", "experiment", "--model", "model-a"]
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert (request.run_root, request.model, request.force) == (
        run_dir("experiment"),
        "model-a",
        False,
    )
    assert len(json.loads(result.stdout)["details"]["reranked"]) == 1


def test_metrics_force_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, MetricsRequest] = {}
    capture_request(monkeypatch, captured)

    result = runner.invoke(app, ["metrics", "--run", "experiment", "--force"])

    assert result.exit_code == 0, result.output
    assert captured["request"].force is True


@pytest.mark.parametrize(
    "option", [["--variant", "abcd"], ["--output-dir", "/tmp/reports"]]
)
def test_metrics_rejects_removed_options(option: list[str]) -> None:
    result = runner.invoke(app, ["metrics", "--run", "experiment", *option])

    assert result.exit_code == 2
```

- [ ] **Step 10: Update the rerank service, cache path and CLI**

In `config/paths.py` replace `rerank_score_cache_path` with the function below; delete `retrieval_run_roots`; and remove the `canonical_sha256` import if nothing else in the file uses it:

```python
def rerank_score_cache_path(model: str) -> Path:
    return RERANK_SCORE_CACHE_DIR / f"{require_model(model).slug}.jsonl"
```

Append `"retrieval_run_roots",` to `REMOVED_PATH_NAMES` in `tests/test_removed_contracts.py`. In `evaluation/rerank_score_cache.py` delete `default_rerank_score_cache_path` and its now-unused import.

In `evaluation/rerank_service.py`:
- Imports: drop `canonical_sha256` and `migrate_legacy_rerank`; change the run_workspace import to `from seed_pipeline.evaluation.run_workspace import RerankVariantRecord, RunConflictError, RunWorkspace, load_run_record`; add `from seed_pipeline.evaluation.variant_identity import RerankVariantIdentity` at module level and delete the two function-local imports of it.
- Replace `RerankRequest` and delete `rerank_checkpoint_path`:

```python
@dataclass(frozen=True)
class RerankRequest:
    run_root: Path
    model: str
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float
    benchmark: bool = False
    benchmark_pairs: int = 512
    kaggle_account: str | None = None
```

- Replace `_identity` with these helpers:

```python
def _workspace(request: RerankRequest) -> RunWorkspace:
    record = load_run_record(request.run_root / "run.json")
    return RunWorkspace(request.run_root, record.identity, record.origin)


def _existing_variant(
    workspace: RunWorkspace, slug: str, identity_sha256: str, *, force: bool
) -> RerankVariantRecord | None:
    """The registered variant to reuse, or None when it must be (re)built."""
    existing = workspace.variant_records().get(slug)
    if existing is None or force:
        return None
    if existing.variant_sha256 != identity_sha256:
        raise RunConflictError(
            f"Rerank variant {workspace.rerank_dir(slug)} already exists for "
            "different inputs; use --force"
        )
    return existing
```

- In `LocalRerankBackend.run`, replace everything from `workspace = RunWorkspace.open_or_create(` down to (and including) the `cache = RerankScoreCache(...)` block with:

<!-- fmt: off -->
```python
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model)
        cache = RerankScoreCache(
            cache_path, model_sha256=spec.sha256, request_contract_sha256=contract
        )
        if request.dry_run:
            missing = cache.validate_subset(
                candidate_bundle.data_path, request.model
            ).missing
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (
                    f"target={workspace.rerank_dir(spec.slug)}",
                    f"missing_pairs={missing}",
                    "dry-run",
                ),
            )
```
<!-- fmt: on -->

  keep the scoring loop that follows, and add `force=request.force,` to its `finalize_run_rerank_bundle(...)` call.
- In the same method, create the reranker only when a pair is actually missing, so a complete cache never starts a llama.cpp server: delete the line `reranker = self.reranker_factory(spec, request.request_timeout_seconds)` above the loop, declare `reranker: Reranker | None = None` there instead, and inside the inner loop, right after the `continue` for cached pairs, add:

<!-- fmt: off -->
```python
                if reranker is None:
                    reranker = self.reranker_factory(
                        spec, request.request_timeout_seconds
                    )
```
<!-- fmt: on -->
- In `KaggleRerankBackend._run_kaggle_unlocked`, replace everything from `workspace = RunWorkspace.open_or_create(` down to the line before `from seed_pipeline.integrations.kaggle.auto_profile import (` with:

<!-- fmt: off -->
```python
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model)
        missing = (
            RerankScoreCache(
                cache_path, model_sha256=spec.sha256, request_contract_sha256=contract
            )
            .validate_subset(candidate_bundle.data_path, request.model)
            .missing
        )
        missing_pairs = f"missing_pairs={missing}"
```
<!-- fmt: on -->

  then, further down: delete the later duplicate `if spec.rerank_contract is None`, `contract = ...` and `cache_path = rerank_score_cache_path(request.model, spec.sha256, contract)` lines; change `(f"profile={resolution.action}",)` to `(missing_pairs, f"profile={resolution.action}")`; change `(f"target rerank variant {identity.sha256}", *actions)` to `(missing_pairs, f"target={workspace.rerank_dir(spec.slug)}", *actions)`; and add `force=request.force,` to its `finalize_run_rerank_bundle(...)` call.

Replace `cli/commands/rerank.py` with:

```python
from __future__ import annotations

from typing import Annotated, cast

import typer

from seed_pipeline.cli.options import Backend
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RERANKER_MODEL,
)
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
    RerankStageResult,
)


def rerank(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_RERANKER_MODEL,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[
        int, typer.Option("--budget-seconds")
    ] = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    adapter = (
        LocalRerankBackend() if backend is Backend.LOCAL else KaggleRerankBackend()
    )
    request = RerankRequest(
        run_root=run_dir(run),
        model=model,
        force=force,
        dry_run=dry_run,
        budget_seconds=budget_seconds,
        request_timeout_seconds=request_timeout_seconds,
        kaggle_account=kaggle_account,
    )
    run_handler(state_from_context(ctx), lambda: _result(adapter.run(request)))


def _result(result: RerankStageResult) -> CommandResult:
    return CommandResult(
        "rerank",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.artifact_dir,
        {
            "actions": result.actions,
            "subset_sha256": result.subset_sha256,
            "variant_sha256": result.variant_sha256,
            "benchmark_report": result.benchmark_report,
            "benchmark_levels": result.benchmark_levels,
        },
    )
```

The command has no `--benchmark` option today, so `test_rerank_benchmark_rejects_local_backend` keeps passing through Typer's exit code 2 for unknown options.

Replace `tests/cli/test_rerank_command.py` with:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import seed_pipeline.cli.commands.rerank as rerank_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import run_dir

runner = CliRunner()


def fake_backend(captured: dict, *, incomplete: bool):
    class FakeBackend:
        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(
                artifact_dir=None if incomplete else Path("run/rerank/model"),
                variant_sha256="variant",
                subset_sha256=None if incomplete else "subset",
                actions=("missing_pairs=0",),
                incomplete=incomplete,
                benchmark_report=None,
                benchmark_levels=0,
            )

    return FakeBackend


def test_rerank_reads_the_run_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "LocalRerankBackend", fake_backend(captured, incomplete=False)
    )

    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--model", "qwen3-reranker:0.6b-fp16"]
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].run_root == run_dir("experiment")
    assert captured["request"].model == "qwen3-reranker:0.6b-fp16"
    assert "variant_sha256=variant" in result.stdout


@pytest.mark.parametrize(
    "option", [["--output-dir", "/tmp/cache.jsonl"], ["--candidates", "/tmp/c"]]
)
def test_rerank_rejects_removed_options(option: list[str]) -> None:
    result = runner.invoke(app, ["rerank", "--run", "experiment", *option])

    assert result.exit_code == 2


def test_rerank_benchmark_rejects_local_backend() -> None:
    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--backend", "local", "--benchmark"]
    )

    assert result.exit_code == 2


def test_rerank_passes_kaggle_account_to_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "KaggleRerankBackend", fake_backend(captured, incomplete=True)
    )

    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "kaggle",
            "--kaggle-account",
            "acc2",
        ],
    )

    assert result.exit_code == 3
    assert captured["request"].kaggle_account == "acc2"
```

In `tests/evaluation/test_rerank_service.py`:
- Delete `candidates_dir=None,` from `request(...)`.
- Delete `test_kaggle_migrates_matching_legacy_variant_before_submit` and the `import json` line.
- Add below the imports, so no test writes scores into the real `data/cache/rerank_scores`:

```python
@pytest.fixture(autouse=True)
def isolated_rerank_cache(tmp_path, monkeypatch):
    from seed_pipeline.config import paths

    monkeypatch.setattr(paths, "RERANK_SCORE_CACHE_DIR", tmp_path / "rerank-cache")
```

- Add at the end of the file:

```python
def test_local_dry_run_reports_target_and_missing_pairs(complete_run):
    result = fake_local_backend().run(
        request(complete_run, "qwen3-reranker:0.6b-fp16", dry_run=True)
    )

    assert "missing_pairs=1" in result.actions
    assert (
        f"target={complete_run / 'rerank' / 'qwen3_reranker_0_6b_fp16'}"
        in result.actions
    )
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_forced_rerank_rescores_into_the_same_variant(complete_run):
    backend = fake_local_backend()
    first = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))

    second = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16", force=True))

    assert second.artifact_dir == first.artifact_dir
    assert "scored=1" in second.actions
    assert list(load_run_record(complete_run / "run.json").rerank_variants) == [
        "qwen3_reranker_0_6b_fp16"
    ]
```

- Add a test that a complete score cache publishes the variant without creating a reranker:

```python
def test_local_rerank_from_a_complete_cache_starts_no_reranker(
    complete_run, complete_rerank_cache, monkeypatch
):
    monkeypatch.setattr(
        rerank_service,
        "rerank_score_cache_path",
        lambda _model: complete_rerank_cache.path,
    )

    def no_server(_spec, _timeout):
        raise AssertionError("a complete cache must not start a reranker")

    result = LocalRerankBackend(reranker_factory=no_server).run(
        request(complete_run, "qwen3-reranker:0.6b-fp16")
    )

    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"
    assert "scored=0" in result.actions
```

- [ ] **Step 11: Run the seed gate**

Run the **Seed gate**. Expected: PASS. Then run (in `seed-pipeline/`) `grep -rnE 'artifact_root|artifact_base|candidates-manifest|metrics-manifest|legacy_rerank|retrieval_run_roots|resolve_relative_path|rerank_checkpoint_path' src tests`. Expected: no output.

- [ ] **Step 12: Commit**

```bash
git add -A seed-pipeline/src seed-pipeline/tests
git commit -m "refactor(seed): keep each evaluation run in one tree without hash names"
```

---

### Task 5: Sealed cache records only and dead evaluation code

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/cache/jsonl_records.py`, `seed-pipeline/src/seed_pipeline/embeddings/text_cache.py`, `seed-pipeline/src/seed_pipeline/evaluation/query_embedding_cache.py`, `seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py`, `seed-pipeline/src/seed_pipeline/evaluation/query_embedding_service.py`, `seed-pipeline/src/seed_pipeline/evaluation/preload_query_embeddings.py`, `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/query_embed.py`, `seed-pipeline/src/seed_pipeline/evaluation/rerank_artifacts.py`, `seed-pipeline/src/seed_pipeline/evaluation/metrics_service.py`, `seed-pipeline/src/seed_pipeline/config/paths.py`
- Create: `seed-pipeline/tests/cache/test_jsonl_records.py`
- Delete: `seed-pipeline/src/seed_pipeline/evaluation/query_embedding_artifact.py`
- Test: `seed-pipeline/tests/test_removed_contracts.py`

**Interfaces:**
- Produces: `load_records(path: Path) -> list[dict[str, Any]]` (no flag, no tuple).
- Removed: `allow_legacy`, `RerankScoreCache.rewrite_legacy`, `finalize_rerank_cache`, module `evaluation/query_embedding_artifact.py` (`migrate_query_bundle`, `query_cache_identity`), `build_query_snapshot`, `inspect_query_cache`, `finalize_query_cache`, `QuerySnapshot`, `QueryCacheArtifact`, `default_query_embedding_cache_path`, `QueryEmbeddingCache.default_path`, `query_checkpoint_path`, `query_embedding_identity`, `_evaluation_sha256` (query service), `_legacy_query_records`, `paths.query_embedding_bundle_dir`.

- [ ] **Step 1: Write the failing cache tests**

Create `seed-pipeline/tests/cache/test_jsonl_records.py`:

```python
import json
from pathlib import Path

import pytest

from seed_pipeline.cache.jsonl_records import (
    CacheRecordError,
    append_record,
    load_records,
)
from seed_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
)


def test_load_records_returns_sealed_records(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    append_record(path, {"value": 1}, schema="probe-v1")

    records = load_records(path)

    assert [record["value"] for record in records] == [1]


def test_unsealed_records_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"value": 1}) + "\n", encoding="utf-8")

    with pytest.raises(CacheRecordError, match="missing"):
        load_records(path)


def test_query_cache_rejects_unsealed_records(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {"model": "m", "query_id": "q", "query_hash": "h", "embedding": [0.1]}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(QueryEmbeddingCacheError):
        QueryEmbeddingCache(path, vector_dim=1)
```

Add `"query_embedding_bundle_dir"` to `REMOVED_PATH_NAMES` and `"seed_pipeline.evaluation.query_embedding_artifact",` to `REMOVED_MODULES` in `tests/test_removed_contracts.py`, and add:

```python
@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("seed_pipeline.evaluation.query_embedding_service", "query_checkpoint_path"),
        ("seed_pipeline.evaluation.rerank_score_cache", "finalize_rerank_cache"),
        (
            "seed_pipeline.integrations.kaggle.workers.query_embed",
            "_legacy_query_records",
        ),
    ],
)
def test_legacy_cache_helpers_are_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/cache tests/test_removed_contracts.py`
Expected: FAIL: `load_records` returns a tuple, the query cache seals the legacy record instead of raising, and the helpers still exist.

- [ ] **Step 3: Remove the legacy branches**

- `cache/jsonl_records.py`: change the signature to `def load_records(path: Path) -> list[dict[str, Any]]:`, delete the `if CHECKSUM_FIELD not in record or "cache_schema" not in record:` block (lines 134–139) so every record goes through `verify_record`, delete the `migrated` flag, and make all three returns (`return [], False` at line 97, `return records, migrated` at line 126 and the return at line 141) return only the list. Keep the torn-last-line repair. In `merge_records` use `existing = load_records(Path(path))`.
- `embeddings/text_cache.py:112`: `records = load_records(self.path)`.
- `evaluation/query_embedding_cache.py`: `records = load_records(self.path)`; delete the sealing (`if "record_sha256" not in record ...`) and the final `if has_legacy or migrated: rewrite_records(...)`; delete `default_query_embedding_cache_path` and `default_path`; drop the `seal_record` and `QUERY_EMBEDDING_CACHE_DIR` imports and the `migrated` and `normalized_records` locals; keep `rewrite_records`, which `replace_keys` still uses.
- `evaluation/rerank_score_cache.py`: delete the `rewrite_legacy` field, `load_records(self.path)`, delete the sealing branch and the conditional rewrite, delete `finalize_rerank_cache`; drop the `seal_record`, `ArtifactManifest`, `Completion` and `write_json` imports and the `migrated` and `normalized_records` locals; keep `json`, `rewrite_records` and `CandidateArtifactReader`.
- Remove `rewrite_legacy=False,` from `evaluation/rerank_artifacts.py` and `evaluation/metrics_service.py`.
- `git rm seed-pipeline/src/seed_pipeline/evaluation/query_embedding_artifact.py`: its last caller, `query_embedding_identity`, is deleted in the next bullet.
- `evaluation/query_embedding_service.py`: delete `query_embedding_identity`, `query_checkpoint_path`, `_evaluation_sha256`, the `if not partial_cache.exists(): legacy_bundle = ...` block and the imports of `query_embedding_bundle_dir`, `canonical_sha256` and the whole `seed_pipeline.evaluation.query_embedding_artifact` import block.
- `evaluation/preload_query_embeddings.py`: make `cache_path: Path` a required parameter, delete `DEFAULT_EVAL_JSONL`, the `default_query_embedding_cache_path` fallback and both now-unused imports (`GOLD_DIR` and `default_query_embedding_cache_path`).
- `integrations/kaggle/workers/query_embed.py`: delete `_legacy_query_records` and the `if not journal.records: legacy = ...` block.
- `config/paths.py`: delete `query_embedding_bundle_dir`.

- [ ] **Step 4: Run the seed gate**

Run the **Seed gate**. Expected: PASS, including `tests/integrations/kaggle/test_worker_bundle_imports.py`.

- [ ] **Step 5: Commit**

```bash
git add -A seed-pipeline/src seed-pipeline/tests
git commit -m "refactor(seed): read sealed cache records only and delete unused cache helpers"
```

---

### Task 6: Kaggle profiles by model and one kernel reference

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/runtime/runtime_profiles.py`, `seed-pipeline/src/seed_pipeline/integrations/kaggle/auto_profile.py`, `seed-pipeline/src/seed_pipeline/integrations/kaggle/kernels.py`
- Test: `seed-pipeline/tests/runtime/test_runtime_profiles.py`, `seed-pipeline/tests/integrations/kaggle/test_auto_profile.py`, `seed-pipeline/tests/integrations/kaggle/test_kernels.py`

**Interfaces:**
- Produces: `RuntimeProfileStore.path(identity, *, model_slug) -> Path` returning `root / workload / f"{model_slug}.json"`; `PipelineKernelService.reference(job) -> str` (the only reference); `references` removed.

- [ ] **Step 1: Write the failing tests**

In `tests/runtime/test_runtime_profiles.py::test_profile_store_round_trips_and_rejects_corruption` add after the save call:

<!-- fmt: off -->
```python
    assert path.parent.parent == tmp_path
    assert path.name == "qwen3_reranker_0_6b_fp16.json"
```
<!-- fmt: on -->

In `tests/integrations/kaggle/test_auto_profile.py::test_inference_cache_policy_change_invalidates_cached_profile` add at the end:

<!-- fmt: off -->
```python
    saved = list(profile_root.rglob("*.json"))
    assert len(saved) == 1
    assert (
        json.loads(saved[0].read_text(encoding="utf-8"))["identity"][
            "inference_cache_policy_sha256"
        ]
        == original(require_model(MODEL)).sha256
    )
```
<!-- fmt: on -->

In `tests/integrations/kaggle/test_kernels.py` rename `test_kernel_references_prefer_sixteen_characters_and_keep_legacy(tmp_path)` to `test_kernel_reference_uses_sixteen_identity_characters(tmp_path)`, keep its body, and replace its two final asserts (lines 107–111) with:

<!-- fmt: off -->
```python
    assert service.reference(job) == f"owner/rerank-{job.identity.sha256[:16]}"
    assert not hasattr(service, "references")
```
<!-- fmt: on -->

Rename `test_discover_falls_back_when_missing_preferred_slug_is_denied(tmp_path)` to `test_discover_reports_absent_when_the_kernel_is_confirmed_missing(tmp_path)`, keep its body and its local `FakeKernelService`, and replace its three final asserts (lines 158–160) with:

<!-- fmt: off -->
```python
    assert state.presence is KernelPresence.ABSENT
    assert state.reference == service.reference(job)
    assert fake.checked == [service.reference(job)]
```
<!-- fmt: on -->

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/runtime/test_runtime_profiles.py tests/integrations/kaggle/test_auto_profile.py tests/integrations/kaggle/test_kernels.py`
Expected: FAIL on the path shape, the two saved profiles, and `references` still existing.

- [ ] **Step 3: Implement**

- `runtime_profiles.py:278`: `return self.root / workload / f"{model_slug}.json"`.
- `auto_profile.py` dry-run path (lines 138–143): `Path(profile_root) / workload / f"{spec.slug}.json"`.
- `kernels.py`: delete `references`; make `reference` return `f"{self.owner}/{job.stage.value}-{job.identity.sha256[:16]}"`; make `discover` inspect only `self.reference(job)`:

<!-- fmt: off -->
```python
    def discover(self, job: StageJob) -> KernelRemoteState:
        reference = self.reference(job)
        state = self.service.inspect_state(reference)
        if state.presence is KernelPresence.UNKNOWN:
            if (
                "kernels.get" in state.detail.casefold()
                and self.service.confirm_missing(reference)
            ):
                return KernelRemoteState(reference, KernelPresence.ABSENT)
            raise KaggleRemoteStateError(
                f"Cannot inspect Kaggle kernel {reference}: {state.detail}"
            )
        return state
```
<!-- fmt: on -->

- [ ] **Step 4: Run the seed gate**

Run the **Seed gate**. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/runtime/runtime_profiles.py seed-pipeline/src/seed_pipeline/integrations/kaggle seed-pipeline/tests/runtime seed-pipeline/tests/integrations/kaggle
git commit -m "refactor(seed): store one Kaggle profile per model and drop the legacy kernel slug"
```

---

### Task 7: Build workspace without a hash and no corpus manifest copies

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/artifacts/paths.py`, `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py`
- Create: `seed-pipeline/tests/artifacts/test_paths.py`
- Modify: `seed-pipeline/tests/orchestration/test_build_metadata.py` (delete `test_publish_corpus_metadata_copies_small_contract_files`)

**Interfaces:**
- Produces: `ArtifactPaths.create(work_root: Path) -> ArtifactPaths` with `root = work_root / "in-progress"`; `retain_failed_workspace(paths)` keeps `work_root / "failed" / "latest"`; `BuildConfig.work_root` default `BUILD_WORK_DIR`; no `BuildConfig.manifest_dir`, no `publish_corpus_metadata`.

- [ ] **Step 1: Write the failing test**

Create `seed-pipeline/tests/artifacts/test_paths.py`:

```python
import re
from pathlib import Path

import pytest

from seed_pipeline.artifacts.paths import ArtifactPaths, retain_failed_workspace


def test_build_workspace_has_a_fixed_readable_name(tmp_path: Path) -> None:
    paths = ArtifactPaths.create(tmp_path / "build")

    assert paths.root == tmp_path / "build" / "in-progress"
    assert all(directory.is_dir() for directory in paths.directories())
    assert not any(
        re.search(r"[0-9a-f]{12,}", part)
        for part in paths.root.relative_to(tmp_path).parts
    )


def test_an_unfinished_workspace_blocks_a_new_build(tmp_path: Path) -> None:
    ArtifactPaths.create(tmp_path / "build")

    with pytest.raises(FileExistsError, match="in-progress"):
        ArtifactPaths.create(tmp_path / "build")


def test_failed_workspace_is_kept_as_latest(tmp_path: Path) -> None:
    paths = ArtifactPaths.create(tmp_path / "build")

    latest = retain_failed_workspace(paths)

    assert latest == tmp_path / "build" / "failed" / "latest"
    assert not paths.root.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/artifacts/test_paths.py`
Expected: FAIL with `TypeError: ArtifactPaths.create() missing 1 required keyword-only argument: 'build_id'`.

- [ ] **Step 3: Implement**

In `artifacts/paths.py` replace `create` with:

<!-- fmt: off -->
```python
    @classmethod
    def create(cls, work_root: Path) -> ArtifactPaths:
        root = Path(work_root) / "in-progress"
        if root.exists():
            raise FileExistsError(
                f"Build workspace already exists: {root}; inspect or delete it first"
            )
        root.mkdir(parents=True)
        paths = cls(root=root)
        for directory in paths.directories():
            directory.mkdir(parents=True, exist_ok=True)
        return paths
```
<!-- fmt: on -->

In `orchestration/build_corpus.py`: import `BUILD_WORK_DIR` instead of `WORK_DIR`; set `work_root: Path = BUILD_WORK_DIR`; delete `manifest_dir` from `BuildConfig`, delete `publish_corpus_metadata` and both calls to it in `run_build`; call `ArtifactPaths.create(config.work_root)`; delete `import os`, which only `publish_corpus_metadata` used. `cli/commands/build.py` never passes `manifest_dir` and stays unchanged. Delete `test_publish_corpus_metadata_copies_small_contract_files` from `tests/orchestration/test_build_metadata.py`.

- [ ] **Step 4: Run the seed gate**

Run the **Seed gate**. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A seed-pipeline/src/seed_pipeline/artifacts seed-pipeline/src/seed_pipeline/orchestration seed-pipeline/tests/artifacts seed-pipeline/tests/orchestration
git commit -m "refactor(seed): build in work/build/in-progress and stop copying corpus manifests"
```

---

### Task 8: Crawl leaflets into html, urls and a manifest

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/corpus/sources/leaflet_source.py`
- Modify: `seed-pipeline/src/seed_pipeline/corpus/sources/crawl.py` (rewrite), `seed-pipeline/src/seed_pipeline/corpus/sources/__init__.py`, `seed-pipeline/src/seed_pipeline/cli/commands/source.py`, `seed-pipeline/src/seed_pipeline/config/paths.py`
- Delete: `seed-pipeline/src/seed_pipeline/corpus/crawling/collect_urls.py`, `seed-pipeline/src/seed_pipeline/corpus/crawling/download_html.py`; in `corpus/crawling/parse_html.py` delete the unused `DEFAULT_HTML_DIR` and `DEFAULT_OUTPUT_DIR` constants and their imports
- Create: `seed-pipeline/tests/corpus/test_crawl.py`, `seed-pipeline/tests/corpus/test_leaflet_source.py`

**Interfaces:**
- Produces (`paths`): `SOURCES_DIR`, `LEAFLETS_DIR = SOURCES_DIR / "leaflets"`, `LEAFLETS_HTML_DIR`, `LEAFLETS_URLS_DIR`, `LEAFLETS_MANIFEST_PATH`; removes `RESOURCES_ANKHANG_DIR`, `RAW_ANKHANG_HTML_DIR`.
- Produces (`leaflet_source`): `LEAFLET_SOURCE_SCHEMA = "leaflet-source-v1"`; `LeafletSourceError(RuntimeError)`; `LeafletFile(path, size, sha256, source_url)`; `LeafletManifest(sitemap_url, crawled_on, url_list_sha256, files)` with `to_dict()`; `file_sha256(path) -> str`; `read_leaflet_manifest(path) -> LeafletManifest`; `write_leaflet_manifest(path, manifest) -> None`.
- Produces (`crawl`): `CrawlRequest(leaflets_dir, sitemap_url=None, workers=8, request_timeout_seconds=10.0, force=False, dry_run=False)`; `CrawlResult(total, downloaded, skipped, failed, manifest_path)`; `Fetch = Callable[[str, float], bytes]`; `parse_locations`, `leaflet_page(url) -> tuple[str, str] | None`, `collect_page_urls(sitemap_url, fetch, *, timeout, workers)`, `crawl_leaflets(request, fetch, *, today: date) -> CrawlResult`; `CrawlConfigurationError`, `CrawlFetchError`.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/corpus/test_leaflet_source.py`:

```python
from pathlib import Path

import pytest

from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    LeafletSourceError,
    read_leaflet_manifest,
    write_leaflet_manifest,
)


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = LeafletManifest(
        sitemap_url="https://example.test/sitemap.xml",
        crawled_on="2026-09-14",
        url_list_sha256="a" * 64,
        files=(
            LeafletFile(
                "thuoc-a/one.html", 3, "b" * 64, "https://example.test/thuoc-a/one"
            ),
        ),
    )
    path = tmp_path / "manifest.json"

    write_leaflet_manifest(path, manifest)

    assert read_leaflet_manifest(path) == manifest


def test_manifest_with_another_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": "other", "files": []}', encoding="utf-8")

    with pytest.raises(LeafletSourceError, match="leaflet-source-v1"):
        read_leaflet_manifest(path)
```

Create `seed-pipeline/tests/corpus/test_crawl.py`:

```python
import json
from datetime import date
from pathlib import Path

import pytest

from seed_pipeline.corpus.sources.crawl import (
    CrawlConfigurationError,
    CrawlFetchError,
    CrawlRequest,
    crawl_leaflets,
    leaflet_page,
)

SITEMAP = "https://example.test/sitemap.xml"
PAGES = {
    SITEMAP: b"<sitemapindex><loc>https://example.test/sub-1.xml</loc>"
    b"<loc>https://example.test/sub-2.xml</loc></sitemapindex>",
    "https://example.test/sub-1.xml": b"<urlset><loc>https://example.test/thuoc-giam-dau/panadol</loc>"
    b"<loc>https://example.test/sua-tam/xa-bong</loc></urlset>",
    "https://example.test/sub-2.xml": b"<urlset><loc>https://example.test/tim-mach-huyet-ap/amlor</loc>"
    b"<loc>https://example.test/thuoc-giam-dau</loc>"
    b"<loc>https://example.test/thuoc-giam-dau/hong/1</loc></urlset>",
    "https://example.test/thuoc-giam-dau/panadol": b"<html>panadol</html>",
    "https://example.test/tim-mach-huyet-ap/amlor": b"<html>amlor</html>",
}


def fetch(url: str, timeout: float) -> bytes:
    del timeout
    return PAGES[url]


@pytest.mark.parametrize(
    ("url", "page"),
    [
        ("https://example.test/thuoc-giam-dau/panadol", ("thuoc-giam-dau", "panadol")),
        (
            "https://example.test/tim-mach-huyet-ap/amlor",
            ("tim-mach-huyet-ap", "amlor"),
        ),
        ("https://example.test/sua-tam/xa-bong", None),
        ("https://example.test/thuoc-giam-dau", None),
        ("https://example.test/thuoc-giam-dau/hong/1", None),
    ],
)
def test_leaflet_pages_are_two_segment_drug_urls(url: str, page: object) -> None:
    assert leaflet_page(url) == page


def test_crawl_writes_html_tree_url_lists_and_manifest(tmp_path: Path) -> None:
    leaflets = tmp_path / "leaflets"

    result = crawl_leaflets(
        CrawlRequest(leaflets, sitemap_url=SITEMAP, workers=2),
        fetch,
        today=date(2026, 9, 14),
    )

    assert (result.total, result.downloaded, result.skipped, result.failed) == (
        2,
        2,
        0,
        0,
    )
    assert (
        leaflets / "html" / "thuoc-giam-dau" / "panadol.html"
    ).read_bytes() == PAGES["https://example.test/thuoc-giam-dau/panadol"]
    assert (leaflets / "urls" / "drug_urls.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "https://example.test/thuoc-giam-dau/panadol",
        "https://example.test/tim-mach-huyet-ap/amlor",
    ]
    assert (
        len(
            (leaflets / "urls" / "all_urls.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 5
    )
    manifest = json.loads((leaflets / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "leaflet-source-v1"
    assert manifest["sitemap_url"] == SITEMAP
    assert manifest["crawled_on"] == "2026-09-14"
    assert [item["path"] for item in manifest["files"]] == [
        "thuoc-giam-dau/panadol.html",
        "tim-mach-huyet-ap/amlor.html",
    ]
    assert (
        manifest["files"][0]["source_url"]
        == "https://example.test/thuoc-giam-dau/panadol"
    )


def test_later_crawls_read_the_sitemap_from_the_manifest_and_skip_files(
    tmp_path: Path,
) -> None:
    leaflets = tmp_path / "leaflets"
    crawl_leaflets(
        CrawlRequest(leaflets, sitemap_url=SITEMAP), fetch, today=date(2026, 9, 14)
    )

    again = crawl_leaflets(CrawlRequest(leaflets), fetch, today=date(2026, 9, 15))

    assert (again.downloaded, again.skipped) == (0, 2)


def test_first_crawl_needs_a_sitemap_url(tmp_path: Path) -> None:
    with pytest.raises(CrawlConfigurationError, match="--sitemap-url"):
        crawl_leaflets(
            CrawlRequest(tmp_path / "leaflets"), fetch, today=date(2026, 9, 14)
        )


def test_failed_pages_are_counted_and_left_out_of_the_manifest(tmp_path: Path) -> None:
    def flaky(url: str, timeout: float) -> bytes:
        if url.endswith("/amlor"):
            raise CrawlFetchError(url)
        return fetch(url, timeout)

    leaflets = tmp_path / "leaflets"
    result = crawl_leaflets(
        CrawlRequest(leaflets, sitemap_url=SITEMAP), flaky, today=date(2026, 9, 14)
    )

    assert result.failed == 1
    manifest = json.loads((leaflets / "manifest.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == [
        "thuoc-giam-dau/panadol.html"
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/corpus/test_crawl.py tests/corpus/test_leaflet_source.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'seed_pipeline.corpus.sources.leaflet_source'`.

- [ ] **Step 3: Create `leaflet_source.py`**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEAFLET_SOURCE_SCHEMA = "leaflet-source-v1"


class LeafletSourceError(RuntimeError):
    """Raised when the leaflet HTML tree or its manifest is invalid."""


@dataclass(frozen=True)
class LeafletFile:
    path: str
    size: int
    sha256: str
    source_url: str | None


@dataclass(frozen=True)
class LeafletManifest:
    sitemap_url: str
    crawled_on: str
    url_list_sha256: str
    files: tuple[LeafletFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEAFLET_SOURCE_SCHEMA,
            "sitemap_url": self.sitemap_url,
            "crawled_on": self.crawled_on,
            "url_list_sha256": self.url_list_sha256,
            "file_count": len(self.files),
            "files": [asdict(item) for item in self.files],
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_leaflet_manifest(path: Path) -> LeafletManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEAFLET_SOURCE_SCHEMA:
        raise LeafletSourceError(
            f"{path} is not a {LEAFLET_SOURCE_SCHEMA} manifest "
            f"(schema_version {payload.get('schema_version')!r})"
        )
    files = tuple(
        LeafletFile(
            str(item["path"]),
            int(item["size"]),
            str(item["sha256"]),
            None if item.get("source_url") is None else str(item["source_url"]),
        )
        for item in payload["files"]
    )
    if int(payload["file_count"]) != len(files):
        raise LeafletSourceError(f"{path} file_count does not match its file list")
    return LeafletManifest(
        str(payload["sitemap_url"]),
        str(payload["crawled_on"]),
        str(payload["url_list_sha256"]),
        files,
    )


def write_leaflet_manifest(path: Path, manifest: LeafletManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
```

- [ ] **Step 4: Rewrite `crawl.py`**

```python
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    file_sha256,
    read_leaflet_manifest,
    write_leaflet_manifest,
)

Fetch = Callable[[str, float], bytes]

DRUG_CATEGORY_PREFIX = "thuoc-"
DRUG_CATEGORIES = frozenset(
    {
        "tim-mach-huyet-ap",
        "duong-tieu-hoa",
        "khang-sinh-khang-nam",
        "tri-ho-hen-phe-quan",
        "thuoc-nho-mat-tai-mui-hong",
        "tri-giun-san",
        "thuoc-dieu-tri-ung-thu",
        "chong-di-ung",
        "thuoc-ke-don",
        "khang-nam-khang-virus",
        "thuoc-dung-ngoai-da",
        "giam-dau-ha-sot",
        "vitamin-va-khoang-chat",
    }
)
NON_DRUG_CATEGORIES = frozenset(
    {
        "sua-rua-mat",
        "kem-chong-nang",
        "kem-duong-da",
        "tinh-chat-duong-da",
        "mat-na-cham-soc-da",
        "dau-goi-dau",
        "kem-danh-rang",
        "ban-chai-danh-rang",
        "nuoc-suc-mieng",
        "bang-ve-sinh",
        "bao-cao-su",
        "ta-cho-be",
        "sua-bot-cong-thuc",
        "khan-uot",
        "son-duong-moi",
        "sua-tam",
        "kem-tri-mun",
        "xit-khoang",
        "nuoc-hoa-hong",
        "tay-trang",
        "kem-duong-the",
        "kem-chong-muoi",
        "dau-xa",
        "gel-rua-tay",
        "khan-giay",
        "dung-dich-ve-sinh",
        "mieng-dan-mun",
        "mat-na",
        "sua-mat",
        "tay-te-bao-chet",
        "bong-tay-trang",
    }
)


class CrawlConfigurationError(ValueError):
    """Raised when crawl options are invalid."""


class CrawlFetchError(RuntimeError):
    """Raised when a page cannot be fetched."""


@dataclass(frozen=True)
class CrawlRequest:
    leaflets_dir: Path
    sitemap_url: str | None = None
    workers: int = 8
    request_timeout_seconds: float = 10.0
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class CrawlResult:
    total: int
    downloaded: int
    skipped: int
    failed: int
    manifest_path: Path


def parse_locations(payload: bytes) -> tuple[str, ...]:
    text = payload.decode("utf-8", errors="replace")
    return tuple(sorted(set(re.findall(r"<loc>\s*(.*?)\s*</loc>", text))))


def leaflet_page(url: str) -> tuple[str, str] | None:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) != 2:
        return None
    category, slug = parts
    if category in NON_DRUG_CATEGORIES:
        return None
    if not (category.startswith(DRUG_CATEGORY_PREFIX) or category in DRUG_CATEGORIES):
        return None
    return category, slug


def collect_page_urls(
    sitemap_url: str, fetch: Fetch, *, timeout: float, workers: int
) -> tuple[str, ...]:
    sub_sitemaps = parse_locations(fetch(sitemap_url, timeout))
    urls: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for payload in pool.map(lambda url: fetch(url, timeout), sub_sitemaps):
            urls.update(parse_locations(payload))
    return tuple(sorted(urls))


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def crawl_leaflets(request: CrawlRequest, fetch: Fetch, *, today: date) -> CrawlResult:
    if request.workers < 1:
        raise CrawlConfigurationError("workers must be >= 1")
    if request.request_timeout_seconds <= 0:
        raise CrawlConfigurationError("request timeout must be positive")
    manifest_path = request.leaflets_dir / "manifest.json"
    previous = read_leaflet_manifest(manifest_path) if manifest_path.is_file() else None
    sitemap_url = request.sitemap_url or (previous.sitemap_url if previous else None)
    if not sitemap_url:
        raise CrawlConfigurationError(
            "The first crawl needs --sitemap-url; later crawls read it from manifest.json"
        )
    if request.dry_run:
        return CrawlResult(0, 0, 0, 0, manifest_path)
    all_urls = collect_page_urls(
        sitemap_url,
        fetch,
        timeout=request.request_timeout_seconds,
        workers=request.workers,
    )
    pages = {url: page for url in all_urls if (page := leaflet_page(url)) is not None}
    urls_dir = request.leaflets_dir / "urls"
    _write_lines(urls_dir / "all_urls.txt", all_urls)
    drug_urls = sorted(pages)
    _write_lines(urls_dir / "drug_urls.txt", drug_urls)
    html_dir = request.leaflets_dir / "html"

    def target_for(url: str) -> Path:
        category, slug = pages[url]
        return html_dir / category / f"{slug}.html"

    pending: list[str] = []
    skipped = 0
    for url in drug_urls:
        if target_for(url).is_file() and not request.force:
            skipped += 1
        else:
            pending.append(url)
    downloaded = failed = 0
    with ThreadPoolExecutor(max_workers=request.workers) as pool:
        futures = {
            pool.submit(fetch, url, request.request_timeout_seconds): url
            for url in pending
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                payload = future.result()
            except CrawlFetchError:
                failed += 1
                continue
            target = target_for(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            downloaded += 1
    # Every HTML file on disk is listed, including pages whose URL has left the sitemap,
    # so `verify_leaflet_source` never meets a file the manifest does not know.
    url_by_path = {
        target_for(url).relative_to(html_dir).as_posix(): url for url in drug_urls
    }
    files = tuple(
        LeafletFile(
            path.relative_to(html_dir).as_posix(),
            path.stat().st_size,
            file_sha256(path),
            url_by_path.get(path.relative_to(html_dir).as_posix()),
        )
        for path in sorted(html_dir.rglob("*.html"))
    )
    write_leaflet_manifest(
        manifest_path,
        LeafletManifest(
            sitemap_url,
            today.isoformat(),
            file_sha256(urls_dir / "drug_urls.txt"),
            files,
        ),
    )
    return CrawlResult(len(drug_urls), downloaded, skipped, failed, manifest_path)
```

Reformat the `NON_DRUG_CATEGORIES` literal with `uv run ruff format` (one item per line).

Update `corpus/sources/__init__.py` imports and its sorted `__all__` to export `CrawlConfigurationError`, `CrawlFetchError`, `CrawlRequest`, `CrawlResult`, `crawl_leaflets`.

Add to `config/paths.py` after `DATA_DIR`:

```python
SOURCES_DIR = DATA_DIR / "sources"
LEAFLETS_DIR = SOURCES_DIR / "leaflets"
LEAFLETS_HTML_DIR = LEAFLETS_DIR / "html"
LEAFLETS_URLS_DIR = LEAFLETS_DIR / "urls"
LEAFLETS_MANIFEST_PATH = LEAFLETS_DIR / "manifest.json"
```

and delete `RESOURCES_ANKHANG_DIR` and `RAW_ANKHANG_HTML_DIR`. `git rm` `corpus/crawling/collect_urls.py` and `corpus/crawling/download_html.py`; delete `DEFAULT_HTML_DIR`, `DEFAULT_OUTPUT_DIR` and their imports from `corpus/crawling/parse_html.py`.

Replace the `crawl` command in `cli/commands/source.py` with:

```python
@source_app.command("crawl")
def crawl(
    ctx: typer.Context,
    leaflets_dir: Annotated[Path, typer.Option("--leaflets-dir")] = LEAFLETS_DIR,
    sitemap_url: Annotated[str | None, typer.Option("--sitemap-url")] = None,
    workers: Annotated[int, typer.Option("--workers")] = 8,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 10.0,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    def run() -> CommandResult:
        result = crawl_leaflets(
            CrawlRequest(
                leaflets_dir,
                sitemap_url,
                workers,
                request_timeout_seconds,
                force,
                dry_run,
            ),
            _fetch,
            today=date.today(),
        )
        return CommandResult(
            "source crawl",
            CommandStatus.COMPLETE,
            result.manifest_path,
            {
                "total": result.total,
                "downloaded": result.downloaded,
                "skipped": result.skipped,
                "failed": result.failed,
            },
        )

    run_handler(state_from_context(ctx), run)


def _fetch(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "seed-pipeline"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise CrawlFetchError(f"{url}: {exc}") from exc
```

with module imports `import http.client`, `from datetime import date`, `from urllib.request import Request, urlopen` and `LEAFLETS_DIR` added to the `seed_pipeline.config.paths` import (keep `RAW_DIR`, which `extract-tables` uses until Task 9), and replace line 20 (`from seed_pipeline.corpus.sources.crawl import CrawlRequest, crawl_source`) with `from seed_pipeline.corpus.sources.crawl import CrawlFetchError, CrawlRequest, crawl_leaflets`.

- [ ] **Step 5: Run the seed gate**

Run the **Seed gate**. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A seed-pipeline/src seed-pipeline/tests
git commit -m "feat(seed): crawl leaflets into an html tree with url lists and a manifest"
```

---

### Task 9: Build from verified leaflet HTML and the sources folder

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/corpus/sources/leaflet_source.py`, `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py`, `seed-pipeline/src/seed_pipeline/artifacts/contract.py`, `seed-pipeline/src/seed_pipeline/artifacts/paths.py`, `seed-pipeline/src/seed_pipeline/bundle/export.py` (`_source_digests`), `seed-pipeline/src/seed_pipeline/cli/commands/{build,bundle,source}.py`, `seed-pipeline/src/seed_pipeline/orchestration/preflight.py`, `seed-pipeline/src/seed_pipeline/corpus/processing/{clean_markdown_corpus,extract_pymupdf_text}.py`, `seed-pipeline/src/seed_pipeline/corpus/tables/extract_docling_tables.py`, `seed-pipeline/src/seed_pipeline/corpus/crawling/integrate_ankhang.py` (mappings default), `seed-pipeline/src/seed_pipeline/config/paths.py`
- Delete: `seed-pipeline/src/seed_pipeline/artifacts/snapshot.py`, `seed-pipeline/tests/orchestration/test_build_metadata.py`
- Test: `seed-pipeline/tests/corpus/test_leaflet_source.py`, `seed-pipeline/tests/artifacts/test_contract.py`, `seed-pipeline/tests/bundle/test_export.py`, `seed-pipeline/tests/fixtures/rag_final_small/manifest.json`, `seed-pipeline/tests/orchestration/test_preflight.py`, `seed-pipeline/tests/config/test_paths.py`, `seed-pipeline/tests/test_removed_contracts.py`

**Interfaces:**
- Produces (`leaflet_source`): `LeafletSource(html_dir: Path, manifest_sha256: str, file_count: int)`; `verify_leaflet_source(leaflets_dir: Path) -> LeafletSource`.
- Produces (`paths`): `FORMULARY_PDF_PATH = SOURCES_DIR / "duoc-thu-quoc-gia-viet-nam.pdf"`, `SOURCES_CURATION_DIR = SOURCES_DIR / "curation"`; removes `HEAVY_DATA_DIR`, `HEAVY_RAW_DIR`, `RAW_DIR`, `RAW_ANKHANG_DIR`, `RAW_ANKHANG_SNAPSHOTS_DIR`, `RAW_CURATION_DIR`, `RESOURCES_DIR`, `RESOURCES_CURATION_DIR`, `MANIFESTS_DIR`; `MIGRATION_DIR = DATA_DIR / "heavy" / "migration"` stays.
- Produces (`contract.build_manifest`): keyword `leaflet_source: Mapping[str, object]` replaces `snapshot_id` and `snapshot_sha256`; manifest key `leaflet_source` = `{"manifest_sha256": str, "file_count": int}`.
- Produces (`export._source_digests`): keys `source_pdf_sha256`, `leaflet_source_manifest_sha256`, `leaflet_source_file_count` (string value).
- Produces (`BuildConfig`): `pdf_path, leaflets_dir, curated_tables_path, table_overrides_path, mappings_path, glossary_path, work_root=BUILD_WORK_DIR, final_dir=RAG_FINAL_DIR, max_chars=DEFAULT_CHUNK_MAX_CHARS`; `seed build --leaflets-dir`.
- Removed: `ArtifactPaths.ankhang_html_dir`, `latest_snapshot_pair`, `seed build --snapshot-archive/--snapshot-manifest`.

- [ ] **Step 1: Write the failing tests**

In `seed-pipeline/tests/corpus/test_leaflet_source.py` replace the `leaflet_source` import block with:

```python
from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    LeafletSourceError,
    file_sha256,
    read_leaflet_manifest,
    verify_leaflet_source,
    write_leaflet_manifest,
)
```

and append:

```python
def write_tree(root: Path, files: dict[str, bytes]) -> LeafletManifest:
    entries = []
    for relative, payload in sorted(files.items()):
        path = root / "html" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(LeafletFile(relative, len(payload), file_sha256(path), None))
    manifest = LeafletManifest(
        "https://example.test/sitemap.xml", "2026-09-14", "c" * 64, tuple(entries)
    )
    write_leaflet_manifest(root / "manifest.json", manifest)
    return manifest


def test_verify_returns_the_manifest_digest_and_count(tmp_path: Path) -> None:
    write_tree(tmp_path, {"thuoc-a/one.html": b"one", "thuoc-b/two.html": b"two"})

    source = verify_leaflet_source(tmp_path)

    assert source.html_dir == tmp_path / "html"
    assert source.file_count == 2
    assert source.manifest_sha256 == file_sha256(tmp_path / "manifest.json")


def test_verify_rejects_a_changed_file(tmp_path: Path) -> None:
    write_tree(tmp_path, {"thuoc-a/one.html": b"one"})
    (tmp_path / "html" / "thuoc-a" / "one.html").write_bytes(b"ONE")

    with pytest.raises(LeafletSourceError, match=r"thuoc-a/one\.html"):
        verify_leaflet_source(tmp_path)


def test_verify_rejects_extra_or_missing_files(tmp_path: Path) -> None:
    write_tree(tmp_path, {"thuoc-a/one.html": b"one"})
    (tmp_path / "html" / "thuoc-a" / "extra.html").write_bytes(b"extra")

    with pytest.raises(LeafletSourceError, match="1 unexpected"):
        verify_leaflet_source(tmp_path)
```

In `tests/artifacts/test_contract.py` replace `snapshot_id="snapshot", snapshot_sha256="archive",` with `leaflet_source={"manifest_sha256": "leaflets", "file_count": 2},` and assert `manifest["leaflet_source"] == {"manifest_sha256": "leaflets", "file_count": 2}` and `"snapshot_id" not in manifest`.

In `tests/fixtures/rag_final_small/manifest.json` replace the `snapshot_id` and `snapshot_sha256` lines with `"leaflet_source": {"file_count": 2, "manifest_sha256": "2222222222222222222222222222222222222222222222222222222222222222"},`. In `tests/bundle/test_export.py` replace the two snapshot entries of the expected `source_digests` with `"leaflet_source_manifest_sha256": "2" * 64,` and `"leaflet_source_file_count": "2",`.

In `tests/config/test_paths.py` add:

```python
def test_build_inputs_live_in_sources() -> None:
    assert paths.SOURCES_DIR == paths.DATA_DIR / "sources"
    assert paths.FORMULARY_PDF_PATH == (
        paths.SOURCES_DIR / "duoc-thu-quoc-gia-viet-nam.pdf"
    )
    assert paths.LEAFLETS_DIR == paths.SOURCES_DIR / "leaflets"
    assert paths.LEAFLETS_MANIFEST_PATH == paths.LEAFLETS_DIR / "manifest.json"
    assert paths.SOURCES_CURATION_DIR == paths.SOURCES_DIR / "curation"
```

Append to `REMOVED_PATH_NAMES`: `"HEAVY_DATA_DIR", "HEAVY_RAW_DIR", "RAW_DIR", "RAW_ANKHANG_DIR", "RAW_ANKHANG_SNAPSHOTS_DIR", "RAW_CURATION_DIR", "RESOURCES_DIR", "RESOURCES_CURATION_DIR", "RESOURCES_ANKHANG_DIR", "RAW_ANKHANG_HTML_DIR", "MANIFESTS_DIR"`, and append `"seed_pipeline.artifacts.snapshot"`, `"seed_pipeline.corpus.crawling.collect_urls"`, `"seed_pipeline.corpus.crawling.download_html"` to `REMOVED_MODULES`.

In `tests/orchestration/test_preflight.py` change the expected resource to `"data/sources/duoc-thu-quoc-gia-viet-nam.pdf"` and the expected message fragment to `"restore data/sources"`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/corpus/test_leaflet_source.py tests/artifacts/test_contract.py tests/bundle/test_export.py tests/config/test_paths.py tests/orchestration/test_preflight.py tests/test_removed_contracts.py`
Expected: FAIL (`verify_leaflet_source` missing, `build_manifest` signature, digests, paths, preflight text, removed names).

- [ ] **Step 3: Add `verify_leaflet_source`**

Append to `leaflet_source.py`:

```python
@dataclass(frozen=True)
class LeafletSource:
    html_dir: Path
    manifest_sha256: str
    file_count: int


def verify_leaflet_source(leaflets_dir: Path) -> LeafletSource:
    leaflets_dir = Path(leaflets_dir)
    manifest_path = leaflets_dir / "manifest.json"
    if not manifest_path.is_file():
        raise LeafletSourceError(f"Leaflet manifest is missing: {manifest_path}")
    manifest = read_leaflet_manifest(manifest_path)
    html_dir = leaflets_dir / "html"
    present = (
        {path.relative_to(html_dir).as_posix() for path in html_dir.rglob("*.html")}
        if html_dir.is_dir()
        else set()
    )
    expected = {item.path for item in manifest.files}
    if present != expected:
        raise LeafletSourceError(
            f"Leaflet HTML under {html_dir} does not match {manifest_path}: "
            f"{len(expected - present)} missing, {len(present - expected)} unexpected"
        )
    for item in manifest.files:
        path = html_dir / item.path
        if path.stat().st_size != item.size or file_sha256(path) != item.sha256:
            raise LeafletSourceError(
                f"Leaflet file changed since the crawl: {item.path}"
            )
    return LeafletSource(html_dir, file_sha256(manifest_path), len(manifest.files))
```

- [ ] **Step 4: Build from sources**

`config/paths.py`: add `FORMULARY_PDF_PATH = SOURCES_DIR / "duoc-thu-quoc-gia-viet-nam.pdf"` and `SOURCES_CURATION_DIR = SOURCES_DIR / "curation"`; delete the whole "Build inputs stay in the pre-migration folders" block except `MIGRATION_DIR`, which becomes `MIGRATION_DIR = DATA_DIR / "heavy" / "migration"`.

`artifacts/contract.py`: replace the `snapshot_id: str, snapshot_sha256: str,` parameters of `build_manifest` with `leaflet_source: Mapping[str, object],` and the two manifest entries with `"leaflet_source": dict(leaflet_source),`; add `from collections.abc import Mapping`.

`artifacts/paths.py`: delete the `ankhang_html_dir` property and its entry in `directories()`.

`bundle/export.py`: replace `_source_digests` with:

```python
def _source_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    leaflet_source = manifest["leaflet_source"]
    digests = {
        "source_pdf_sha256": str(manifest["source_pdf_sha256"]),
        "leaflet_source_manifest_sha256": str(leaflet_source["manifest_sha256"]),
        "leaflet_source_file_count": str(leaflet_source["file_count"]),
    }
```

keeping the curated-input lines that follow in the current function.

`orchestration/build_corpus.py`:
- Imports: delete the `seed_pipeline.artifacts.snapshot` import; add `from seed_pipeline.corpus.sources.leaflet_source import verify_leaflet_source`; the paths import becomes `BUILD_WORK_DIR, FORMULARY_PDF_PATH, LEAFLETS_DIR, RAG_FINAL_DIR, SOURCES_CURATION_DIR, SOURCES_DIR`.
- `BuildConfig`: replace `snapshot_archive: Path` and `snapshot_manifest: Path` with `leaflets_dir: Path`.
- `run_build`: call `verify_leaflet_source(config.leaflets_dir)` first, before the build id is computed, because the id hashes only the leaflet manifest and the early "already built" return would otherwise accept HTML edited without a manifest change.
- `_digest_payload`: replace the two snapshot keys with `"leaflet_manifest": sha256_file(config.leaflets_dir / "manifest.json"),` and use `SOURCES_DIR / "vietnamese_valid_syllables.json"` for `valid_syllables`.
- `build_candidate`: replace `require_materialized_file(config.snapshot_archive)`, `require_materialized_file(config.snapshot_manifest)`, `verify_snapshot(...)` and `extract_snapshot(...)` with `leaflet_source = verify_leaflet_source(config.leaflets_dir)`; replace the required-file check for `RESOURCES_DIR / "vietnamese_valid_syllables.json"` (line 149) with `SOURCES_DIR / "vietnamese_valid_syllables.json"`; compute capacity with `3 * config.pdf_path.stat().st_size + sum(path.stat().st_size for path in leaflet_source.html_dir.rglob("*.html"))` (the HTML is read in place, no longer copied); call `parse_html_tree(leaflet_source.html_dir, paths.ankhang_markdown_dir)`; pass `leaflet_source={"manifest_sha256": leaflet_source.manifest_sha256, "file_count": leaflet_source.file_count}` to `build_manifest`.
- Delete `latest_snapshot_pair`. `default_config()` returns:

<!-- fmt: off -->
```python
    return BuildConfig(
        pdf_path=FORMULARY_PDF_PATH,
        leaflets_dir=LEAFLETS_DIR,
        curated_tables_path=SOURCES_CURATION_DIR / "docling_tables.jsonl",
        table_overrides_path=SOURCES_CURATION_DIR / "table_duplicate_overrides.json",
        mappings_path=SOURCES_DIR / "colloquial_mappings.json",
        glossary_path=SOURCES_DIR / "term_glossary.json",
    )
```
<!-- fmt: on -->

`cli/commands/build.py`: replace `--snapshot-archive`/`--snapshot-manifest` with `leaflets_dir: Annotated[Path | None, typer.Option("--leaflets-dir")] = None` and pass `leaflets_dir=(leaflets_dir if leaflets_dir is not None else config.leaflets_dir)`.

Replace remaining users of removed constants: `cli/commands/bundle.py` glossary/mappings defaults → `SOURCES_DIR / "term_glossary.json"` and `SOURCES_DIR / "colloquial_mappings.json"`; `cli/commands/source.py`, `corpus/processing/extract_pymupdf_text.py`, `corpus/tables/extract_docling_tables.py` PDF defaults → `FORMULARY_PDF_PATH`; `corpus/processing/clean_markdown_corpus.py` syllables default → `SOURCES_DIR / "vietnamese_valid_syllables.json"`; `corpus/crawling/integrate_ankhang.py` `MAPPINGS_PATH` → `SOURCES_DIR / "colloquial_mappings.json"` and delete its unused `SECTIONS_PATH`, `CHUNKS_PATH`, `AUDIT_PATH`, `MANIFEST_PATH`, `MARKDOWN_DIR` constants; its paths import becomes `from seed_pipeline.config.paths import SOURCES_DIR`.

`orchestration/preflight.py`: required inputs become `FORMULARY_PDF_PATH`, `LEAFLETS_MANIFEST_PATH`, `SOURCES_CURATION_DIR / "docling_tables.jsonl"`, `SOURCES_CURATION_DIR / "table_duplicate_overrides.json"`, and the three `SOURCES_DIR` JSON files (resolved against `project_root` the same way the current code resolves `HEAVY_RAW_DIR`); the message becomes `f"Missing {len(missing)} required source input(s); restore data/sources from the data archive"`.

`git rm seed-pipeline/src/seed_pipeline/artifacts/snapshot.py seed-pipeline/tests/orchestration/test_build_metadata.py`.

- [ ] **Step 5: Run the seed gate**

Run the **Seed gate**. Expected: PASS. Then `grep -rnwE 'HEAVY_DATA_DIR|HEAVY_RAW_DIR|RAW_DIR|RESOURCES_DIR|MANIFESTS_DIR|snapshot_archive|snapshot_manifest|snapshot_id' src tests`. Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add -A seed-pipeline/src seed-pipeline/tests
git commit -m "refactor(seed): build from verified leaflet HTML in data/sources"
```

---

### Task 10: Leaflet identifiers and neutral text in seed-pipeline

**Files:**
- Rename: `seed-pipeline/src/seed_pipeline/corpus/crawling/integrate_ankhang.py` → `integrate_leaflets.py`
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py`, `seed-pipeline/src/seed_pipeline/artifacts/paths.py`, `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py`, `seed-pipeline/src/seed_pipeline/corpus/crawling/parse_html.py`, `seed-pipeline/src/seed_pipeline/corpus/validation/validate_final_rag.py`, `seed-pipeline/src/seed_pipeline/bundle/export.py`, `seed-pipeline/src/seed_pipeline/evaluation/{section_eval_schema,build_section_retrieval_eval,patient_query_generation,metrics_service}.py`
- Test/fixtures: `seed-pipeline/tests/fixtures/rag_final_small/{sections,chunks}.jsonl`, `seed-pipeline/tests/bundle/test_export.py`, `seed-pipeline/tests/bundle/test_parity.py`, `seed-pipeline/tests/evaluation/test_metrics_service.py`, `seed-pipeline/tests/test_removed_contracts.py`

**Interfaces:**
- Produces: module `seed_pipeline.corpus.crawling.integrate_leaflets` with `integrate_leaflet_corpus`, `is_leaflet_record`, `separate_leaflet_heading_runs`, `product_names_from_title`, `resolve_colloquial_mapping`; section ids `f"leaflet:{category}:{slug}"`; `parse_html.parse_leaflet_html`; `paths.LEAFLET_MARKDOWN_INTERIM_DIR`; `ArtifactPaths.leaflet_markdown_dir`; `export.COLLECTION_TITLE = "Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng thuốc"`, `export.LEAFLET_SECTION_PREFIX = "leaflet:"`; validation category `leaflet`, metric `leaflet_markdown_file_count`; evaluation identifiers per Global Constraints.

- [ ] **Step 1: Write the failing expectations**

In `tests/bundle/test_export.py`:
- `PANADOL = "leaflet:thuoc:panadol-extra-gsk-150-vien-11440"`, `HAPACOL = "leaflet:thuoc:hapacol-250-dhg-11500"`.
- Expected document keys `"leaflet:thuoc:hapacol-250-dhg-11500"` and `"leaflet:thuoc:panadol-extra-gsk-150-vien-11440"`.
- Expected source `SourceInfo(title="Tờ hướng dẫn sử dụng", url=None)`.
- Add `assert COLLECTION_TITLE == "Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng thuốc"`.

In `tests/bundle/test_parity.py`: `HAPACOL = "leaflet:thuoc:hapacol-250-dhg-11500"`.

In `tests/evaluation/test_metrics_service.py`: the breakdown key `"ankhang"` becomes `"leaflet"` and the last assertion becomes `assert "| leaflet |" not in rendered`.

Rewrite the fixture ids: `sed -i 's/brand:ankhang:/leaflet:/g' tests/fixtures/rag_final_small/sections.jsonl tests/fixtures/rag_final_small/chunks.jsonl`.

Append `"seed_pipeline.corpus.crawling.integrate_ankhang"` to `REMOVED_MODULES`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/bundle tests/evaluation/test_metrics_service.py tests/test_removed_contracts.py`
Expected: FAIL (export still emits `leaflet:ankhang:` document keys, the old title and URL, and the label map uses `ankhang`).

- [ ] **Step 3: Rename identifiers**

Run (in `seed-pipeline/`):

```bash
git mv src/seed_pipeline/corpus/crawling/integrate_ankhang.py src/seed_pipeline/corpus/crawling/integrate_leaflets.py
FILES=$(grep -rliE 'ankhang|an khang' src)
sed -i -E \
  -e 's/crawling\.integrate_ankhang\b/crawling.integrate_leaflets/g' \
  -e 's/integrate_ankhang_corpus/integrate_leaflet_corpus/g' \
  -e 's/brand:ankhang:/leaflet:/g' \
  -e 's/leaflet:ankhang:/leaflet:/g' \
  -e 's/ankhang-alias-any-/leaflet-alias-any-/g' \
  -e 's/ankhang-alias-/leaflet-alias-/g' \
  -e 's/f"ankhang-\{/f"leaflet-{/g' \
  -e 's/ankhang_brand/leaflet_brand/g' \
  -e 's/ANKHANG_/LEAFLET_/g' \
  -e 's/ankhang_/leaflet_/g' \
  -e 's/_ankhang/_leaflet/g' \
  -e 's/"ankhang"/"leaflet"/g' \
  -e 's/ankhang-html/leaflet-html/g' \
  -e 's/ankhang-markdown/leaflet-markdown/g' \
  $FILES
```

Then edit the remaining text by hand until `grep -rniE 'ankhang|an khang' src tests --exclude=test_removed_contracts.py` prints nothing:
- `bundle/export.py`: `COLLECTION_TITLE = "Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng thuốc"`; delete the base URL constant; `LEAFLET_SECTION_PREFIX = "leaflet:"`; `document_for_section` leaflet branch becomes:

<!-- fmt: off -->
```python
    if section_key.startswith(LEAFLET_SECTION_PREFIX):
        _leaflet_path(section_key)
        return DocumentRecord(
            key=section_key,
            kind=DocumentKind.LEAFLET,
            title=title,
            source=SourceInfo(title=source_title),
        )
```
<!-- fmt: on -->

  and the malformed-key message becomes `f"Leaflet section key is malformed: {section_key}"`.
- `corpus/validation/validate_final_rag.py`: messages `"Non-leaflet chunk is missing source_block_id"`, `"...; some single-column leaflet layouts are intentionally flattened."`, `"Leaflet page is missing expected H1 heading"`, `"Leaflet text contains raw HTML-like markup"`, `"Fix the leaflet HTML-to-Markdown parser only after verifying the raw HTML source."`.
- `evaluation/patient_query_generation.py`: `"No leaflet product sections found in final RAG sections"`, `"No leaflet product sections produced patient query candidates"`, `f"leaflet patient query for {candidate.category}"`, `f"Generated non-leaflet patient row: {row['expected_section_id']}"`, `f"Could only generate {len(rows)} unique leaflet patient queries for target {target_count}"`.
- `evaluation/build_section_retrieval_eval.py`: `f"coverage leaflet subsection {category}"`, `"Leaflet eval coverage collapsed: only leaflet_product/leaflet_ingredient categories are present"`, `f"Leaflet eval coverage below threshold: {', '.join(missing_thresholds)}"`, `"Patient query expected_section_id must be a leaflet section (leaflet:*), got ..."`.
- Any remaining comment that names the website: rewrite to describe leaflet pages.

- [ ] **Step 4: Run the seed gate**

Run the **Seed gate**. Expected: PASS. Then `grep -rniE 'ankhang|an khang' src tests --exclude=test_removed_contracts.py`. Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add -A seed-pipeline/src seed-pipeline/tests
git commit -m "refactor(seed): name leaflet data after its content"
```

---

### Task 11: Neutral leaflet naming in the backend

**Files:**
- Modify: `backend/src/pharma_agent/domain/agent/prompts.py:22`, `backend/skills/brand-to-generic/SKILL.md:10`, `backend/src/pharma_agent/domain/corpus/bundle.py:101,118`
- Modify: `backend/tests/corpus_fixtures.py`, `backend/tests/corpus_rows.py`, `backend/tests/domain/corpus/test_enrichment.py`, `backend/tests/domain/corpus/test_bundle_io.py`, `backend/tests/domain/corpus/golden/chunking_v1.json`
- Regenerate: `backend/tests/fixtures/knowledge_bundle_small/`
- Test: `backend/tests/domain/test_prompts.py`

**Interfaces:** no public API change. Fixture constants become `LEAFLET = "leaflet:thuoc-giam-dau:panadol-extra"`, `LEAFLET_SECTION = "leaflet:thuoc-giam-dau:panadol-extra:cong-dung"`.

- [ ] **Step 1: Write the failing prompt test**

In `backend/tests/domain/test_prompts.py` add `ASSISTANT_ROLE,` to the existing `from pharma_agent.domain.agent.prompts import (...)` block, then append (the file already defines `all_prompt_text()`):

```python
def test_prompts_describe_the_leaflet_corpus_by_content() -> None:
    assert "tờ hướng dẫn sử dụng thuốc" in ASSISTANT_ROLE
    assert "an khang" not in all_prompt_text().casefold()
```

- [ ] **Step 2: Run the test to verify it fails**

Run (in `backend/`): `uv run pytest -q tests/domain/test_prompts.py`
Expected: FAIL.

- [ ] **Step 3: Update prompt, skill and docstring**

- `prompts.py:22`: `"Bạn là trợ lý tra cứu thuốc dựa trên Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng thuốc. "`
- `skills/brand-to-generic/SKILL.md:10`: `- Truy vấn đầu tiên theo tên biệt dược thường trả về tờ hướng dẫn sử dụng thuốc có kèm hoạt chất trong "gợi ý thuật ngữ".`
- `bundle.py:118`: ``` ``key`` is the curated leaflet slug from ``data/sources/colloquial_mappings.json`` ```
- `bundle.py:101`: `"""One row of ``data/sources/term_glossary.json``."""`

- [ ] **Step 4: Update fixtures and tests**

Run (in `backend/`):

```bash
sed -i -E -e 's/brand:ankhang:/leaflet:/g' -e 's/leaflet:ankhang:/leaflet:/g' \
  tests/corpus_fixtures.py tests/corpus_rows.py tests/domain/corpus/test_enrichment.py \
  tests/domain/corpus/test_bundle_io.py tests/domain/corpus/golden/chunking_v1.json
sed -i -E 's#"url": "https://www\.nhathuocankhang\.com[^"]*"#"url": null#g' tests/domain/corpus/golden/chunking_v1.json
```

Then by hand:
- `tests/corpus_fixtures.py`: the leaflet `SourceInfo(title="Tờ hướng dẫn sử dụng", url=None)`; `source_digests={"source_pdf_sha256": sha256_hex("fixture source pdf"), "leaflet_source_manifest_sha256": sha256_hex("fixture leaflet source")}`.
- `tests/corpus_rows.py:145-146`: `title="Tờ hướng dẫn sử dụng",` and `url=None,`.
- `tests/domain/corpus/test_bundle_io.py:48`: `url=None,` (keep the neutral key in the expected error message at line 332).
- Regenerate the committed fixture: `uv run python -m tests.corpus_fixtures`.
- Confirm: `grep -rniE 'ankhang|an khang' src skills tests` prints nothing.

- [ ] **Step 5: Run both gates**

Run the **Backend gate** (unit tests; the golden and fixture tests must pass). Run the **Seed gate** too, because `seed-pipeline/tests/bundle/test_embed.py` reads `backend/tests/fixtures/knowledge_bundle_small`.

- [ ] **Step 6: Commit**

```bash
git add -A backend/src backend/skills backend/tests
git commit -m "refactor(backend): describe leaflet data by content in prompts and fixtures"
```

---

### Task 12: `corpus gc` deletes orphaned documents and sections

**Files:**
- Modify: `backend/src/pharma_agent/domain/corpus/models.py` (`PurgeResult`), `backend/src/pharma_agent/infrastructure/persistence/postgres/corpus_repository.py` (`purge_retired`), `backend/tests/corpus_memory.py` (`InMemoryCorpusRepository.purge_retired`), `backend/src/pharma_agent/cli.py` (gc output), `backend/src/pharma_agent/domain/corpus/ports.py` (docstring)
- Modify: `backend/tests/corpus_fixtures.py` (add `with_renamed_documents`)
- Test: `backend/tests/application/test_release_service.py`, `backend/tests/infrastructure/test_corpus_integration.py`

**Interfaces:**
- Produces: `PurgeResult.sections_deleted: int = 0`, `PurgeResult.documents_deleted: int = 0`; `with_renamed_documents(bundle: KnowledgeBundle, suffix: str) -> KnowledgeBundle`.

- [ ] **Step 1: Add the fixture helper and the failing unit test**

Append to `backend/tests/corpus_fixtures.py`:

```python
def with_renamed_documents(bundle: KnowledgeBundle, suffix: str) -> KnowledgeBundle:
    """Copy of `bundle` whose document keys end with `suffix`; section keys follow them."""
    documents = {
        document.key: f"{document.key}{suffix}" for document in bundle.documents
    }
    sections = {
        section.key: documents[section.document_key]
        + section.key.removeprefix(section.document_key)
        for section in bundle.sections
    }
    return bundle.model_copy(
        update={
            "documents": [
                document.model_copy(update={"key": documents[document.key]})
                for document in bundle.documents
            ],
            "sections": [
                section.model_copy(
                    update={
                        "key": sections[section.key],
                        "document_key": documents[section.document_key],
                    }
                )
                for section in bundle.sections
            ],
            "colloquial_mappings": [
                mapping.model_copy(
                    update={
                        "section_keys": [sections[key] for key in mapping.section_keys]
                    }
                )
                for mapping in bundle.colloquial_mappings
            ],
        }
    )
```

Append to `backend/tests/application/test_release_service.py` (add `with_renamed_documents` to its `tests.corpus_fixtures` import):

```python
async def test_gc_deletes_documents_and_sections_no_release_uses() -> None:
    s = services()
    old = await s.publish_bundle(small_bundle())
    renamed = with_renamed_documents(small_bundle(), "-v2")
    await s.publish_bundle(renamed)

    report = await s.releases.gc("formulary", keep=1)

    repository = s.adapters.repository
    assert report.retired == [old.id]
    assert {document.key for document in repository.documents.values()} == {
        document.key for document in renamed.documents
    }
    assert {section.key for section in repository.sections.values()} == {
        section.key for section in renamed.sections
    }
    assert report.purge.documents_deleted == len(small_bundle().documents)
    assert report.purge.sections_deleted == len(small_bundle().sections)
```

- [ ] **Step 2: Run the test to verify it fails**

Run (in `backend/`): `uv run pytest -q tests/application/test_release_service.py -k documents_and_sections`
Expected: FAIL with `AssertionError` (the old document keys are still in the repository).

- [ ] **Step 3: Implement the domain field and the in-memory purge**

`domain/corpus/models.py` `PurgeResult`: add `sections_deleted: int = 0` and `documents_deleted: int = 0`.

In `tests/corpus_memory.py`, inside `purge_retired`, insert this after the loop that deletes orphan revisions and before `return PurgeResult(`:

<!-- fmt: off -->
```python
        live_sections = {revision.section_id for revision in self.revisions.values()} | {
            placement.section_id
            for placements in self.placements.values()
            for placement in placements
        }
        orphan_sections = [
            section_id
            for section_id, section in self.sections.items()
            if self.documents[section.document_id].collection_id == collection_id
            and section_id not in live_sections
        ]
        for section_id in orphan_sections:
            del self.sections[section_id]
        live_documents = {section.document_id for section in self.sections.values()}
        orphan_documents = [
            document_id
            for document_id, document in self.documents.items()
            if document.collection_id == collection_id and document_id not in live_documents
        ]
        for document_id in orphan_documents:
            del self.documents[document_id]
```
<!-- fmt: on -->

and add `sections_deleted=len(orphan_sections), documents_deleted=len(orphan_documents),` to the returned `PurgeResult`.

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest -q tests/application/test_release_service.py`
Expected: PASS.

- [ ] **Step 5: Write the failing integration test**

Append to `backend/tests/infrastructure/test_corpus_integration.py` (add `with_renamed_documents` to its `tests.corpus_fixtures` import):

```python
async def test_gc_deletes_documents_and_sections_of_retired_keys(stack: Stack) -> None:
    await stack.importer(small_bundle(), publish=True)
    renamed = with_renamed_documents(small_bundle(), "-v2")
    await stack.importer(renamed, publish=True)

    report = await stack.releases.gc("formulary", keep=1)

    async with stack.database.engine.connect() as connection:
        documents = await connection.execute(text("SELECT key FROM corpus.documents"))
        document_keys = set(documents.scalars())
        sections = await connection.execute(text("SELECT key FROM corpus.sections"))
        section_keys = set(sections.scalars())
    assert document_keys == {document.key for document in renamed.documents}
    assert section_keys == {section.key for section in renamed.sections}
    assert report.purge.documents_deleted == len(small_bundle().documents)
    assert report.purge.sections_deleted == len(small_bundle().sections)
```

Run: `uv run pytest -q -m integration tests/infrastructure/test_corpus_integration.py -k retired_keys`
Expected: FAIL (old documents and sections remain).

- [ ] **Step 6: Implement the Postgres purge**

In `corpus_repository.py` `purge_retired`, after `revisions_deleted, _ = await _delete_unreferenced(...)` and still inside `async with self._sessions.begin() as session:`, add:

<!-- fmt: off -->
```python
            collection_documents = select(DocumentTable.id).where(
                DocumentTable.collection_id == collection_id
            )
            sections_deleted = len(
                (
                    await session.execute(
                        delete(SectionTable)
                        .where(
                            SectionTable.document_id.in_(collection_documents),
                            ~exists().where(
                                SectionRevisionTable.section_id == SectionTable.id
                            ),
                            ~exists().where(ReleaseChunkTable.section_id == SectionTable.id),
                        )
                        .returning(SectionTable.id)
                    )
                ).all()
            )
            documents_deleted = len(
                (
                    await session.execute(
                        delete(DocumentTable)
                        .where(
                            DocumentTable.collection_id == collection_id,
                            ~exists().where(SectionTable.document_id == DocumentTable.id),
                        )
                        .returning(DocumentTable.id)
                    )
                ).all()
            )
```
<!-- fmt: on -->

and add `sections_deleted=sections_deleted, documents_deleted=documents_deleted,` to the returned `PurgeResult`.

Replace the `purge_retired` docstring in `domain/corpus/ports.py` (line 72) with:

<!-- fmt: off -->
```python
        """Spec C §8.5 steps 3-4 for every retired release of the collection.

        Then deletes the collection's sections that no section revision or release chunk
        references, and the documents left without sections.
        """
```
<!-- fmt: on -->

In `cli.py` replace the gc `typer.echo(...)` (lines 419–425) with:

<!-- fmt: off -->
```python
        typer.echo(
            f"retired {len(report.retired)} releases (keep {retained}) | points updated "
            f"{report.points_updated} deleted {report.points_deleted} | chunk versions "
            f"deleted {report.purge.chunk_versions_deleted} kept "
            f"{report.purge.chunk_versions_kept} | section revisions deleted "
            f"{report.purge.section_revisions_deleted} | sections deleted "
            f"{report.purge.sections_deleted} | documents deleted "
            f"{report.purge.documents_deleted}"
        )
```
<!-- fmt: on -->

The foreign keys make this safe: sections, revisions and release chunks cascade from their parents, and chunk versions and message citations restrict. The `WHERE` clauses only pick sections that have no revision and no release chunk, and documents that have no section, so no cascade fires and no restrict rule is hit.

- [ ] **Step 7: Run the backend gate with integration tests**

Run the **Backend gate** and `uv run pytest -q -m integration`. Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/src backend/tests
git commit -m "feat(corpus): let gc delete documents and sections no release uses"
```

---

### Task 13: Repository guards, documentation and full verification

**Files:**
- Modify: `seed-pipeline/tests/test_repository_data_policy.py`, `seed-pipeline/README.md`, `seed-pipeline/data/README.md`, `seed-pipeline/docs/guides/{cli-reference,downstream,workflow-local-kaggle,workflow-local-only}.md`, `README.md` (root), `backend/README.md`

**Interfaces:** none.

- [ ] **Step 1: Write the failing policy test**

Append to `seed-pipeline/tests/test_repository_data_policy.py`:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_TREES = (
    "backend/src",
    "backend/skills",
    "seed-pipeline/src",
    "frontend/app",
)


def test_production_code_does_not_name_the_leaflet_source_site() -> None:
    found = subprocess.run(
        [
            "git",
            "grep",
            "-I",
            "-i",
            "-l",
            "-e",
            "ankhang",
            "-e",
            "an khang",
            "--",
            *PRODUCTION_TREES,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    names = subprocess.run(
        ["git", "ls-files", "--", *PRODUCTION_TREES],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert found.stdout.splitlines() == []
    assert [name for name in names if "ankhang" in name.lower()] == []
```

- [ ] **Step 2: Run it**

Run (in `seed-pipeline/`): `uv run pytest -q tests/test_repository_data_policy.py`
Expected: PASS if Tasks 10–11 are complete (it only guards). If it fails, fix the listed files, then re-run.

- [ ] **Step 3: Update the documentation to the new layout and commands**

- `seed-pipeline/data/README.md`: rewrite to describe `sources/`, `corpus/`, `evaluation/`, `cache/`, `work/`, what Git tracks (spec §4.1) and that everything else lives in the Kaggle archive restored with `seed data pull` (added by plan `2026-09-14-seed-data-archive.md`).
- `seed-pipeline/README.md` "Luồng dữ liệu" and "Đầu ra": paths `data/sources/`, `data/corpus/rag-final/`, `data/corpus/formulary/`, `data/evaluation/gold/`, `data/evaluation/runs/<run>/`, `data/cache/...`; `seed source crawl --sitemap-url` on the first crawl; describe leaflet sources as "tờ hướng dẫn sử dụng thuốc" (the source site is named at lines 3, 22, 56 and 73; old paths are at lines 22–26 and 56–66).
- `docs/guides/cli-reference.md`: cache paths at lines 36, 46 and 79 become `data/cache/text_embeddings/` and `data/cache/query_embeddings/`, line 83 becomes `data/cache/kaggle_profiles/<workload>/<model>.json`, line 40 describes one tree `data/evaluation/runs/<run>/`; `seed metrics --force`; no `--variant`/`--output-dir`/`--candidates`; `seed rerank --dry-run` prints `missing_pairs`; `seed build --leaflets-dir`.
- `docs/guides/downstream.md`: document key `leaflet:<category>:<slug>` without a source URL; data artifact policy paths from spec §4.
- `docs/guides/workflow-local-kaggle.md` (lines 31–58, including the cache path at line 43) and `workflow-local-only.md` (lines 18–36): bundle path `data/corpus/formulary`, gold path `data/evaluation/gold`, caches under `data/cache/`.
- Do not add `uv run seed data ...` lines to `seed-pipeline/README.md` or `docs/guides/*.md` in this plan: `tests/test_docs.py::test_documented_seed_commands_parse` runs every documented `uv run seed` line with `--help`, and `seed data` arrives in the archive plan. `data/README.md` is not scanned.
- Root `README.md`: line 5 becomes "(Dược thư Quốc gia, tờ hướng dẫn sử dụng thuốc)"; `<bundle_dir>` at line 55 becomes `../seed-pipeline/data/corpus/formulary`.
- `backend/README.md`: `<bundle_dir>` at line 20 becomes `../seed-pipeline/data/corpus/formulary`.
- Do not edit `docs/guides/migration-2026-09.md` (its test still requires the old strings; the data migration plan deletes it).

- [ ] **Step 4: Full verification**

Run the **Seed gate**, the **Backend gate** with `uv run pytest -q -m integration`, and from the repository root `uv run pre-commit run --all-files`. Expected: all PASS. Then run from `seed-pipeline/`: `uv run seed --help`, `uv run seed metrics --help`, `uv run seed rerank --help`, `uv run seed source crawl --help`, `uv run seed build --help` and confirm the options match this plan.

- [ ] **Step 5: Commit**

```bash
git add seed-pipeline/tests/test_repository_data_policy.py seed-pipeline/README.md seed-pipeline/data/README.md seed-pipeline/docs/guides README.md backend/README.md
git commit -m "docs(seed): describe the new data layout and guard production naming"
```
