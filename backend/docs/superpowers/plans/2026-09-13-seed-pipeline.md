# Seed Pipeline Implementation Plan (Plan 4 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `corpus-pipeline` into the offline `seed-pipeline` tool: rename it, depend on the backend as a library, export the built corpus as a `knowledge-bundle/v1` directory, prove that the backend chunker reproduces the old `chunks.jsonl` exactly, pre-compute bundle embeddings on Kaggle or locally, evaluate retrieval through the backend `RetrievalService`, delete the old Qdrant/chunk contract code, and document the dev migration.

**Architecture:** `seed-pipeline` keeps every source-processing stage (PDF, cleaning, sections, Docling tables, An Khang) and gains a `seed_pipeline.bundle` package that maps the published canonical sections and blocks to the pinned backend bundle models (`pharma_agent.domain.corpus.bundle`) and runs the backend `chunk_section` for parity checks, embedding inputs and evaluation chunks. Embedding reuses the existing Kaggle orchestration with a new text-only input contract (`embedding_text_sha256`, `embedding_text`), so Kaggle kernels never import the backend. `seed retrieve` builds the backend retrieval stack with `build_retrieval_service(settings, embedder=...)` against dev Postgres and Qdrant, injecting an embedder that serves the query vectors cached by `seed embed queries`, and writes the same candidate artifacts that `seed rerank` and `seed metrics` already consume.

**Tech Stack:** Python 3.12, uv (`uv_build`), Typer, pydantic v2 (through the backend models), `pharma-agent` as an editable path dependency, `qdrant-client>=1.19,<2`, existing llama.cpp and Kaggle integrations, pytest + pytest-asyncio.

**Spec:** `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §3 (repo layout), §5 (bundle), §7.3 (parity), §10 (seed-pipeline), §11 (migration), §12 (contract row). Builds on P1 (`backend/docs/superpowers/plans/2026-09-13-corpus-domain.md`), P2 (`2026-09-13-corpus-store-import.md`) and P3 (`2026-09-13-corpus-retrieval.md`).

## Global Constraints

- The environment is development only. Postgres and Qdrant may be reset; no data backfill or backward compatibility is needed.
- Python projects: Python 3.12, uv, shared `ruff.toml`, strict `pyrefly.toml` with `unused-ignore = true`, pytest `filterwarnings = ["error"]`. Lint and type errors are fixed in code; never add rule ignores, `# noqa`, `# type: ignore` or new `# pyrefly: ignore`.
- Every seed-pipeline task ends green on (run inside `seed-pipeline/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`. A task that edits backend files also runs the backend command: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q` inside `backend/`.
- Backend layering stays enforced by `backend/tests/architecture/test_layering.py`. The backend never imports `seed_pipeline`.
- Prefer established libraries over custom code. Do not add a feature flag or "fake mode" to production code for tests; fakes live under `tests/`.
- Commits: one commit per task, conventional message, ending with the session attribution trailer given in the executing session.
- Historical backend specs and plans (`backend/docs/superpowers/specs/*`, `backend/docs/superpowers/plans/*`) keep the old names; never edit them.
- Names used by this plan (exact values):
  - Folder `seed-pipeline/`, `[project].name = "seed-pipeline"`, package `seed_pipeline`, console script `seed = "seed_pipeline.cli.app:main"`, workspace env var `SEED_PIPELINE_ROOT`.
  - Kaggle dataset slug `BUNDLE_INPUT_DATASET_SLUG = "seed-pipeline-bundle"`; workspace prefixes `seed-pipeline-kaggle-` and `seed-pipeline-stage-`; pre-commit hook ids `pyrefly-seed-pipeline` and `uv-lock (seed-pipeline)`.
  - Backend names used exactly as pinned in `backend/docs/superpowers/plans/2026-09-13-plans-overview.md` §3: `BUNDLE_SCHEMA_VERSION`, `DocumentKind`, `BlockKind`, `RetrievalMode`, `SourceInfo`, `DocumentRecord`, `BlockRecord`, `SectionRecord`, `GlossaryEntry`, `ColloquialMappingRecord`, `BundleCollection`, `BundleGenerator`, `BundleManifest`, `KnowledgeBundle`, `BundleValidationError`, `model_slug`, `encode_vector`, `read_bundle`, `write_bundle`, `sha256_hex`, `compose_embedding_text`, `hydrate_strategy_for`, `CHUNKER_VERSION`, `MAX_CHUNK_CHARS`, `ChunkDraft`, `chunk_section`, `build_retrieval_service(settings, *, database=None, embedder=None)`, `RetrievalStack`, `RetrievalSettings.mode` (`hybrid`, `dense`, `bm25`), `RetrievalSettings.qdrant_collection`, `RetrievalSettings.collections`, `Hit` fields of §3.4 (including `embedding_text`). `pharma_agent.domain.corpus.hydrate` is public API (P1 `PUBLIC_API`).
  - Collection key `formulary`, collection title `Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng An Khang`.
  - Document keys: `drug:<slugify(title)>` for `drug_monograph`, `general:<slugify(title)>` for `general_monograph`, `leaflet:ankhang:<category>:<slug>` for An Khang (`source.url = https://www.nhathuocankhang.com/<category>/<slug>`). Section keys are the old section ids unchanged.
  - Published build contract `rag-final-v3` (after Task 10) = `sections.jsonl`, `blocks.jsonl`, `manifest.json`, `validation_report.json`.
  - Gold chunk label used by evaluation artifacts: `f"{section_key}:chunk-{ordinal:03d}"` (equals the old `chunk_id` when parity holds).
  - Text embedding cache schema `text-embedding-v1`, cache file `data/heavy/cache/text_embeddings/<catalog slug>.jsonl`, Kaggle input file `data/heavy/.work/bundle-embed/<catalog slug>/embedding_inputs.jsonl`, Kaggle stage `corpus-embed` contract version 3, artifact file `text_embeddings.jsonl`.
  - Parity normalizations (the only differences allowed between old chunks and `chunk_section` output): old page `0` equals `None`; an old leaflet `colloquial_mapping` without `key` is compared without the key the bundle gives it (the leaflet slug, P1 decision 1). IDs are not compared.
  - Migration acceptance: Hit@10 and MRR of the new hybrid + rerank evaluation drop at most 1 percentage point (MRR: at most 0.01) versus run `hybrid-qwen4b-p50-k30-rrf2` with the same reranker.
  - Colloquial mapping records (P1): `colloquial_mappings.json` is a JSON array with one record per curated slug (kept even when no leaflet resolves to it) and one record per uncurated leaflet keyed by the leaflet slug; `seed bundle export` itself guarantees unique keys (one record per curated slug, `product_names` of every leaflet resolved to it merged; P2's primary key `(release_id, position)` does not enforce uniqueness), asserted by `test_export_copies_glossary_and_attaches_mappings`; a section key appears in at most one record.
  - Leaflet blocks (P1 decision 4): the cleaned An Khang text split on blank lines, consecutive paragraphs joined with `"\n\n"` into one `prose` block, and every pipe-table paragraph longer than 3 000 characters as its own `table` block.
  - Evaluation never embeds queries through an endpoint: `dense` and `hybrid` read the vectors written by `seed embed queries` through `CachedQueryEmbedder`; `bm25` needs none. Candidate `document_text` is `Hit.embedding_text`, the text the old benchmark and the backend reranker score.

---

## File Structure

```text
thesis/
  README.md                                        project table row renamed                                   # Task 1
  .pre-commit-config.yaml                          seed-pipeline hook ids and paths                            # Task 1
  backend/README.md, backend/.env.example, backend/src (docstrings)   corpus-pipeline mentions renamed        # Task 1
  seed-pipeline/                                   git mv from corpus-pipeline/                                # Task 1
    pyproject.toml                                 name, script (T1); pharma-agent path dep, qdrant range (T2); data marker (T5)
    uv.lock                                        relocked                                                    # Task 1, 2
    .gitignore                                     seed-pipeline-kaggle-*/                                     # Task 1
    README.md, data/README.md, docs/guides/{downstream,cli-reference,workflow-local-kaggle,workflow-local-only}.md
                                                   renamed commands (T1), rewritten around the bundle (T10)
    docs/guides/migration-2026-09.md               dev migration runbook, spec §11                             # Task 11
    src/seed_pipeline/                             git mv from src/corpus_pipeline/                            # Task 1
      cli/app.py                                   Typer app `seed`; `bundle` group (T4); no `vectors` (T10)
      cli/runtime.py                               reattach hint `uv run seed`                                 # Task 1
      cli/commands/bundle.py                       `seed bundle export` (T4), `parity` (T5), `embed` (T7)
      cli/commands/embed.py                        only `embed queries` remains                                # Task 6
      cli/commands/evaluation.py                   `evaluation build --bundle`                                 # Task 8
      cli/commands/retrieve.py                     backend-based `seed retrieve`                               # Task 9
      config/paths.py                              SEED_PIPELINE_ROOT (T1); bundle/migration dirs (T5); embedding and
                                                   compose paths (T6); EVALUATION_CHUNKS_PATH (T8); BACKEND_ENV_FILE (T9);
                                                   old chunk paths removed (T10)
      config/defaults.py                           DEFAULT_QDRANT_URL removed                                  # Task 10
      artifacts/contract.py                        blocks.jsonl in the contract (T3); rag-final-v3 (T10)
      orchestration/build_corpus.py                publish blocks (T3); no unified chunks (T10)
      orchestration/validation_service.py          contract validation only                                    # Task 10
      corpus/validation/validate_final_rag.py      validate_unified_chunks removed                             # Task 10
      corpus/canonical/build_canonical_rag.py, corpus/tables/curate_docling_tables.py,
      corpus/processing/clean_markdown_corpus.py, integrations/kaggle/model_artifacts.py   rename damage fixed    # Task 1
      bundle/__init__.py, bundle/chunks.py         backend chunk_section over a bundle, gold labels            # Task 4
      bundle/export.py                             rag-final -> knowledge-bundle/v1                            # Task 4, 7
      bundle/parity.py                             old chunks.jsonl vs chunk_section                           # Task 5
      bundle/io.py                                 validated staging write                                     # Task 7
      bundle/embed.py                              embedding inputs, embeddings written into the bundle        # Task 7
      bundle/evaluation_chunks.py                  evaluation chunk rows                                       # Task 8
      embeddings/__init__.py, embeddings/text_cache.py   worker-safe text embedding cache and embed loop       # Task 6
      embeddings/service.py                        local and Kaggle text embedding backends                    # Task 6
      integrations/kaggle/dependencies.py          BUNDLE_INPUT_DATASET_SLUG                                   # Task 1
      integrations/kaggle/workspace.py             seed-pipeline prefixes                                      # Task 1
      integrations/kaggle/stages.py, kernels.py, checkpoints.py, workers/corpus_embed.py   text contract v3   # Task 6
      integrations/kaggle/service.py               GGUF_ROOT from config.paths                                 # Task 10
      evaluation/backend_retrieval.py              retrieval through RetrievalService                          # Task 9
      evaluation/cached_query_embedder.py          query vectors from the `seed embed queries` cache           # Task 9
      evaluation/candidate_text.py                 candidate_document_text moved out of retrievers.py          # Task 9
      evaluation/retrieval_candidate_artifact.py, rerankers.py, rerank_score_cache.py   new text helper import # Task 9
      evaluation/run_workspace.py                  RunIdentity.release_id, chunker_version                     # Task 9
      evaluation/patient_query_generation.py, build_section_retrieval_eval.py   default chunks path            # Task 8
      evaluation/query_embedding_service.py, rerank_service.py, preload_query_embeddings.py   compose paths    # Task 10
      (deleted) vector_store/embedding_service.py                                                              # Task 6
      (deleted) evaluation/retrievers.py, evaluation/retrieval_service.py                                      # Task 9
      (deleted) corpus/metadata/{build_rag_metadata,qdrant_payload_contract,term_enrichment}.py, vector_store/,
                integrations/postgres/, cli/commands/vectors.py                                                # Task 10
    tests/
      test_project_identity.py                     rename guard                                                # Task 1
      test_dependency_pins.py, test_backend_public_api.py, integrations/kaggle/test_worker_bundle_imports.py   # Task 2
      artifacts/test_contract.py                   published contract                                          # Task 3, 10
      fixtures/rag_final_small/                    tiny build produced by the pre-migration code               # Task 4
      bundle/test_export.py, cli/test_bundle_command.py   export (T4), parity CLI (T5), embed CLI (T7)
      bundle/test_parity.py, bundle/test_parity_data.py   parity unit and `data`-marked real check            # Task 5
      embeddings/test_text_cache.py, integrations/kaggle/test_stages.py, cli/test_embed_command.py             # Task 6
      bundle/test_embed.py                         bundle embed, shared backend fixture bundle                 # Task 7
      bundle/test_evaluation_chunks.py, cli/test_evaluation_command.py                                         # Task 8
      evaluation/test_backend_retrieval.py, cli/test_retrieve_command.py                                       # Task 9
      test_removed_contracts.py                    removed modules and commands                                # Task 10
      test_docs.py                                 documented commands parse, runbook steps                    # Task 11
      (deleted) vector_store/test_embedding_service.py (T6); evaluation/test_retrievers.py,
                evaluation/test_retrieval_service.py (T9); vector_store/ (T10)
```

---

### Task 1: Rename `corpus-pipeline` to `seed-pipeline`

**Files:**
- Move: `corpus-pipeline/` → `seed-pipeline/`, `seed-pipeline/src/corpus_pipeline/` → `seed-pipeline/src/seed_pipeline/`
- Create: `seed-pipeline/tests/test_project_identity.py`
- Modify (mechanical rename): every `*.py`, `*.toml`, `*.md`, `.gitignore` under `seed-pipeline/` except `data/` (only `data/README.md`)
- Modify: `seed-pipeline/pyproject.toml` (`[project].name`, `[project.scripts]`), `seed-pipeline/src/seed_pipeline/cli/app.py:18-22,36`, `seed-pipeline/src/seed_pipeline/cli/runtime.py:73`
- Modify (restore strings damaged by an earlier mass rename): `seed-pipeline/src/seed_pipeline/corpus/canonical/build_canonical_rag.py:29`, `seed-pipeline/src/seed_pipeline/corpus/tables/curate_docling_tables.py:118`, `seed-pipeline/src/seed_pipeline/corpus/processing/clean_markdown_corpus.py:14`, `seed-pipeline/src/seed_pipeline/integrations/kaggle/model_artifacts.py:65`
- Modify: `README.md` (project table), `.pre-commit-config.yaml`, `backend/README.md`, `backend/.env.example`, and any remaining `corpus-pipeline`/`corpus_pipeline` mention in `backend/src`, `backend/tests`, `docker/`, `docker-compose.yml`
- Relock: `seed-pipeline/uv.lock`

**Interfaces:**
- Consumes: nothing new.
- Produces: console script `seed`; Typer app `seed_pipeline.cli.app.app` named `seed`; `seed_pipeline.config.paths.WORKSPACE_ROOT_ENV == "SEED_PIPELINE_ROOT"`; `seed_pipeline.integrations.kaggle.dependencies.BUNDLE_INPUT_DATASET_SLUG == "seed-pipeline-bundle"` (replaces `CORPUS_INPUT_DATASET_SLUG`); `TemporaryWorkspace.PREFIX == "seed-pipeline-kaggle-"`; `managed_staging_directory(parent=None, *, prefix="seed-pipeline-stage-")`.

- [ ] **Step 1: Write the failing identity test (in the old folder, before moving)**

Create `corpus-pipeline/tests/test_project_identity.py`:

```python
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import PROJECT_ROOT, WORKSPACE_ROOT_ENV
from seed_pipeline.integrations.kaggle.dependencies import BUNDLE_INPUT_DATASET_SLUG
from seed_pipeline.integrations.kaggle.workspace import (
    TemporaryWorkspace,
    managed_staging_directory,
)


def test_project_metadata_uses_seed_pipeline_names() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))

    assert PROJECT_ROOT.name == "seed-pipeline"
    assert config["project"]["name"] == "seed-pipeline"
    assert config["project"]["scripts"] == {"seed": "seed_pipeline.cli.app:main"}
    assert WORKSPACE_ROOT_ENV == "SEED_PIPELINE_ROOT"


def test_cli_program_is_named_seed() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage: seed" in result.output


def test_kaggle_names_use_seed_pipeline(tmp_path: Path) -> None:
    assert BUNDLE_INPUT_DATASET_SLUG == "seed-pipeline-bundle"
    assert TemporaryWorkspace.PREFIX == "seed-pipeline-kaggle-"
    with managed_staging_directory(tmp_path) as staging:
        assert staging.name.startswith("seed-pipeline-stage-")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd /home/andv/personal/thesis/corpus-pipeline && uv run pytest -q tests/test_project_identity.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'seed_pipeline'`.

- [ ] **Step 3: Move the folder and package, then rename mechanically**

```bash
cd /home/andv/personal/thesis
git mv corpus-pipeline seed-pipeline
git mv seed-pipeline/src/corpus_pipeline seed-pipeline/src/seed_pipeline
rm -rf seed-pipeline/.venv seed-pipeline/.pytest_cache seed-pipeline/.ruff_cache
cd seed-pipeline
grep -rlZ -e corpus_pipeline -e corpus-pipeline -e CORPUS_PIPELINE_ROOT -e CORPUS_INPUT_DATASET_SLUG \
  src tests docs README.md data/README.md pyproject.toml .gitignore \
  | xargs -0 sed -i \
      -e 's/corpus-pipeline-rag-final/seed-pipeline-bundle/g' \
      -e 's/CORPUS_INPUT_DATASET_SLUG/BUNDLE_INPUT_DATASET_SLUG/g' \
      -e 's/CORPUS_PIPELINE_ROOT/SEED_PIPELINE_ROOT/g' \
      -e 's/corpus_pipeline/seed_pipeline/g' \
      -e 's/corpus-pipeline/seed-pipeline/g'
sed -i -E \
  -e 's/uv run corpus\b/uv run seed/g' \
  -e 's/(^|[`( ])corpus (doctor|build|validate|evaluation|embed|vectors|retrieve|rerank|metrics|source)\b/\1seed \2/g' \
  README.md docs/guides/*.md
sed -i 's/^corpus = "seed_pipeline.cli.app:main"$/seed = "seed_pipeline.cli.app:main"/' pyproject.toml
cd /home/andv/personal/thesis
grep -rlZ -e corpus-pipeline -e corpus_pipeline \
  README.md .pre-commit-config.yaml backend/README.md backend/.env.example backend/src backend/tests docker docker-compose.yml \
  | xargs -0 sed -i -e 's/corpus_pipeline/seed_pipeline/g' -e 's/corpus-pipeline/seed-pipeline/g'
```

`data/retrieval_eval/**` is not edited: those run records are historical evaluation artifacts.

- [ ] **Step 4: Apply the non-mechanical edits**

`seed-pipeline/src/seed_pipeline/cli/app.py` — the Typer app and callback docstring:

```python
app = typer.Typer(
    name="seed",
    help="Build the seed corpus, export knowledge bundles, embed and evaluate retrieval.",
    no_args_is_help=True,
    add_completion=False,
)
```

```python
    """Operate the seed pipeline by explicit stage."""
```

`seed-pipeline/src/seed_pipeline/cli/runtime.py` inside `run_handler`:

```python
        command = shlex.join(["uv", "run", "seed", *sys.argv[1:]])
```

Restore the four strings that an earlier bulk rename had turned into module paths:

`seed-pipeline/src/seed_pipeline/corpus/canonical/build_canonical_rag.py:29`

```python
DEFAULT_TABLES = DOCLING_INTERIM_DIR / "tables.curated.jsonl"
```

`seed-pipeline/src/seed_pipeline/corpus/tables/curate_docling_tables.py:118`

```python
        for item in read_jsonl(input_dir / "tables.jsonl")
```

`seed-pipeline/src/seed_pipeline/corpus/processing/clean_markdown_corpus.py:14`

```python
# Load Vietnamese syllables database for spacing validation
```

`seed-pipeline/src/seed_pipeline/integrations/kaggle/model_artifacts.py:65`

```python
            f"GGUF source changed after validation: {source}"
```

`README.md` (repo root), replace the first table row with:

```markdown
| `seed-pipeline/` | Công cụ offline: nguồn nội bộ (Dược thư Quốc gia, An Khang) → knowledge bundle, embedding trên Kaggle, đánh giá retrieval |
```

Check that nothing outside history still uses the old names:

```bash
cd /home/andv/personal/thesis
grep -rn -e corpus-pipeline -e corpus_pipeline -e CORPUS_PIPELINE --exclude-dir=.venv --exclude-dir=data --exclude=uv.lock \
  seed-pipeline README.md .pre-commit-config.yaml backend/README.md backend/.env.example backend/src backend/tests docker docker-compose.yml
```

Expected: no output.

- [ ] **Step 5: Relock, sync and run the identity test**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv lock
uv sync
uv run pytest -q tests/test_project_identity.py
```

Expected: `3 passed`.

- [ ] **Step 6: Run the full checks (seed-pipeline, and backend because its docs and docstring changed)**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q
cd ../backend
uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q
cd ..
uv run --project backend pre-commit run --all-files
```

Expected: all green; pre-commit runs `pyrefly check (seed-pipeline)` and `uv-lock (seed-pipeline)`.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add -A seed-pipeline README.md .pre-commit-config.yaml backend/README.md backend/.env.example backend/src backend/tests docker docker-compose.yml
git commit -m "refactor(seed): rename corpus-pipeline to seed-pipeline"
```

The commit message ends with the session attribution trailer.

---

### Task 2: Depend on the backend and align `qdrant-client`

**Files:**
- Modify: `seed-pipeline/pyproject.toml` (`[project].dependencies`, `[tool.uv.sources]`)
- Modify: `seed-pipeline/tests/test_dependency_pins.py` (whole file)
- Modify: `seed-pipeline/tests/integrations/kaggle/test_worker_bundle_imports.py:37-45` (worker must not import the backend)
- Create: `seed-pipeline/tests/test_backend_public_api.py`
- Relock: `seed-pipeline/uv.lock`

**Interfaces:**
- Consumes (P1, P3, pinned): `pharma_agent.domain.corpus.bundle` (`BUNDLE_SCHEMA_VERSION`, `model_slug`), `pharma_agent.domain.corpus.identity`, `pharma_agent.domain.corpus.chunking` (`CHUNKER_VERSION`, `MAX_CHUNK_CHARS`, `chunk_section`), `pharma_agent.domain.corpus.enrichment`, `pharma_agent.domain.corpus.hydrate`, `pharma_agent.infrastructure.composition` (`build_retrieval_service(settings, *, database=None, embedder=None)`, `RetrievalStack`).
- Produces: `pharma-agent` installed editable into the seed-pipeline environment; the rule "a direct dependency is either exactly pinned, a local path source, or uses the exact specifier the backend declares".

- [ ] **Step 1: Write the failing tests**

Replace `seed-pipeline/tests/test_dependency_pins.py`:

```python
import re
import tomllib
from pathlib import Path

EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==[^,;<>=!~]+$")
REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9_.-]+")


def _load(path: str) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    assert match is not None, requirement
    return match.group(0).lower()


def test_direct_dependencies_are_pinned_local_or_shared_with_backend() -> None:
    config = _load("pyproject.toml")
    backend = _load("../backend/pyproject.toml")
    path_sources = {
        name
        for name, source in config["tool"]["uv"]["sources"].items()
        if "path" in source
    }
    backend_specifiers = {
        _name(item): item for item in backend["project"]["dependencies"]
    }
    requirements = list(config["project"]["dependencies"])
    for group in config.get("dependency-groups", {}).values():
        requirements.extend(group)

    assert requirements
    for item in requirements:
        name = _name(item)
        if name in path_sources:
            assert item == name, item
        elif EXACT_REQUIREMENT.fullmatch(item) is None:
            assert backend_specifiers.get(name) == item, item


def test_backend_is_an_editable_path_dependency() -> None:
    config = _load("pyproject.toml")

    assert "pharma-agent" in config["project"]["dependencies"]
    assert "qdrant-client>=1.19,<2" in config["project"]["dependencies"]
    assert config["tool"]["uv"]["sources"]["pharma-agent"] == {
        "path": "../backend",
        "editable": True,
    }
```

Create `seed-pipeline/tests/test_backend_public_api.py`:

```python
import importlib
import inspect

import pytest
from pharma_agent.domain.corpus.bundle import BUNDLE_SCHEMA_VERSION, model_slug
from pharma_agent.domain.corpus.chunking import (
    CHUNKER_VERSION,
    MAX_CHUNK_CHARS,
    chunk_section,
)
from pharma_agent.infrastructure.composition import (
    RetrievalStack,
    build_retrieval_service,
)

PUBLIC_MODULES = (
    "pharma_agent.domain.corpus.bundle",
    "pharma_agent.domain.corpus.identity",
    "pharma_agent.domain.corpus.chunking",
    "pharma_agent.domain.corpus.enrichment",
    "pharma_agent.domain.corpus.hydrate",
    "pharma_agent.infrastructure.composition",
)


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_backend_public_module_imports(module: str) -> None:
    assert importlib.import_module(module).__name__ == module


def test_pinned_backend_names_have_expected_values() -> None:
    assert BUNDLE_SCHEMA_VERSION == "knowledge-bundle/v1"
    assert CHUNKER_VERSION == "chunker-v1"
    assert MAX_CHUNK_CHARS == 3000
    assert model_slug("qwen3-embedding:4b-fp16") == "qwen3_embedding_4b_fp16"
    assert callable(chunk_section)
    assert "embedder" in inspect.signature(build_retrieval_service).parameters
    assert RetrievalStack.__name__ == "RetrievalStack"
```

In `seed-pipeline/tests/integrations/kaggle/test_worker_bundle_imports.py`, replace the `code = (...)` expression so a worker never pulls in the backend (Kaggle kernels only receive the zipped `seed_pipeline` sources):

```python
    code = (
        "import importlib, sys; "
        f"sys.path.insert(0, {str(bundle)!r}); "
        f"module = importlib.import_module({worker_module!r}); "
        f"assert str(module.__file__).startswith({str(bundle)!r}); "
        "assert 'seed_pipeline.config.paths' not in sys.modules; "
        "assert 'pharma_agent' not in sys.modules"
    )
```

- [ ] **Step 2: Run them and confirm the failures**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/test_dependency_pins.py tests/test_backend_public_api.py`
Expected: `test_backend_is_an_editable_path_dependency` FAILS with `KeyError: 'pharma-agent'` (or the `in` assertion), and `tests/test_backend_public_api.py` errors at collection with `ModuleNotFoundError: No module named 'pharma_agent'`.

- [ ] **Step 3: Declare the dependencies**

In `seed-pipeline/pyproject.toml`, replace `"qdrant-client==1.18.0",` with `"qdrant-client>=1.19,<2",` and add `"pharma-agent",` right after `"openpyxl==3.1.5",`. Replace the `[tool.uv.sources]` table with:

```toml
[tool.uv.sources]
pharma-agent = { path = "../backend", editable = true }
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

- [ ] **Step 4: Lock, sync and verify there is no conflict**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv lock
uv sync
uv run python -c "import pharma_agent, qdrant_client.version; print(pharma_agent.__file__)"
uv tree --depth 1 --package seed-pipeline | grep -E "pharma-agent|qdrant-client|typer"
```

Expected: `uv lock` resolves without conflicts (a trial lock on 2026-09-13 reported `Added pharma-agent v0.1.0` and `Updated qdrant-client v1.18.0 -> v1.19.0`, with `typer 0.21.2` satisfying both `==0.21.2` and the backend's `>=0.21`); the Python command prints `/home/andv/personal/thesis/backend/src/pharma_agent/__init__.py`.

- [ ] **Step 5: Run the tests and pyrefly (which must resolve `pharma_agent` through the editable install)**

```bash
uv run pytest -q tests/test_dependency_pins.py tests/test_backend_public_api.py tests/integrations/kaggle/test_worker_bundle_imports.py
uv run pyrefly check --min-severity warn
```

Expected: all tests pass; pyrefly reports `0 errors` (it reads the interpreter search path of `.venv`, which includes the editable `.pth` entry for `backend/src`; an unresolved import would show up as `Cannot find module pharma_agent...` on `tests/test_backend_public_api.py`).

- [ ] **Step 6: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/pyproject.toml seed-pipeline/uv.lock seed-pipeline/tests/test_dependency_pins.py seed-pipeline/tests/test_backend_public_api.py seed-pipeline/tests/integrations/kaggle/test_worker_bundle_imports.py
git commit -m "build(seed): depend on pharma-agent and align qdrant-client"
```

The commit message ends with the session attribution trailer.

---

### Task 3: Publish canonical blocks with the corpus build

`seed bundle export` needs the canonical blocks (paragraph and Docling table blocks per section). Today they only exist in the build workspace, which is deleted after publish. This task adds `blocks.jsonl` to the published `rag-final/` contract. An existing `rag-final/` without `blocks.jsonl` fails `validate_contract_directory`, so the next `seed build` rebuilds it.

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/artifacts/contract.py:9-11` (`CONTRACT_FILES`), `:68-102` (`build_manifest`), `:105-152` (`validate_contract_directory`)
- Modify: `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py:218-221` (copy `blocks.jsonl` next to `sections.jsonl`)
- Create: `seed-pipeline/tests/artifacts/test_contract.py`

**Interfaces:**
- Consumes: `canonical/blocks.jsonl` written by `process_canonical_rag` (records of `CanonicalBlock`: `block_id, section_index, section_id, title, section, content_type, source_type, page_start, page_end, text, markdown, source_refs, quality_flags, table_id`).
- Produces: `rag-final/blocks.jsonl`; manifest key `block_count`; `CONTRACT_FILES == {"sections.jsonl", "blocks.jsonl", "chunks.jsonl", "manifest.json", "validation_report.json"}`; `validate_contract_directory` rejects missing, duplicate or orphan blocks.

- [ ] **Step 1: Write the failing test**

Create `seed-pipeline/tests/artifacts/test_contract.py`:

```python
import json
from pathlib import Path

import pytest

from seed_pipeline.artifacts.contract import (
    ContractError,
    build_manifest,
    validate_contract_directory,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _published(tmp_path: Path, *, block_section: str = "drug:a:b") -> Path:
    final_dir = tmp_path / "rag-final"
    final_dir.mkdir()
    _write_jsonl(final_dir / "sections.jsonl", [{"id": "drug:a:b", "text": "Văn bản"}])
    _write_jsonl(
        final_dir / "blocks.jsonl",
        [{"block_id": "block-000001", "section_id": block_section, "text": "Văn bản"}],
    )
    _write_jsonl(
        final_dir / "chunks.jsonl",
        [
            {
                "chunk_id": "drug:a:b:chunk-001",
                "section_id": "drug:a:b",
                "chunk_text": "Văn bản",
                "embedding_text": "Văn bản",
            }
        ],
    )
    (final_dir / "validation_report.json").write_text('{"ok": true}\n', encoding="utf-8")
    manifest = build_manifest(
        final_dir,
        build_id="build",
        source_pdf_sha256="pdf",
        snapshot_id="snapshot",
        snapshot_sha256="archive",
        curated_input_digests={"glossary": "g"},
        config_digest="config",
    )
    (final_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return final_dir


def test_manifest_tracks_blocks_and_contract_validates(tmp_path: Path) -> None:
    final_dir = _published(tmp_path)

    manifest = validate_contract_directory(final_dir)

    assert manifest["block_count"] == 1
    assert set(manifest["files"]) == {
        "sections.jsonl",
        "blocks.jsonl",
        "chunks.jsonl",
        "validation_report.json",
    }


def test_contract_requires_blocks_file(tmp_path: Path) -> None:
    final_dir = _published(tmp_path)
    (final_dir / "blocks.jsonl").unlink()

    with pytest.raises(ContractError, match="must contain exactly"):
        validate_contract_directory(final_dir)


def test_contract_rejects_blocks_of_unknown_sections(tmp_path: Path) -> None:
    final_dir = _published(tmp_path, block_section="drug:missing:section")

    with pytest.raises(ContractError, match="unknown sections"):
        validate_contract_directory(final_dir)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/artifacts/test_contract.py`
Expected: FAIL — `test_manifest_tracks_blocks_and_contract_validates` raises `ContractError: Final contract must contain exactly [...]` because `blocks.jsonl` is not in `CONTRACT_FILES` yet (and `block_count` is absent).

- [ ] **Step 3: Implement the contract change**

In `seed-pipeline/src/seed_pipeline/artifacts/contract.py` replace `CONTRACT_FILES`:

```python
CONTRACT_FILES = frozenset(
    {
        "sections.jsonl",
        "blocks.jsonl",
        "chunks.jsonl",
        "manifest.json",
        "validation_report.json",
    }
)
TRACKED_FILES = ("sections.jsonl", "blocks.jsonl", "chunks.jsonl", "validation_report.json")
```

Replace `build_manifest`:

```python
def build_manifest(
    final_dir: Path,
    *,
    build_id: str,
    source_pdf_sha256: str,
    snapshot_id: str,
    snapshot_sha256: str,
    curated_input_digests: dict[str, str],
    config_digest: str,
) -> dict[str, Any]:
    final_dir = Path(final_dir)
    sections = _jsonl_records(final_dir / "sections.jsonl")
    blocks = _jsonl_records(final_dir / "blocks.jsonl")
    chunks = _jsonl_records(final_dir / "chunks.jsonl")
    report = json.loads((final_dir / "validation_report.json").read_text("utf-8"))
    tracked_files = {
        name: {
            "sha256": sha256_file(final_dir / name),
            "bytes": (final_dir / name).stat().st_size,
        }
        for name in TRACKED_FILES
    }
    return {
        "schema_version": "rag-final-v2",
        "build_id": build_id,
        "source_pdf_sha256": source_pdf_sha256,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "curated_input_digests": dict(sorted(curated_input_digests.items())),
        "config_digest": config_digest,
        "section_count": len(sections),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "validation_ok": bool(report.get("ok")),
        "files": tracked_files,
    }
```

Replace `validate_contract_directory`:

```python
def validate_contract_directory(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        raise ContractError(f"Final contract directory is missing: {path}")
    actual = {entry.name for entry in path.iterdir()}
    if actual != CONTRACT_FILES:
        raise ContractError(
            f"Final contract must contain exactly {sorted(CONTRACT_FILES)}; "
            f"found {sorted(actual)}"
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rag-final-v2":
        raise ContractError("Unsupported final contract schema")
    report = json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("ok") is not True or manifest.get("validation_ok") is not True:
        raise ContractError("Final contract validation report is not successful")
    sections = _jsonl_records(path / "sections.jsonl")
    blocks = _jsonl_records(path / "blocks.jsonl")
    chunks = _jsonl_records(path / "chunks.jsonl")
    if not sections or not blocks or not chunks:
        raise ContractError("Final contract JSONL files must not be empty")
    if any(not record.get("id") or not record.get("text") for record in sections):
        raise ContractError("Every section requires id and text")
    if any(not record.get("block_id") or not record.get("section_id") for record in blocks):
        raise ContractError("Every block requires block_id and section_id")
    if any(
        not record.get("chunk_id")
        or not record.get("section_id")
        or not record.get("chunk_text")
        or not record.get("embedding_text")
        for record in chunks
    ):
        raise ContractError(
            "Every unified chunk requires chunk_id, section_id, chunk_text, and embedding_text"
        )
    section_ids = {record["id"] for record in sections}
    if len(section_ids) != len(sections):
        raise ContractError("Final sections contain duplicate IDs")
    if len({record["block_id"] for record in blocks}) != len(blocks):
        raise ContractError("Final blocks contain duplicate IDs")
    orphans = sorted({str(record["section_id"]) for record in blocks} - section_ids)
    if orphans:
        raise ContractError(f"Final blocks reference unknown sections: {orphans[:5]}")
    if len({record["chunk_id"] for record in chunks}) != len(chunks):
        raise ContractError("Final chunks contain duplicate IDs")
    for name in TRACKED_FILES:
        expected = manifest.get("files", {}).get(name, {})
        file_path = path / name
        if expected.get("sha256") != sha256_file(file_path):
            raise ContractError(f"Final contract checksum mismatch: {name}")
        if int(expected.get("bytes", -1)) != file_path.stat().st_size:
            raise ContractError(f"Final contract size mismatch: {name}")
    if int(manifest.get("section_count", -1)) != len(sections):
        raise ContractError("Final section count mismatch")
    if int(manifest.get("block_count", -1)) != len(blocks):
        raise ContractError("Final block count mismatch")
    if int(manifest.get("chunk_count", -1)) != len(chunks):
        raise ContractError("Final chunk count mismatch")
    return manifest
```

In `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py` (`build_candidate`), replace the single `shutil.copy2(... "sections.jsonl" ...)` call with:

```python
    shutil.copy2(
        paths.source_final_dir / "sections.jsonl",
        paths.candidate_final_dir / "sections.jsonl",
    )
    shutil.copy2(
        paths.canonical_dir / "blocks.jsonl",
        paths.candidate_final_dir / "blocks.jsonl",
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest -q tests/artifacts/test_contract.py`
Expected: `3 passed`.

- [ ] **Step 5: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/artifacts/contract.py seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py seed-pipeline/tests/artifacts/test_contract.py
git commit -m "feat(seed): publish canonical blocks with the corpus build"
```

The commit message ends with the session attribution trailer.

---

### Task 4: `seed bundle export`

**Files:**
- Create: `seed-pipeline/tests/fixtures/rag_final_small/{sections.jsonl,blocks.jsonl,chunks.jsonl,manifest.json,validation_report.json,term_glossary.json,colloquial_mappings.json}`
- Create: `seed-pipeline/src/seed_pipeline/bundle/__init__.py`, `seed-pipeline/src/seed_pipeline/bundle/chunks.py`, `seed-pipeline/src/seed_pipeline/bundle/export.py`
- Create: `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/app.py` (register the `bundle` group)
- Create: `seed-pipeline/tests/bundle/test_export.py`, `seed-pipeline/tests/cli/test_bundle_command.py`

**Interfaces:**
- Consumes (P1): `BlockKind`, `BlockRecord`, `BundleCollection`, `BundleGenerator`, `BundleManifest`, `ColloquialMappingRecord`, `DocumentKind`, `DocumentRecord`, `GlossaryEntry`, `KnowledgeBundle`, `RetrievalMode`, `SectionRecord`, `SourceInfo`, `BundleValidationError` (`problems: list[str]`), `read_bundle(directory) -> KnowledgeBundle`, `write_bundle(bundle, directory) -> BundleManifest`, `MAX_CHUNK_CHARS`, `ChunkDraft`, `chunk_section(document, section, glossary, mappings, *, max_chars)`.
- Consumes (seed-pipeline): `BRAND_INDEX_SECTION_ID`, `APPENDIX_LIST_SECTION_IDS` (`corpus/canonical/build_canonical_rag.py`), `product_names_from_title`, `resolve_colloquial_mapping` (`corpus/crawling/integrate_ankhang.py`), `compact_colloquial_mapping` (`corpus/metadata/payload_layers.py`), `slugify` (`corpus/processing/preprocess_rag_corpus.py`), `iter_jsonl_objects` (`evaluation/artifact_contracts.py`).
- Produces:
  - `seed_pipeline.bundle.export`: `COLLECTION_KEY = "formulary"`, `COLLECTION_TITLE`, `ExportRequest(rag_final_dir: Path, glossary_path: Path, mappings_path: Path, output_dir: Path, force: bool = False)`, `ExportResult(manifest: BundleManifest, skipped_sections: tuple[str, ...])`, `export_bundle(request) -> ExportResult`, `block_kind_for(section_key: str, content_type: str) -> BlockKind`, `document_for_section(row: Mapping[str, Any]) -> DocumentRecord`, `leaflet_blocks(text: str) -> list[BlockRecord]`.
  - `seed_pipeline.bundle.chunks`: `SectionChunks(document, section, drafts)`, `iter_section_chunks(bundle, *, max_chars=MAX_CHUNK_CHARS) -> Iterator[SectionChunks]`, `gold_chunk_label(section_key: str, ordinal: int) -> str`.
  - CLI: `seed bundle export --output DIR [--rag-final-dir DIR] [--glossary FILE] [--mappings FILE] [--force]`.

Mapping rules implemented here (spec §5.2, §10.3):

| Source | Bundle |
| --- | --- |
| Formulary section (`content_type` `drug_monograph`/`general_monograph`) | Document `drug:<slugify(title)>` / `general:<slugify(title)>`, `source.title` = section `source`, `url = None`; section `ordinal` = position inside the document in `sections.jsonl` order, from 1 |
| Formulary block `content_type == "table"` | `BlockKind.TABLE`, `markdown` = block `markdown`, `table_key` = `table_id` |
| Paragraph block of `BRAND_INDEX_SECTION_ID` | `BlockKind.INDEX_ENTRIES`, section `retrieval = RetrievalMode.INDEX_ONLY` |
| Paragraph block of `APPENDIX_LIST_SECTION_IDS` | `BlockKind.LIST` |
| Other paragraph blocks | `BlockKind.PROSE`, `markdown` = block `text` unchanged |
| An Khang section `brand:ankhang:<category>:<slug>` | Document `leaflet:ankhang:<category>:<slug>`, `url = https://www.nhathuocankhang.com/<category>/<slug>`; blocks follow P1 decision 4 (`leaflet_blocks`): paragraphs split on blank lines, consecutive paragraphs joined with `"\n\n"` into one `PROSE` block, and each pipe-table paragraph longer than `MAX_CHUNK_CHARS` (3 000) as its own `TABLE` block |
| Page `0` or missing | `None` |
| Blocks whose text is blank | dropped; a section left without blocks is skipped and listed in `ExportResult.skipped_sections` |
| `colloquial_mappings.json` | JSON array (P1): one record per curated slug (aliases, visual sign; kept without leaflets) plus one record per uncurated leaflet keyed by its slug; `product_names` from `product_names_from_title(title) or [title]`; `section_keys` = leaflets resolved to that key by `resolve_colloquial_mapping`; keys unique, a section in at most one record |
| `rag-final/manifest.json` | `generator = {name: "seed-pipeline", version: <installed version>, build_id}`, `source_digests = {source_pdf_sha256, snapshot_id, snapshot_sha256, curated_<name>_sha256...}` |

- [ ] **Step 1: Create the fixture**

The fixture is a tiny published build produced on 2026-09-13 by the pre-migration code (`build_canonical_sections`, `build_final_chunk_records`, `integrate_ankhang_corpus`, `compile_unified_chunks`) from two formulary sections of one drug (prose + table), the brand index section and two An Khang leaflets (one with a curated colloquial mapping). Its `chunks.jsonl` is the old unified chunk contract that Task 5 compares against. Write the files byte for byte:

`seed-pipeline/tests/fixtures/rag_final_small/sections.jsonl`:

```json
{"block_ids": ["block-000001", "block-000002"], "content_type": "drug_monograph", "context_header": "PARACETAMOL\n> Liều lượng và cách dùng", "context_path": ["Liều lượng và cách dùng"], "end_page": 1134, "hydrate_strategy": "full_section", "id": "drug:paracetamol:lieu-luong-va-cach-dung", "quality_flags": [], "section": "Liều lượng và cách dùng", "section_char_count": 202, "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "source_mix": ["docling_table", "pymupdf_text"], "start_page": 1133, "table_ids": ["curated-table-1134-001"], "text": "Người lớn: uống 500 mg đến 1 g mỗi 4 đến 6 giờ khi cần. Không dùng đồng thời với NSAID khác khi chưa có chỉ định.\n\n| Tuổi | Liều |\n| --- | --- |\n| Trẻ 6-12 tuổi | 250-500 mg |\n| Người lớn | 500 mg-1 g |", "title": "PARACETAMOL", "warnings": []}
{"block_ids": ["block-000003"], "content_type": "drug_monograph", "context_header": "PARACETAMOL\n> Chống chỉ định", "context_path": ["Chống chỉ định"], "end_page": 1133, "hydrate_strategy": "full_section", "id": "drug:paracetamol:chong-chi-dinh", "quality_flags": [], "section": "Chống chỉ định", "section_char_count": 72, "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "source_mix": ["pymupdf_text"], "start_page": 1133, "table_ids": [], "text": "Người bệnh quá mẫn với paracetamol. Đã từng có ADR nghiêm trọng trên da.", "title": "PARACETAMOL", "warnings": []}
{"block_ids": ["block-000004"], "content_type": "general_monograph", "context_header": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược", "context_path": ["Bảng tra cứu biệt dược"], "end_page": 1600, "hydrate_strategy": "search_only", "id": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc", "quality_flags": [], "section": "Bảng tra cứu biệt dược", "section_char_count": 140, "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "source_mix": ["pymupdf_text"], "start_page": 1529, "table_ids": [], "text": "- **Efferalgan**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)\n- **Panadol**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)", "title": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT", "warnings": []}
{"id": "brand:ankhang:thuoc:hapacol-250-dhg-11500", "content_type": "brand_page", "title": "Hapacol 250 DHG (hộp 24 gói)", "section": "Thông tin chi tiết", "text": "# Hapacol 250 DHG (hộp 24 gói)\n\n## Công dụng\n\nHạ sốt, giảm đau cho trẻ em.", "source": "Tờ hướng dẫn sử dụng", "context_path": ["Thông tin chi tiết"], "context_header": "Hapacol 250 DHG (hộp 24 gói)\n> Thông tin chi tiết", "colloquial_mapping": {"product_names": ["Hapacol 250 DHG"]}, "start_page": 0, "end_page": 0, "warnings": [], "hydrate_strategy": "full_section", "section_char_count": 74}
{"id": "brand:ankhang:thuoc:panadol-extra-gsk-150-vien-11440", "content_type": "brand_page", "title": "Panadol Extra GSK (hộp 15 vỉ x 10 viên)", "section": "Thông tin chi tiết", "text": "# Panadol Extra GSK (hộp 15 vỉ x 10 viên)\n\n## Thành phần\n\n| Thông tin thành phần | Hàm lượng |\n| --- | --- |\n| Paracetamol | 500 mg |\n| Caffeine | 65 mg |\n\n## Công dụng\n\nGiảm đau đầu, đau răng, hạ sốt.", "source": "Tờ hướng dẫn sử dụng", "context_path": ["Thông tin chi tiết"], "context_header": "Panadol Extra GSK (hộp 15 vỉ x 10 viên)\n> Thông tin chi tiết", "colloquial_mapping": {"key": "panadol-extra-gsk-150-vien-11440", "aliases": ["Panadol đỏ", "Panadol extra đỏ", "Panadol vỉ đỏ", "Panadol hộp đỏ"], "visual_sign": "Hộp màu đỏ, vỉ thuốc màu đỏ", "product_names": ["Panadol Extra GSK"]}, "start_page": 0, "end_page": 0, "warnings": [], "hydrate_strategy": "full_section", "section_char_count": 201}
```

`seed-pipeline/tests/fixtures/rag_final_small/blocks.jsonl`:

```json
{"block_id": "block-000001", "content_type": "paragraph", "markdown": null, "page_end": 1134, "page_start": 1133, "quality_flags": [], "section": "Liều lượng và cách dùng", "section_id": "drug:paracetamol:lieu-luong-va-cach-dung", "section_index": 0, "source_refs": ["sections.jsonl:0:drug:paracetamol:lieu-luong-va-cach-dung"], "source_type": "pymupdf_text", "table_id": null, "text": "Người lớn: uống 500 mg đến 1 g mỗi 4 đến 6 giờ khi cần. Không dùng đồng thời với NSAID khác khi chưa có chỉ định.", "title": "PARACETAMOL"}
{"block_id": "block-000002", "content_type": "table", "markdown": "| Tuổi | Liều |\n| --- | --- |\n| Trẻ 6-12 tuổi | 250-500 mg |\n| Người lớn | 500 mg-1 g |", "page_end": 1134, "page_start": 1134, "quality_flags": [], "section": "Liều lượng và cách dùng", "section_id": "drug:paracetamol:lieu-luong-va-cach-dung", "section_index": 0, "source_refs": ["sections.jsonl:0:drug:paracetamol:lieu-luong-va-cach-dung"], "source_type": "docling_table", "table_id": "curated-table-1134-001", "text": null, "title": "PARACETAMOL"}
{"block_id": "block-000003", "content_type": "paragraph", "markdown": null, "page_end": 1133, "page_start": 1133, "quality_flags": [], "section": "Chống chỉ định", "section_id": "drug:paracetamol:chong-chi-dinh", "section_index": 1, "source_refs": ["sections.jsonl:1:drug:paracetamol:chong-chi-dinh"], "source_type": "pymupdf_text", "table_id": null, "text": "Người bệnh quá mẫn với paracetamol. Đã từng có ADR nghiêm trọng trên da.", "title": "PARACETAMOL"}
{"block_id": "block-000004", "content_type": "paragraph", "markdown": null, "page_end": 1600, "page_start": 1529, "quality_flags": [], "section": "Bảng tra cứu biệt dược", "section_id": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc", "section_index": 2, "source_refs": ["sections.jsonl:2:general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc"], "source_type": "pymupdf_text", "table_id": null, "text": "- **Efferalgan**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)\n- **Panadol**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)", "title": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT"}
```

`seed-pipeline/tests/fixtures/rag_final_small/chunks.jsonl`:

```json
{"chunk_id": "drug:paracetamol:lieu-luong-va-cach-dung:chunk-001", "section_id": "drug:paracetamol:lieu-luong-va-cach-dung", "chunk_index": 1, "hydrate_strategy": "full_section", "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "title": "PARACETAMOL", "section": "Liều lượng và cách dùng", "start_page": 1133, "end_page": 1134, "context_header": "PARACETAMOL\n> Liều lượng và cách dùng", "chunk_text": "Người lớn: uống 500 mg đến 1 g mỗi 4 đến 6 giờ khi cần. Không dùng đồng thời với NSAID khác khi chưa có chỉ định.", "embedding_text": "PARACETAMOL\n> Liều lượng và cách dùng\n\nNgười lớn: uống 500 mg đến 1 g mỗi 4 đến 6 giờ khi cần. Không dùng đồng thời với NSAID khác khi chưa có chỉ định.\n\nThuật ngữ: NSAID = thuốc chống viêm không steroid; thuốc kháng viêm không steroid; Nonsteroidal Anti-inflammatory Drug", "term_annotations": [{"term": "NSAID", "vi": ["thuốc chống viêm không steroid", "thuốc kháng viêm không steroid"], "en": ["Nonsteroidal Anti-inflammatory Drug"]}], "content_type": "drug_monograph", "chunk_role": "prose", "chunk_content_type": "paragraph", "source_block_id": "block-000001", "context_path": ["Liều lượng và cách dùng"], "section_char_count": 202, "warnings": []}
{"chunk_id": "drug:paracetamol:lieu-luong-va-cach-dung:chunk-002", "section_id": "drug:paracetamol:lieu-luong-va-cach-dung", "chunk_index": 2, "hydrate_strategy": "full_section", "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "title": "PARACETAMOL", "section": "Liều lượng và cách dùng", "start_page": 1134, "end_page": 1134, "context_header": "PARACETAMOL\n> Liều lượng và cách dùng", "chunk_text": "| Tuổi | Liều |\n| --- | --- |\n| Trẻ 6-12 tuổi | 250-500 mg |\n| Người lớn | 500 mg-1 g |", "embedding_text": "PARACETAMOL\n> Liều lượng và cách dùng\n\n| Tuổi | Liều |\n| --- | --- |\n| Trẻ 6-12 tuổi | 250-500 mg |\n| Người lớn | 500 mg-1 g |", "content_type": "drug_monograph", "table_id": "curated-table-1134-001", "chunk_role": "table", "chunk_content_type": "table", "source_block_id": "block-000002", "context_path": ["Liều lượng và cách dùng"], "section_char_count": 202, "warnings": []}
{"chunk_id": "drug:paracetamol:chong-chi-dinh:chunk-001", "section_id": "drug:paracetamol:chong-chi-dinh", "chunk_index": 1, "hydrate_strategy": "full_section", "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "title": "PARACETAMOL", "section": "Chống chỉ định", "start_page": 1133, "end_page": 1133, "context_header": "PARACETAMOL\n> Chống chỉ định", "chunk_text": "Người bệnh quá mẫn với paracetamol. Đã từng có ADR nghiêm trọng trên da.", "embedding_text": "PARACETAMOL\n> Chống chỉ định\n\nNgười bệnh quá mẫn với paracetamol. Đã từng có ADR nghiêm trọng trên da.\n\nThuật ngữ: ADR = tác dụng không mong muốn; phản ứng có hại của thuốc; Adverse Drug Reactions", "term_annotations": [{"term": "ADR", "vi": ["tác dụng không mong muốn", "phản ứng có hại của thuốc"], "en": ["Adverse Drug Reactions"]}], "content_type": "drug_monograph", "chunk_role": "prose", "chunk_content_type": "paragraph", "source_block_id": "block-000003", "context_path": ["Chống chỉ định"], "section_char_count": 72, "warnings": []}
{"chunk_id": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc:chunk-001", "section_id": "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat:bang-tra-cuu-biet-duoc", "chunk_index": 1, "hydrate_strategy": "search_only", "source": "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) - Nhà xuất bản Y học, Hà Nội, 2018", "title": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT", "section": "Bảng tra cứu biệt dược", "start_page": 1529, "end_page": 1600, "context_header": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược", "chunk_text": "- **Efferalgan**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)\n- **Panadol**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)", "embedding_text": "MỤC LỤC TRA CỨU BIỆT DƯỢC VÀ HOẠT CHẤT\n> Bảng tra cứu biệt dược\n\n- **Efferalgan**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)\n- **Panadol**: biệt dược chứa hoạt chất **Paracetamol** (Trang 1133)", "content_type": "general_monograph", "chunk_role": "index_entry", "chunk_content_type": "paragraph", "source_block_id": "block-000004", "context_path": ["Bảng tra cứu biệt dược"], "section_char_count": 140, "warnings": []}
{"chunk_id": "brand:ankhang:thuoc:hapacol-250-dhg-11500:chunk-001", "section_id": "brand:ankhang:thuoc:hapacol-250-dhg-11500", "chunk_index": 1, "hydrate_strategy": "full_section", "source": "Tờ hướng dẫn sử dụng", "title": "Hapacol 250 DHG (hộp 24 gói)", "section": "Thông tin chi tiết", "start_page": 0, "end_page": 0, "context_header": "Hapacol 250 DHG (hộp 24 gói)\n> Thông tin chi tiết", "chunk_text": "# Hapacol 250 DHG (hộp 24 gói)\n\n## Công dụng\n\nHạ sốt, giảm đau cho trẻ em.", "embedding_text": "Hapacol 250 DHG (hộp 24 gói)\n> Thông tin chi tiết\n\n# Hapacol 250 DHG (hộp 24 gói)\n\n## Công dụng\n\nHạ sốt, giảm đau cho trẻ em.", "colloquial_mapping": {"product_names": ["Hapacol 250 DHG"]}, "content_type": "brand_page", "chunk_role": "prose", "chunk_content_type": "paragraph", "source_block_id": "", "context_path": ["Thông tin chi tiết"], "section_char_count": 74, "warnings": []}
{"chunk_id": "brand:ankhang:thuoc:panadol-extra-gsk-150-vien-11440:chunk-001", "section_id": "brand:ankhang:thuoc:panadol-extra-gsk-150-vien-11440", "chunk_index": 1, "hydrate_strategy": "full_section", "source": "Tờ hướng dẫn sử dụng", "title": "Panadol Extra GSK (hộp 15 vỉ x 10 viên)", "section": "Thông tin chi tiết", "start_page": 0, "end_page": 0, "context_header": "Panadol Extra GSK (hộp 15 vỉ x 10 viên)\n> Thông tin chi tiết", "chunk_text": "# Panadol Extra GSK (hộp 15 vỉ x 10 viên)\n\n## Thành phần\n\n| Thông tin thành phần | Hàm lượng |\n| --- | --- |\n| Paracetamol | 500 mg |\n| Caffeine | 65 mg |\n\n## Công dụng\n\nGiảm đau đầu, đau răng, hạ sốt.", "embedding_text": "Panadol Extra GSK (hộp 15 vỉ x 10 viên)\n> Thông tin chi tiết\n\nTên gọi khác: Panadol đỏ, Panadol extra đỏ, Panadol vỉ đỏ, Panadol hộp đỏ\nDấu hiệu nhận biết: Hộp màu đỏ, vỉ thuốc màu đỏ\n\n# Panadol Extra GSK (hộp 15 vỉ x 10 viên)\n\n## Thành phần\n\n| Thông tin thành phần | Hàm lượng |\n| --- | --- |\n| Paracetamol | 500 mg |\n| Caffeine | 65 mg |\n\n## Công dụng\n\nGiảm đau đầu, đau răng, hạ sốt.", "colloquial_mapping": {"key": "panadol-extra-gsk-150-vien-11440", "aliases": ["Panadol đỏ", "Panadol extra đỏ", "Panadol vỉ đỏ", "Panadol hộp đỏ"], "visual_sign": "Hộp màu đỏ, vỉ thuốc màu đỏ", "product_names": ["Panadol Extra GSK"]}, "content_type": "brand_page", "chunk_role": "prose", "chunk_content_type": "paragraph", "source_block_id": "", "context_path": ["Thông tin chi tiết"], "section_char_count": 201, "warnings": []}
```

`seed-pipeline/tests/fixtures/rag_final_small/manifest.json`:

```json
{
  "build_id": "fixture-build-0001",
  "chunk_count": 6,
  "config_digest": "4444444444444444444444444444444444444444444444444444444444444444",
  "curated_input_digests": {
    "curated_tables": "5555555555555555555555555555555555555555555555555555555555555555",
    "glossary": "6666666666666666666666666666666666666666666666666666666666666666",
    "mappings": "7777777777777777777777777777777777777777777777777777777777777777",
    "table_overrides": "8888888888888888888888888888888888888888888888888888888888888888"
  },
  "schema_version": "rag-final-v2",
  "section_count": 5,
  "snapshot_id": "ankhang-2026-07-24-97f5b5c43eee",
  "snapshot_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
  "source_pdf_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "validation_ok": true
}
```

`seed-pipeline/tests/fixtures/rag_final_small/validation_report.json`:

```json
{"ok": true}
```

`seed-pipeline/tests/fixtures/rag_final_small/term_glossary.json`:

```json
[
  {
    "term": "ADR",
    "case_sensitive": true,
    "vietnamese_expansions": [
      "tác dụng không mong muốn",
      "phản ứng có hại của thuốc"
    ],
    "english_expansions": [
      "Adverse Drug Reactions"
    ],
    "aliases": [
      "adverse drug reaction"
    ],
    "category": "safety",
    "confidence": "high",
    "source": "curated"
  },
  {
    "term": "NSAID",
    "case_sensitive": true,
    "vietnamese_expansions": [
      "thuốc chống viêm không steroid",
      "thuốc kháng viêm không steroid"
    ],
    "english_expansions": [
      "Nonsteroidal Anti-inflammatory Drug"
    ],
    "aliases": [
      "NSAIDs"
    ],
    "category": "drug_class",
    "confidence": "high",
    "source": "curated"
  }
]
```

`seed-pipeline/tests/fixtures/rag_final_small/colloquial_mappings.json`:

```json
{
  "panadol-extra-gsk-150-vien-11440": {
    "aliases": [
      "Panadol đỏ",
      "Panadol extra đỏ",
      "Panadol vỉ đỏ",
      "Panadol hộp đỏ"
    ],
    "visual_sign": "Hộp màu đỏ, vỉ thuốc màu đỏ"
  }
}
```

- [ ] **Step 2: Write the failing tests**

Create `seed-pipeline/tests/bundle/test_export.py`:

```python
import json
from pathlib import Path

import pytest
from pharma_agent.domain.corpus.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BlockKind,
    BundleCollection,
    ColloquialMappingRecord,
    DocumentKind,
    KnowledgeBundle,
    RetrievalMode,
    SourceInfo,
    read_bundle,
)

from seed_pipeline.bundle.chunks import gold_chunk_label, iter_section_chunks
from seed_pipeline.bundle.export import (
    COLLECTION_KEY,
    COLLECTION_TITLE,
    ExportRequest,
    ExportResult,
    block_kind_for,
    export_bundle,
    leaflet_blocks,
)
from seed_pipeline.corpus.canonical.build_canonical_rag import (
    ATC_SECTION_ID,
    BRAND_INDEX_SECTION_ID,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
DOSAGE = "drug:paracetamol:lieu-luong-va-cach-dung"
PANADOL = "brand:ankhang:thuoc:panadol-extra-gsk-150-vien-11440"
HAPACOL = "brand:ankhang:thuoc:hapacol-250-dhg-11500"


def _request(output_dir: Path, *, force: bool = False) -> ExportRequest:
    return ExportRequest(
        rag_final_dir=FIXTURE,
        glossary_path=FIXTURE / "term_glossary.json",
        mappings_path=FIXTURE / "colloquial_mappings.json",
        output_dir=output_dir,
        force=force,
    )


def _export(tmp_path: Path) -> tuple[ExportResult, KnowledgeBundle]:
    result = export_bundle(_request(tmp_path / "bundle"))
    return result, read_bundle(tmp_path / "bundle")


def test_export_groups_documents_and_keeps_section_keys(tmp_path: Path) -> None:
    result, bundle = _export(tmp_path)

    assert [document.key for document in bundle.documents] == [
        "drug:paracetamol",
        "general:muc-luc-tra-cuu-biet-duoc-va-hoat-chat",
        "leaflet:ankhang:thuoc:hapacol-250-dhg-11500",
        "leaflet:ankhang:thuoc:panadol-extra-gsk-150-vien-11440",
    ]
    assert [document.kind for document in bundle.documents] == [
        DocumentKind.DRUG_MONOGRAPH,
        DocumentKind.GENERAL_MONOGRAPH,
        DocumentKind.LEAFLET,
        DocumentKind.LEAFLET,
    ]
    assert bundle.documents[0].source.url is None
    assert bundle.documents[2].source == SourceInfo(
        title="Tờ hướng dẫn sử dụng",
        url="https://www.nhathuocankhang.com/thuoc/hapacol-250-dhg-11500",
    )
    assert [section.key for section in bundle.sections] == [
        DOSAGE,
        "drug:paracetamol:chong-chi-dinh",
        BRAND_INDEX_SECTION_ID,
        HAPACOL,
        PANADOL,
    ]
    assert [section.ordinal for section in bundle.sections] == [1, 2, 1, 1, 1]
    assert result.skipped_sections == ()
    assert result.manifest.schema_version == BUNDLE_SCHEMA_VERSION
    assert result.manifest.collection == BundleCollection(
        key=COLLECTION_KEY, title=COLLECTION_TITLE
    )
    assert (result.manifest.document_count, result.manifest.section_count) == (4, 5)
    assert result.manifest.generator.name == "seed-pipeline"
    assert result.manifest.generator.build_id == "fixture-build-0001"
    assert result.manifest.source_digests == {
        "source_pdf_sha256": "1" * 64,
        "snapshot_id": "ankhang-2026-07-24-97f5b5c43eee",
        "snapshot_sha256": "2" * 64,
        "curated_curated_tables_sha256": "5" * 64,
        "curated_glossary_sha256": "6" * 64,
        "curated_mappings_sha256": "7" * 64,
        "curated_table_overrides_sha256": "8" * 64,
    }


def test_export_maps_block_kinds_retrieval_and_pages(tmp_path: Path) -> None:
    _, bundle = _export(tmp_path)
    sections = {section.key: section for section in bundle.sections}
    old_chunks = [
        json.loads(line)
        for line in (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    ]

    dosage = sections[DOSAGE]
    assert [block.kind for block in dosage.blocks] == [BlockKind.PROSE, BlockKind.TABLE]
    assert dosage.blocks[1].table_key == "curated-table-1134-001"
    assert (dosage.blocks[1].start_page, dosage.blocks[1].end_page) == (1134, 1134)
    assert (dosage.start_page, dosage.end_page) == (1133, 1134)
    assert dosage.heading == "Liều lượng và cách dùng"
    assert dosage.context_path == ["Liều lượng và cách dùng"]

    index = sections[BRAND_INDEX_SECTION_ID]
    assert index.retrieval is RetrievalMode.INDEX_ONLY
    assert [block.kind for block in index.blocks] == [BlockKind.INDEX_ENTRIES]
    assert sections[DOSAGE].retrieval is RetrievalMode.DEFAULT

    leaflet = sections[PANADOL]
    assert (leaflet.start_page, leaflet.end_page) == (None, None)
    assert [block.kind for block in leaflet.blocks] == [BlockKind.PROSE]
    assert [block.start_page for block in leaflet.blocks] == [None]
    old_leaflet = next(chunk for chunk in old_chunks if chunk["section_id"] == PANADOL)
    assert leaflet.blocks[0].markdown == old_leaflet["chunk_text"]

    assert block_kind_for(ATC_SECTION_ID, "paragraph") is BlockKind.LIST
    assert block_kind_for(ATC_SECTION_ID, "table") is BlockKind.TABLE


def test_export_copies_glossary_and_attaches_mappings(tmp_path: Path) -> None:
    _, bundle = _export(tmp_path)
    mappings = {mapping.key: mapping for mapping in bundle.colloquial_mappings}

    assert len(mappings) == len(bundle.colloquial_mappings)
    assert [entry.term for entry in bundle.glossary] == ["ADR", "NSAID"]
    assert mappings["panadol-extra-gsk-150-vien-11440"] == ColloquialMappingRecord(
        key="panadol-extra-gsk-150-vien-11440",
        aliases=["Panadol đỏ", "Panadol extra đỏ", "Panadol vỉ đỏ", "Panadol hộp đỏ"],
        visual_sign="Hộp màu đỏ, vỉ thuốc màu đỏ",
        product_names=["Panadol Extra GSK"],
        section_keys=[PANADOL],
    )
    assert mappings["hapacol-250-dhg-11500"] == ColloquialMappingRecord(
        key="hapacol-250-dhg-11500",
        product_names=["Hapacol 250 DHG"],
        section_keys=[HAPACOL],
    )


def test_export_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    export_bundle(_request(tmp_path / "bundle"))

    with pytest.raises(FileExistsError, match="--force"):
        export_bundle(_request(tmp_path / "bundle"))
    replaced = export_bundle(_request(tmp_path / "bundle", force=True))
    assert replaced.manifest.section_count == 5


def test_leaflet_blocks_keep_long_pipe_tables_apart() -> None:
    rows = "\n".join(f"| thuốc {index} | {'x' * 40} |" for index in range(80))
    table = f"| Thành phần | Hàm lượng |\n| --- | --- |\n{rows}"

    blocks = leaflet_blocks(f"# Thuốc\n\nĐoạn một.\n\n{table}\n\n\nĐoạn hai.")

    assert [block.kind for block in blocks] == [
        BlockKind.PROSE,
        BlockKind.TABLE,
        BlockKind.PROSE,
    ]
    assert blocks[0].markdown == "# Thuốc\n\nĐoạn một."
    assert blocks[1].markdown == table
    assert blocks[2].markdown == "Đoạn hai."


def test_iter_section_chunks_runs_backend_chunker(tmp_path: Path) -> None:
    _, bundle = _export(tmp_path)

    items = list(iter_section_chunks(bundle))

    assert [item.section.key for item in items] == [s.key for s in bundle.sections]
    assert [draft.ordinal for draft in items[0].drafts] == [1, 2]
    assert items[0].document.key == "drug:paracetamol"
    assert gold_chunk_label(DOSAGE, 2) == f"{DOSAGE}:chunk-002"
```

Create `seed-pipeline/tests/cli/test_bundle_command.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from seed_pipeline.cli.app import app

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
runner = CliRunner()


def test_bundle_export_command_writes_a_valid_bundle(tmp_path: Path) -> None:
    output = tmp_path / "bundle"

    result = runner.invoke(
        app,
        [
            "--json",
            "bundle",
            "export",
            "--output",
            str(output),
            "--rag-final-dir",
            str(FIXTURE),
            "--glossary",
            str(FIXTURE / "term_glossary.json"),
            "--mappings",
            str(FIXTURE / "colloquial_mappings.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "complete"
    assert payload["details"] == {"documents": 4, "sections": 5, "skipped_sections": []}
    assert (output / "manifest.json").is_file()
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/bundle/test_export.py tests/cli/test_bundle_command.py`
Expected: collection errors `ModuleNotFoundError: No module named 'seed_pipeline.bundle'`.

- [ ] **Step 4: Implement the export**

Create `seed-pipeline/src/seed_pipeline/bundle/__init__.py`:

```python
"""Knowledge-bundle export, parity checks and embeddings built on the backend corpus domain."""
```

Create `seed-pipeline/src/seed_pipeline/bundle/chunks.py`:

```python
"""Run the backend chunker over every section of a knowledge bundle."""

from collections.abc import Iterator
from dataclasses import dataclass

from pharma_agent.domain.corpus.bundle import (
    DocumentRecord,
    KnowledgeBundle,
    SectionRecord,
)
from pharma_agent.domain.corpus.chunking import (
    MAX_CHUNK_CHARS,
    ChunkDraft,
    chunk_section,
)


@dataclass(frozen=True)
class SectionChunks:
    document: DocumentRecord
    section: SectionRecord
    drafts: list[ChunkDraft]


def gold_chunk_label(section_key: str, ordinal: int) -> str:
    """Position label used by evaluation gold sets; equals the pre-migration chunk_id."""
    return f"{section_key}:chunk-{ordinal:03d}"


def iter_section_chunks(
    bundle: KnowledgeBundle, *, max_chars: int = MAX_CHUNK_CHARS
) -> Iterator[SectionChunks]:
    documents = {document.key: document for document in bundle.documents}
    for section in bundle.sections:
        document = documents[section.document_key]
        yield SectionChunks(
            document=document,
            section=section,
            drafts=chunk_section(
                document,
                section,
                bundle.glossary,
                bundle.colloquial_mappings,
                max_chars=max_chars,
            ),
        )
```

Create `seed-pipeline/src/seed_pipeline/bundle/export.py`:

```python
"""Map a published corpus build (`rag-final/`) to a knowledge-bundle/v1 directory."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pharma_agent.domain.corpus.bundle import (
    BlockKind,
    BlockRecord,
    BundleCollection,
    BundleGenerator,
    BundleManifest,
    ColloquialMappingRecord,
    DocumentKind,
    DocumentRecord,
    GlossaryEntry,
    KnowledgeBundle,
    RetrievalMode,
    SectionRecord,
    SourceInfo,
    read_bundle,
    write_bundle,
)
from pharma_agent.domain.corpus.chunking import MAX_CHUNK_CHARS

from seed_pipeline.corpus.canonical.build_canonical_rag import (
    APPENDIX_LIST_SECTION_IDS,
    BRAND_INDEX_SECTION_ID,
)
from seed_pipeline.corpus.crawling.integrate_ankhang import (
    product_names_from_title,
    resolve_colloquial_mapping,
)
from seed_pipeline.corpus.metadata.payload_layers import compact_colloquial_mapping
from seed_pipeline.corpus.processing.preprocess_rag_corpus import slugify
from seed_pipeline.evaluation.artifact_contracts import iter_jsonl_objects

COLLECTION_KEY = "formulary"
COLLECTION_TITLE = "Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng An Khang"
ANKHANG_SECTION_PREFIX = "brand:ankhang:"
ANKHANG_BASE_URL = "https://www.nhathuocankhang.com"
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ExportRequest:
    rag_final_dir: Path
    glossary_path: Path
    mappings_path: Path
    output_dir: Path
    force: bool = False


@dataclass(frozen=True)
class ExportResult:
    manifest: BundleManifest
    skipped_sections: tuple[str, ...]


@dataclass
class _MappingDraft:
    aliases: list[str]
    visual_sign: str
    product_names: list[str] = field(default_factory=list)
    section_keys: list[str] = field(default_factory=list)


def block_kind_for(section_key: str, content_type: str) -> BlockKind:
    if content_type == "table":
        return BlockKind.TABLE
    if section_key == BRAND_INDEX_SECTION_ID:
        return BlockKind.INDEX_ENTRIES
    if section_key in APPENDIX_LIST_SECTION_IDS:
        return BlockKind.LIST
    return BlockKind.PROSE


def document_for_section(row: Mapping[str, Any]) -> DocumentRecord:
    section_key = str(row["id"])
    title = str(row["title"])
    source_title = str(row.get("source") or "")
    if section_key.startswith(ANKHANG_SECTION_PREFIX):
        category, slug = _leaflet_path(section_key)
        return DocumentRecord(
            key=f"leaflet:ankhang:{category}:{slug}",
            kind=DocumentKind.LEAFLET,
            title=title,
            source=SourceInfo(
                title=source_title, url=f"{ANKHANG_BASE_URL}/{category}/{slug}"
            ),
        )
    content_type = str(row.get("content_type") or "")
    if content_type == "drug_monograph":
        return DocumentRecord(
            key=f"drug:{slugify(title)}",
            kind=DocumentKind.DRUG_MONOGRAPH,
            title=title,
            source=SourceInfo(title=source_title),
        )
    if content_type == "general_monograph":
        return DocumentRecord(
            key=f"general:{slugify(title)}",
            kind=DocumentKind.GENERAL_MONOGRAPH,
            title=title,
            source=SourceInfo(title=source_title),
        )
    raise ValueError(f"Section {section_key} has unsupported content_type {content_type!r}")


def export_bundle(request: ExportRequest) -> ExportResult:
    output_dir = Path(request.output_dir)
    if output_dir.exists() and not request.force:
        raise FileExistsError(
            f"Bundle directory already exists: {output_dir}; use --force to replace it"
        )
    rag_final_dir = Path(request.rag_final_dir)
    blocks_by_section = _blocks_by_section(
        iter_jsonl_objects(rag_final_dir / "blocks.jsonl")
    )
    build_manifest = _read_json_object(rag_final_dir / "manifest.json")
    mappings = _MappingCollector(_read_json_object(request.mappings_path))
    documents: dict[str, DocumentRecord] = {}
    ordinals: dict[str, int] = {}
    sections: list[SectionRecord] = []
    skipped: list[str] = []
    for row in iter_jsonl_objects(rag_final_dir / "sections.jsonl"):
        section_key = str(row["id"])
        is_leaflet = section_key.startswith(ANKHANG_SECTION_PREFIX)
        blocks = (
            leaflet_blocks(str(row.get("text") or ""))
            if is_leaflet
            else _formulary_blocks(section_key, blocks_by_section.get(section_key, []))
        )
        if not blocks:
            skipped.append(section_key)
            continue
        document = document_for_section(row)
        existing = documents.setdefault(document.key, document)
        if existing.title != document.title:
            raise ValueError(
                f"Document key {document.key} is shared by titles "
                f"{existing.title!r} and {document.title!r}"
            )
        ordinals[document.key] = ordinals.get(document.key, 0) + 1
        heading = str(row["section"])
        sections.append(
            SectionRecord(
                key=section_key,
                document_key=document.key,
                heading=heading,
                context_path=[str(part) for part in row.get("context_path") or [heading]],
                ordinal=ordinals[document.key],
                start_page=_page(row.get("start_page")),
                end_page=_page(row.get("end_page")),
                retrieval=(
                    RetrievalMode.INDEX_ONLY
                    if section_key == BRAND_INDEX_SECTION_ID
                    else RetrievalMode.DEFAULT
                ),
                blocks=blocks,
            )
        )
        if is_leaflet:
            mappings.add_leaflet(section_key, document.title)

    bundle = KnowledgeBundle(
        manifest=BundleManifest(
            schema_version="knowledge-bundle/v1",
            collection=BundleCollection(key=COLLECTION_KEY, title=COLLECTION_TITLE),
            generator=BundleGenerator(
                name="seed-pipeline",
                version=version("seed-pipeline"),
                build_id=str(build_manifest["build_id"]),
            ),
            source_digests=_source_digests(build_manifest),
            document_count=len(documents),
            section_count=len(sections),
            files={},
        ),
        documents=list(documents.values()),
        sections=sections,
        glossary=_glossary(request.glossary_path),
        colloquial_mappings=mappings.records(),
    )
    staging = output_dir.with_name(f".{output_dir.name}.next")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest = write_bundle(bundle, staging)
    read_bundle(staging)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return ExportResult(manifest=manifest, skipped_sections=tuple(skipped))


class _MappingCollector:
    def __init__(self, curated: dict[str, Any]) -> None:
        self._curated = curated
        self._drafts: dict[str, _MappingDraft] = {}
        for key, value in curated.items():
            entry = value if isinstance(value, dict) else {}
            self._drafts[str(key)] = _MappingDraft(
                aliases=[str(alias) for alias in entry.get("aliases") or []],
                visual_sign=str(entry.get("visual_sign") or ""),
            )

    def add_leaflet(self, section_key: str, title: str) -> None:
        _, slug = _leaflet_path(section_key)
        resolved = resolve_colloquial_mapping(slug, self._curated)
        key = str(resolved["mapping_key"]) or slug
        draft = self._drafts.setdefault(key, _MappingDraft(aliases=[], visual_sign=""))
        for name in product_names_from_title(title) or [title]:
            if name not in draft.product_names:
                draft.product_names.append(name)
        draft.section_keys.append(section_key)

    def records(self) -> list[ColloquialMappingRecord]:
        records: list[ColloquialMappingRecord] = []
        for key, draft in self._drafts.items():
            compact = compact_colloquial_mapping(
                {
                    "product_aliases": draft.aliases,
                    "visual_sign": draft.visual_sign,
                    "product_names": draft.product_names,
                }
            )
            records.append(
                ColloquialMappingRecord(
                    key=key,
                    aliases=[str(alias) for alias in compact.get("aliases", [])],
                    visual_sign=str(compact.get("visual_sign", "")),
                    product_names=[str(name) for name in compact.get("product_names", [])],
                    section_keys=list(draft.section_keys),
                )
            )
        return records


def _leaflet_path(section_key: str) -> tuple[str, str]:
    category, _, slug = section_key.removeprefix(ANKHANG_SECTION_PREFIX).rpartition(":")
    if not category or not slug:
        raise ValueError(f"An Khang section key is malformed: {section_key}")
    return category, slug


def _blocks_by_section(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["section_id"]), []).append(row)
    return grouped


def _formulary_blocks(
    section_key: str, rows: list[dict[str, Any]]
) -> list[BlockRecord]:
    blocks: list[BlockRecord] = []
    for row in rows:
        content_type = str(row.get("content_type") or "")
        is_table = content_type == "table"
        markdown = str((row.get("markdown") if is_table else row.get("text")) or "")
        if not markdown.strip():
            continue
        blocks.append(
            BlockRecord(
                kind=block_kind_for(section_key, content_type),
                markdown=markdown,
                start_page=_page(row.get("page_start")),
                end_page=_page(row.get("page_end")),
                table_key=str(row["table_id"]) if is_table and row.get("table_id") else None,
            )
        )
    return blocks


def leaflet_blocks(text: str) -> list[BlockRecord]:
    """P1 decision 4: paragraphs joined into prose blocks, long pipe tables kept apart."""
    blocks: list[BlockRecord] = []
    prose: list[str] = []
    for paragraph in (part.strip() for part in _PARAGRAPH_BREAK.split(text)):
        if not paragraph:
            continue
        if paragraph.startswith("|") and len(paragraph) > MAX_CHUNK_CHARS:
            if prose:
                blocks.append(
                    BlockRecord(kind=BlockKind.PROSE, markdown="\n\n".join(prose))
                )
                prose = []
            blocks.append(BlockRecord(kind=BlockKind.TABLE, markdown=paragraph))
        else:
            prose.append(paragraph)
    if prose:
        blocks.append(BlockRecord(kind=BlockKind.PROSE, markdown="\n\n".join(prose)))
    return blocks


def _page(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _glossary(path: Path) -> list[GlossaryEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Glossary must be a JSON list: {path}")
    return [GlossaryEntry.model_validate(entry) for entry in payload]


def _source_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    digests = {
        "source_pdf_sha256": str(manifest["source_pdf_sha256"]),
        "snapshot_id": str(manifest["snapshot_id"]),
        "snapshot_sha256": str(manifest["snapshot_sha256"]),
    }
    curated = manifest.get("curated_input_digests") or {}
    for name, digest in sorted(curated.items()):
        digests[f"curated_{name}_sha256"] = str(digest)
    return digests
```

Create `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError

from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import RAG_FINAL_DIR, RESOURCES_DIR

bundle_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Export, check and embed knowledge bundles.",
)


def invalid_bundle(exc: BundleValidationError) -> ValueError:
    return ValueError("knowledge bundle is invalid: " + "; ".join(exc.problems[:20]))


def export_command(
    *,
    output: Path,
    rag_final_dir: Path = RAG_FINAL_DIR,
    glossary: Path = RESOURCES_DIR / "term_glossary.json",
    mappings: Path = RESOURCES_DIR / "colloquial_mappings.json",
    force: bool = False,
) -> CommandResult:
    try:
        result = export_bundle(
            ExportRequest(
                rag_final_dir=rag_final_dir,
                glossary_path=glossary,
                mappings_path=mappings,
                output_dir=output,
                force=force,
            )
        )
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    return CommandResult(
        command="bundle export",
        status=CommandStatus.COMPLETE,
        artifact=output,
        details={
            "documents": result.manifest.document_count,
            "sections": result.manifest.section_count,
            "skipped_sections": list(result.skipped_sections),
        },
    )


@bundle_app.command("export")
def export(
    ctx: typer.Context,
    output: Annotated[
        Path, typer.Option("--output", file_okay=False, resolve_path=True)
    ],
    rag_final_dir: Annotated[Path, typer.Option("--rag-final-dir")] = RAG_FINAL_DIR,
    glossary: Annotated[Path, typer.Option("--glossary")] = RESOURCES_DIR
    / "term_glossary.json",
    mappings: Annotated[Path, typer.Option("--mappings")] = RESOURCES_DIR
    / "colloquial_mappings.json",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: export_command(
            output=output,
            rag_final_dir=rag_final_dir,
            glossary=glossary,
            mappings=mappings,
            force=force,
        ),
    )
```

In `seed-pipeline/src/seed_pipeline/cli/app.py` add the import next to the other command imports and register the group after `source`:

```python
from seed_pipeline.cli.commands.bundle import bundle_app
```

```python
app.add_typer(bundle_app, name="bundle")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/bundle/test_export.py tests/cli/test_bundle_command.py`
Expected: `7 passed`. If `read_bundle` raises `BundleValidationError`, print `exc.problems`: every problem names a file, line and field (spec §5.3) and points at the mapping rule to fix in `export.py`.

- [ ] **Step 6: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/bundle seed-pipeline/src/seed_pipeline/cli/commands/bundle.py seed-pipeline/src/seed_pipeline/cli/app.py seed-pipeline/tests/fixtures/rag_final_small seed-pipeline/tests/bundle/test_export.py seed-pipeline/tests/cli/test_bundle_command.py
git commit -m "feat(seed): export the corpus build as a knowledge bundle"
```

The commit message ends with the session attribution trailer.

---

### Task 5: Chunk parity check (`seed bundle parity`)

Spec §7.3: for every section, `chunk_section` output must equal the old `rag-final/chunks.jsonl` in `chunk_text`, pages, `hydrate_strategy`, `embedding_text`, `term_annotations` and `colloquial_mapping` (plus `chunk_index == ordinal`, which keeps evaluation gold labels valid). The real build is not on this machine (`data/heavy/processed/` is missing), so the real-data test is opt-in with a new `data` marker and the migration runbook (Task 11) runs the command; the synthetic fixture from Task 4 is checked on every run.

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/bundle/parity.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py` (add `parity`)
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py` (add `BUNDLES_DIR`, `DEFAULT_BUNDLE_DIR`, `MIGRATION_DIR` after `RUNTIME_PROFILE_DIR`)
- Modify: `seed-pipeline/pyproject.toml` (`[tool.pytest.ini_options]` markers and addopts)
- Create: `seed-pipeline/tests/bundle/test_parity.py`, `seed-pipeline/tests/bundle/test_parity_data.py`
- Modify: `seed-pipeline/tests/cli/test_bundle_command.py` (parity command tests)

**Interfaces:**
- Consumes (P1): `read_bundle`, `KnowledgeBundle`, `BundleValidationError`, `MAX_CHUNK_CHARS`, `ChunkDraft`, `hydrate_strategy_for(section) -> HydrateStrategy`; (Task 4) `iter_section_chunks`, `invalid_bundle`, fixture `tests/fixtures/rag_final_small`.
- Produces:
  - `seed_pipeline.bundle.parity`: `COMPARED_FIELDS`, `ParityMismatch(section_key: str, ordinal: int | None, field: str, old: object, new: object)`, `ParityReport(sections_checked, chunks_checked, mismatch_count, mismatches, max_recorded)` with `ok` and `to_dict()`, `legacy_chunk_view(chunk) -> dict[str, object]`, `draft_view(draft, hydrate_strategy: str) -> dict[str, object]`, `comparable_mapping(old: object, new: object) -> object`, `check_chunk_parity(bundle, old_chunks, *, max_chars=MAX_CHUNK_CHARS, max_recorded=50) -> ParityReport`.
  - `seed_pipeline.config.paths`: `BUNDLES_DIR = HEAVY_DATA_DIR / "bundles"`, `DEFAULT_BUNDLE_DIR = BUNDLES_DIR / "formulary"`, `MIGRATION_DIR = HEAVY_DATA_DIR / "migration"`.
  - CLI: `seed bundle parity --bundle DIR --old-chunks FILE [--report FILE] [--max-chars N]`, exit 0 when identical, exit 1 with the first mismatches otherwise.
  - pytest marker `data` (excluded by default, run with `-m data`).

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/bundle/test_parity.py`:

```python
import json
from pathlib import Path

from pharma_agent.domain.corpus.bundle import KnowledgeBundle, read_bundle

from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.bundle.parity import check_chunk_parity, legacy_chunk_view

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
HAPACOL = "brand:ankhang:thuoc:hapacol-250-dhg-11500"


def _bundle(tmp_path: Path) -> KnowledgeBundle:
    export_bundle(
        ExportRequest(
            rag_final_dir=FIXTURE,
            glossary_path=FIXTURE / "term_glossary.json",
            mappings_path=FIXTURE / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    return read_bundle(tmp_path / "bundle")


def _old_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    ]


def test_backend_chunker_matches_the_old_fixture_build(tmp_path: Path) -> None:
    report = check_chunk_parity(_bundle(tmp_path), _old_chunks())

    assert report.ok, report.to_dict()
    assert (report.sections_checked, report.chunks_checked) == (5, 6)


def test_changed_embedding_text_is_reported(tmp_path: Path) -> None:
    old = _old_chunks()
    old[2]["embedding_text"] = old[2]["embedding_text"] + " thay đổi"

    report = check_chunk_parity(_bundle(tmp_path), old)

    assert not report.ok
    assert report.mismatch_count == 1
    mismatch = report.mismatches[0]
    assert (mismatch.section_key, mismatch.ordinal, mismatch.field) == (
        "drug:paracetamol:chong-chi-dinh",
        1,
        "embedding_text",
    )


def test_sections_missing_from_the_bundle_are_reported(tmp_path: Path) -> None:
    old = _old_chunks()
    old.append({**old[0], "section_id": "drug:ghost:lieu-dung"})

    report = check_chunk_parity(_bundle(tmp_path), old)

    assert [(m.section_key, m.field) for m in report.mismatches] == [
        ("drug:ghost:lieu-dung", "missing_section")
    ]


def test_legacy_view_maps_page_zero_to_none_and_keeps_the_old_mapping() -> None:
    view = legacy_chunk_view(
        {
            "section_id": HAPACOL,
            "chunk_index": 1,
            "chunk_text": "Hạ sốt",
            "embedding_text": "Hạ sốt",
            "hydrate_strategy": "full_section",
            "start_page": 0,
            "end_page": 0,
            "colloquial_mapping": {"product_names": ["Hapacol 250 DHG"]},
        }
    )

    assert (view["start_page"], view["end_page"]) == (None, None)
    assert view["colloquial_mapping"] == {"product_names": ["Hapacol 250 DHG"]}
    assert view["term_annotations"] == []
```

Create `seed-pipeline/tests/bundle/test_parity_data.py`:

```python
import pytest
from pharma_agent.domain.corpus.bundle import read_bundle

from seed_pipeline.bundle.parity import check_chunk_parity
from seed_pipeline.config.paths import DEFAULT_BUNDLE_DIR, MIGRATION_DIR
from seed_pipeline.evaluation.artifact_contracts import iter_jsonl_objects

pytestmark = pytest.mark.data


def test_backend_chunker_reproduces_the_pre_migration_build() -> None:
    report = check_chunk_parity(
        read_bundle(DEFAULT_BUNDLE_DIR),
        iter_jsonl_objects(MIGRATION_DIR / "rag-final-chunks.jsonl"),
    )

    assert report.chunks_checked > 0
    assert report.ok, report.to_dict()
```

Append to `seed-pipeline/tests/cli/test_bundle_command.py`:

```python
def _export(output: Path) -> None:
    result = runner.invoke(
        app,
        [
            "bundle",
            "export",
            "--output",
            str(output),
            "--rag-final-dir",
            str(FIXTURE),
            "--glossary",
            str(FIXTURE / "term_glossary.json"),
            "--mappings",
            str(FIXTURE / "colloquial_mappings.json"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_bundle_parity_command_passes_on_the_fixture(tmp_path: Path) -> None:
    _export(tmp_path / "bundle")
    report = tmp_path / "parity.json"

    result = runner.invoke(
        app,
        [
            "--json",
            "bundle",
            "parity",
            "--bundle",
            str(tmp_path / "bundle"),
            "--old-chunks",
            str(FIXTURE / "chunks.jsonl"),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["details"]["mismatches"] == 0
    assert json.loads(report.read_text("utf-8"))["ok"] is True


def test_bundle_parity_command_fails_on_a_changed_chunk(tmp_path: Path) -> None:
    _export(tmp_path / "bundle")
    lines = (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    changed = json.loads(lines[0])
    changed["chunk_text"] = "Nội dung khác"
    old_chunks = tmp_path / "chunks.jsonl"
    old_chunks.write_text(
        "\n".join([json.dumps(changed, ensure_ascii=False), *lines[1:]]) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "bundle",
            "parity",
            "--bundle",
            str(tmp_path / "bundle"),
            "--old-chunks",
            str(old_chunks),
        ],
    )

    assert result.exit_code == 1
    assert "mismatches=" in result.output
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/bundle/test_parity.py tests/cli/test_bundle_command.py`
Expected: collection error `ModuleNotFoundError: No module named 'seed_pipeline.bundle.parity'`.

- [ ] **Step 3: Implement the parity check**

Create `seed-pipeline/src/seed_pipeline/bundle/parity.py`:

```python
"""Compare `chunk_section` output with the pre-migration `rag-final/chunks.jsonl` (spec §7.3).

Only two normalizations are applied: an old page 0 means "no page", and an old leaflet
mapping without `key` (an uncurated leaflet, keyed by its slug in the bundle, P1) is
compared without that key.
Chunk identifiers are not compared; `chunk_index` must equal the draft ordinal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from pharma_agent.domain.corpus.bundle import KnowledgeBundle
from pharma_agent.domain.corpus.chunking import MAX_CHUNK_CHARS, ChunkDraft
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for

from seed_pipeline.bundle.chunks import iter_section_chunks

COMPARED_FIELDS = (
    "chunk_text",
    "start_page",
    "end_page",
    "hydrate_strategy",
    "embedding_text",
    "term_annotations",
    "colloquial_mapping",
)


@dataclass(frozen=True)
class ParityMismatch:
    section_key: str
    ordinal: int | None
    field: str
    old: object
    new: object


@dataclass
class ParityReport:
    sections_checked: int = 0
    chunks_checked: int = 0
    mismatch_count: int = 0
    mismatches: list[ParityMismatch] = field(default_factory=list)
    max_recorded: int = 50

    @property
    def ok(self) -> bool:
        return self.mismatch_count == 0

    def record(self, mismatch: ParityMismatch) -> None:
        self.mismatch_count += 1
        if len(self.mismatches) < self.max_recorded:
            self.mismatches.append(mismatch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sections_checked": self.sections_checked,
            "chunks_checked": self.chunks_checked,
            "mismatch_count": self.mismatch_count,
            "mismatches": [asdict(mismatch) for mismatch in self.mismatches],
        }


def legacy_chunk_view(chunk: Mapping[str, Any]) -> dict[str, object]:
    mapping = dict(chunk.get("colloquial_mapping") or {})
    return {
        "chunk_text": str(chunk["chunk_text"]),
        "start_page": _legacy_page(chunk.get("start_page")),
        "end_page": _legacy_page(chunk.get("end_page")),
        "hydrate_strategy": str(chunk.get("hydrate_strategy") or ""),
        "embedding_text": str(chunk["embedding_text"]),
        "term_annotations": list(chunk.get("term_annotations") or []),
        "colloquial_mapping": mapping,
    }


def draft_view(draft: ChunkDraft, hydrate_strategy: str) -> dict[str, object]:
    colloquial = (
        {}
        if draft.colloquial is None
        else {
            name: value
            for name, value in draft.colloquial.model_dump().items()
            if value not in ("", [])
        }
    )
    return {
        "chunk_text": draft.chunk_text,
        "start_page": draft.start_page,
        "end_page": draft.end_page,
        "hydrate_strategy": hydrate_strategy,
        "embedding_text": draft.embedding_text,
        "term_annotations": [term.model_dump() for term in draft.term_annotations],
        "colloquial_mapping": colloquial,
    }


def check_chunk_parity(
    bundle: KnowledgeBundle,
    old_chunks: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    max_recorded: int = 50,
) -> ParityReport:
    old_by_section: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in old_chunks:
        old_by_section[str(chunk["section_id"])].append(chunk)
    report = ParityReport(max_recorded=max_recorded)
    for item in iter_section_chunks(bundle, max_chars=max_chars):
        section_key = item.section.key
        old = sorted(
            old_by_section.pop(section_key, []),
            key=lambda chunk: int(chunk["chunk_index"]),
        )
        report.sections_checked += 1
        strategy = hydrate_strategy_for(item.section).value
        if len(old) != len(item.drafts):
            report.record(
                ParityMismatch(section_key, None, "chunk_count", len(old), len(item.drafts))
            )
        for chunk, draft in zip(old, item.drafts, strict=False):
            report.chunks_checked += 1
            old_ordinal = int(chunk["chunk_index"])
            if old_ordinal != draft.ordinal:
                report.record(
                    ParityMismatch(section_key, draft.ordinal, "ordinal", old_ordinal, draft.ordinal)
                )
            before = legacy_chunk_view(chunk)
            after = draft_view(draft, strategy)
            after["colloquial_mapping"] = comparable_mapping(
                before["colloquial_mapping"], after["colloquial_mapping"]
            )
            for name in COMPARED_FIELDS:
                if before[name] != after[name]:
                    report.record(
                        ParityMismatch(section_key, draft.ordinal, name, before[name], after[name])
                    )
    for section_key, chunks in sorted(old_by_section.items()):
        report.record(
            ParityMismatch(section_key, None, "missing_section", len(chunks), 0)
        )
    return report


def comparable_mapping(old: object, new: object) -> object:
    """Drop the bundle's slug key where the old build had no key for that leaflet."""
    if isinstance(old, dict) and isinstance(new, dict) and "key" not in old:
        return {name: value for name, value in new.items() if name != "key"}
    return new


def _legacy_page(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value
```

Add to `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py` (imports merged into the existing import block):

```python
from dataclasses import asdict

from pharma_agent.domain.corpus.bundle import read_bundle
from pharma_agent.domain.corpus.chunking import MAX_CHUNK_CHARS

from seed_pipeline.bundle.parity import check_chunk_parity
from seed_pipeline.evaluation.artifact_contracts import iter_jsonl_objects, write_json


def parity_command(
    *,
    bundle: Path,
    old_chunks: Path,
    report: Path | None = None,
    max_chars: int = MAX_CHUNK_CHARS,
) -> CommandResult:
    try:
        knowledge = read_bundle(bundle)
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    result = check_chunk_parity(
        knowledge, iter_jsonl_objects(old_chunks), max_chars=max_chars
    )
    if report is not None:
        write_json(report, result.to_dict())
    return CommandResult(
        command="bundle parity",
        status=CommandStatus.COMPLETE if result.ok else CommandStatus.FAILED,
        artifact=report,
        details={
            "sections": result.sections_checked,
            "chunks": result.chunks_checked,
            "mismatches": result.mismatch_count,
            "first_mismatches": [asdict(item) for item in result.mismatches[:5]],
        },
    )


@bundle_app.command("parity")
def parity(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Option("--bundle", file_okay=False, resolve_path=True)],
    old_chunks: Annotated[
        Path, typer.Option("--old-chunks", dir_okay=False, resolve_path=True)
    ],
    report: Annotated[Path | None, typer.Option("--report", dir_okay=False)] = None,
    max_chars: Annotated[int, typer.Option("--max-chars")] = MAX_CHUNK_CHARS,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: parity_command(
            bundle=bundle, old_chunks=old_chunks, report=report, max_chars=max_chars
        ),
    )
```

In `seed-pipeline/src/seed_pipeline/config/paths.py`, after `RUNTIME_PROFILE_DIR = ...`:

```python
BUNDLES_DIR = HEAVY_DATA_DIR / "bundles"
DEFAULT_BUNDLE_DIR = BUNDLES_DIR / "formulary"
MIGRATION_DIR = HEAVY_DATA_DIR / "migration"
```

In `seed-pipeline/pyproject.toml`, `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: needs Docker (run with -m integration)",
    "data: needs the full build under data/heavy (run with -m data)",
]
addopts = ["--import-mode=importlib", "-m", "not integration and not data"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/bundle tests/cli/test_bundle_command.py`
Expected: all pass (`test_parity_data.py` is deselected). A failure of `test_backend_chunker_matches_the_old_fixture_build` means the P1 chunker port differs from the old algorithm; the mismatch list names the section, ordinal and field; fix the port in P1's `chunking.py`/`enrichment.py`, never the fixture.

- [ ] **Step 5: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green. The real-data check is documented in Task 11 and runs with `uv run pytest -q -m data tests/bundle/test_parity_data.py`.

- [ ] **Step 6: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/bundle/parity.py seed-pipeline/src/seed_pipeline/cli/commands/bundle.py seed-pipeline/src/seed_pipeline/config/paths.py seed-pipeline/pyproject.toml seed-pipeline/tests/bundle/test_parity.py seed-pipeline/tests/bundle/test_parity_data.py seed-pipeline/tests/cli/test_bundle_command.py
git commit -m "feat(seed): check backend chunk parity against the old build"
```

The commit message ends with the session attribution trailer.

---

### Task 6: Text embedding cache and the text-only Kaggle embedding stage

The Kaggle `corpus-embed` stage keeps its orchestration (runtime profile benchmark, input dataset reconciliation, checkpoint by hash, resume) but its input becomes `embedding_inputs.jsonl` rows `{"embedding_text_sha256", "embedding_text"}` and its artifact a checksummed cache keyed by `(model, embedding_text_sha256, vector_dim)`. The kernel only receives text and never imports the backend. The old chunk-based `seed embed chunks` command and `vector_store/embedding_service.py` depended on the replaced stage contract and are removed here; the rest of `vector_store/` goes in Task 10.

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/embeddings/__init__.py`, `seed-pipeline/src/seed_pipeline/embeddings/text_cache.py`, `seed-pipeline/src/seed_pipeline/embeddings/service.py`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/corpus_embed.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/stages.py:96-146` (`CorpusEmbedStage`)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/kernels.py:83` (checkpoint filename), `seed-pipeline/src/seed_pipeline/integrations/kaggle/checkpoints.py:23` (artifact type)
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py` (add `COMPOSE_FILE`, `GGUF_ROOT`, `TEXT_EMBEDDING_CACHE_DIR`, `BUNDLE_EMBED_WORK_DIR`, `text_embedding_cache_path`)
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/embed.py` (whole file: only `embed queries` remains)
- Delete: `seed-pipeline/src/seed_pipeline/vector_store/embedding_service.py`, `seed-pipeline/tests/vector_store/test_embedding_service.py`
- Modify: `seed-pipeline/tests/cli/test_embed_command.py` (drop the two `embed chunks` tests), `seed-pipeline/tests/integrations/kaggle/test_stages.py:35-47`
- Create: `seed-pipeline/tests/embeddings/test_text_cache.py`

**Interfaces:**
- Consumes: `seed_pipeline.cache.jsonl_records` (`CacheRecordError`, `append_record(path, record, *, schema)`, `load_records(path) -> tuple[list[dict], bool]`, `merge_records(path, incoming, *, key, equivalent)`), `stream_map_ordered(items, resources, concurrency_per_resource, operation, deadline, on_completed=None, clock=...) -> ScheduledBatch` (`.stopped_early`), `LlamaCppClient(base_url, timeout=...)` with `embed(texts, model, expected_dimension)`, `LlamaCppComposeManager`, `resolve_server`, `require_model`/`ModelKind`, `kaggle_job_lock`, `kaggle_cache_lock`, `ensure_runtime_profile(...) -> ProfileResolution(profile, path, action)`, `run_kaggle_stage(...) -> PipelineResult(job, completion, actions, artifact_path, run_count)`, worker runtime helpers (`identity_from_config`, `resolve_input_file`, `resolve_optional_input_file`, `managed_model_servers`, `worker_deadline`, `artifact_from_output`), `RuntimeTelemetry`.
- Produces:
  - `seed_pipeline.embeddings.text_cache` (worker-safe): `CACHE_SCHEMA = "text-embedding-v1"`, `TextEmbeddingError(RuntimeError)`, `EmbeddingInput(embedding_text_sha256: str, embedding_text: str)`, `text_sha256(text) -> str`, `write_embedding_inputs(inputs, path) -> int`, `read_embedding_inputs(path) -> list[EmbeddingInput]`, `cache_record_key(record) -> tuple[str, str, int]`, `TextEmbeddingCache(path, *, model, model_sha256, vector_dim)` with `get`, `set`, `missing`, `digests`, `records_for`, `__len__`, `EmbeddingClient` protocol, `EmbedRun(total, cached, embedded, stopped_early)`, `embed_missing(inputs, cache, clients, *, batch_size, concurrency=1, deadline=inf, clock=time.monotonic) -> EmbedRun`.
  - `seed_pipeline.embeddings.service`: `TextEmbeddingRequest(inputs_path, model, cache_path, force, dry_run, budget_seconds, request_timeout_seconds, kaggle_account=None)`, `TextEmbeddingResult(cache_path, actions, incomplete=False)`, `TextEmbeddingBackend` protocol, `open_text_cache(path, model) -> TextEmbeddingCache`, `LocalTextEmbeddingBackend(*, compose_file=COMPOSE_FILE, gguf_root=GGUF_ROOT, server_mode="compose", server_urls=())`, `KaggleTextEmbeddingBackend`.
  - `seed_pipeline.config.paths`: `COMPOSE_FILE`, `GGUF_ROOT`, `TEXT_EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / "text_embeddings"`, `BUNDLE_EMBED_WORK_DIR = WORK_DIR / "bundle-embed"`, `text_embedding_cache_path(model) -> Path`.
  - `CorpusEmbedStage.contract_version == 3`, `data_filename == "text_embeddings.jsonl"`, artifact type `text_embedding_cache`.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/embeddings/test_text_cache.py`:

```python
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.embeddings.service import (
    KaggleTextEmbeddingBackend,
    LocalTextEmbeddingBackend,
    TextEmbeddingRequest,
    open_text_cache,
)
from seed_pipeline.embeddings.text_cache import (
    EmbeddingInput,
    TextEmbeddingError,
    embed_missing,
    read_embedding_inputs,
    text_sha256,
    write_embedding_inputs,
)
from seed_pipeline.integrations.kaggle import auto_profile
from seed_pipeline.integrations.kaggle import service as kaggle_service
from seed_pipeline.runtime.catalog import require_model

MODEL = "qwen3-embedding:4b-fp16"
DIMS = require_model(MODEL).vector_dimension or 0


def _input(text: str) -> EmbeddingInput:
    return EmbeddingInput(embedding_text_sha256=text_sha256(text), embedding_text=text)


def _vector(seed: float) -> list[float]:
    return [seed] * DIMS


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(
        self, texts: list[str], model: str, expected_dimension: int
    ) -> list[list[float]]:
        assert model == MODEL
        assert expected_dimension == DIMS
        self.calls.append(list(texts))
        return [_vector(float(len(text))) for text in texts]


def test_inputs_file_is_unique_sorted_and_verified(tmp_path: Path) -> None:
    path = tmp_path / "embedding_inputs.jsonl"

    count = write_embedding_inputs([_input("b"), _input("a"), _input("b")], path)

    assert count == 2
    rows = read_embedding_inputs(path)
    assert [row.embedding_text_sha256 for row in rows] == sorted(
        [text_sha256("a"), text_sha256("b")]
    )
    tampered = json.loads(path.read_text("utf-8").splitlines()[0])
    tampered["embedding_text"] = "khác"
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(TextEmbeddingError, match="sha256 mismatch"):
        read_embedding_inputs(path)


def test_cache_persists_vectors_per_model(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = open_text_cache(path, MODEL)
    cache.set(text_sha256("a"), _vector(0.5))

    reloaded = open_text_cache(path, MODEL)

    assert reloaded.get(text_sha256("a")) == _vector(0.5)
    assert reloaded.missing([_input("a"), _input("b")]) == [_input("b")]
    assert len(reloaded.records_for([text_sha256("a")])) == 1
    with pytest.raises(TextEmbeddingError, match="dimension"):
        cache.set(text_sha256("b"), [0.1])


def test_embed_missing_only_embeds_new_texts(tmp_path: Path) -> None:
    cache = open_text_cache(tmp_path / "cache.jsonl", MODEL)
    cache.set(text_sha256("đã có"), _vector(1.0))
    client = RecordingClient()
    inputs = [_input("đã có"), _input("mới một"), _input("mới hai")]

    first = embed_missing(inputs, cache, [client], batch_size=1)
    second = embed_missing(inputs, cache, [client], batch_size=1)

    assert (first.total, first.cached, first.embedded, first.stopped_early) == (3, 1, 2, False)
    assert sorted(text for call in client.calls for text in call) == ["mới hai", "mới một"]
    assert (second.cached, second.embedded) == (3, 0)
    assert cache.get(text_sha256("mới một")) == _vector(float(len("mới một")))


def test_local_backend_does_not_start_a_server_when_cache_is_complete(
    tmp_path: Path,
) -> None:
    inputs_path = tmp_path / "embedding_inputs.jsonl"
    write_embedding_inputs([_input("a")], inputs_path)
    cache_path = tmp_path / "cache.jsonl"
    open_text_cache(cache_path, MODEL).set(text_sha256("a"), _vector(0.1))
    backend = LocalTextEmbeddingBackend(
        compose_file=tmp_path / "missing-compose.yml", gguf_root=tmp_path
    )

    result = backend.run(
        TextEmbeddingRequest(
            inputs_path=inputs_path,
            model=MODEL,
            cache_path=cache_path,
            force=False,
            dry_run=False,
            budget_seconds=60,
            request_timeout_seconds=5.0,
        )
    )

    assert result == type(result)(cache_path, ("cached",), False)


def test_kaggle_backend_propagates_account_and_merges_remote_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs_path = tmp_path / "embedding_inputs.jsonl"
    write_embedding_inputs([_input("a")], inputs_path)
    remote_path = tmp_path / "remote" / "text_embeddings.jsonl"
    open_text_cache(remote_path, MODEL).set(text_sha256("a"), _vector(0.25))
    selected = require_model(MODEL).embedding_search_space
    assert selected is not None
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        auto_profile,
        "ensure_runtime_profile",
        lambda **kwargs: (
            seen.append(("profile", kwargs["kaggle_account"]))
            or SimpleNamespace(
                profile=SimpleNamespace(selected=selected.corpus.candidates[0]),
                action="reuse",
            )
        ),
    )
    monkeypatch.setattr(
        kaggle_service,
        "run_kaggle_stage",
        lambda **kwargs: (
            seen.append(("stage", kwargs["kaggle_account"]))
            or SimpleNamespace(
                artifact_path=remote_path, completion=Completion(1, 1, 0), actions=()
            )
        ),
    )
    cache_path = tmp_path / "cache.jsonl"

    result = KaggleTextEmbeddingBackend().run(
        TextEmbeddingRequest(
            inputs_path=inputs_path,
            model=MODEL,
            cache_path=cache_path,
            force=False,
            dry_run=False,
            budget_seconds=60,
            request_timeout_seconds=5.0,
            kaggle_account="acc2",
        )
    )

    assert not result.incomplete
    assert seen == [("profile", "acc2"), ("stage", "acc2")]
    assert open_text_cache(cache_path, MODEL).get(text_sha256("a")) == _vector(0.25)
```

In `seed-pipeline/tests/integrations/kaggle/test_stages.py`, replace `test_corpus_embed_builds_version_2_single_file_bundle` with:

```python
def test_corpus_embed_builds_version_3_text_input_bundle(tmp_path):
    source = tmp_path / "embedding_inputs.jsonl"
    source.write_text(
        json.dumps({"embedding_text_sha256": "0" * 64, "embedding_text": "x"}) + "\n",
        encoding="utf-8",
    )

    job = CorpusEmbedStage().build_job(
        _request(StageName.CORPUS_EMBED, "qwen3-embedding:4b-fp16", source)
    )

    assert job.contract_version == 3
    assert job.data_filename == "text_embeddings.jsonl"
    assert job.local_cache_path.name == "text_embeddings.jsonl"
    assert job.identity.payload["input_sha256"] == job.input_bundle.sha256
    assert set(job.input_bundle.descriptors()) == {"input"}
    assert "input_path" not in job.worker_config
```

In `seed-pipeline/tests/cli/test_embed_command.py`, delete `test_corpus_embedding_benchmark_rejects_local_backend` and `test_chunk_embedding_passes_kaggle_account_to_request`, and the now unused `import seed_pipeline.cli.commands.embed as embed_command` if nothing else uses it (the query test uses it: keep it).

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/embeddings tests/integrations/kaggle/test_stages.py`
Expected: collection error `ModuleNotFoundError: No module named 'seed_pipeline.embeddings'`; `test_corpus_embed_builds_version_3_text_input_bundle` fails with `assert 2 == 3`.

- [ ] **Step 3: Add the paths**

In `seed-pipeline/src/seed_pipeline/config/paths.py`, after `VECTOR_EMBEDDING_CACHE_DIR = ...`:

```python
TEXT_EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / "text_embeddings"
```

After `RUNTIME_PROFILE_DIR = ...` (next to the Task 5 constants):

```python
BUNDLE_EMBED_WORK_DIR = WORK_DIR / "bundle-embed"
COMPOSE_FILE = PROJECT_ROOT.parent / "docker-compose.yml"
GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"
```

After `query_embedding_cache_path`:

```python
def text_embedding_cache_path(model: str) -> Path:
    return TEXT_EMBEDDING_CACHE_DIR / f"{require_model(model).slug}.jsonl"
```

- [ ] **Step 4: Implement the worker-safe cache**

Create `seed-pipeline/src/seed_pipeline/embeddings/__init__.py`:

```python
"""Text embeddings keyed by sha256(embedding_text). Imported by Kaggle workers: no config.paths, no backend."""
```

Create `seed-pipeline/src/seed_pipeline/embeddings/text_cache.py`:

```python
"""Checksummed JSONL cache of text embeddings keyed by (model, embedding_text_sha256, vector_dim)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from seed_pipeline.cache.jsonl_records import (
    CacheRecordError,
    append_record,
    load_records,
)
from seed_pipeline.integrations.kaggle.workers.scheduling import stream_map_ordered

CACHE_SCHEMA = "text-embedding-v1"


class TextEmbeddingError(RuntimeError):
    """Raised when embedding inputs or the text embedding cache are invalid."""


@dataclass(frozen=True)
class EmbeddingInput:
    embedding_text_sha256: str
    embedding_text: str


@dataclass(frozen=True)
class EmbedRun:
    total: int
    cached: int
    embedded: int
    stopped_early: bool


class EmbeddingClient(Protocol):
    def embed(
        self, texts: list[str], model: str, expected_dimension: int
    ) -> list[list[float]]: ...


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_embedding_inputs(inputs: Iterable[EmbeddingInput], path: Path) -> int:
    unique = {item.embedding_text_sha256: item for item in inputs}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for digest in sorted(unique):
            row = {
                "embedding_text_sha256": digest,
                "embedding_text": unique[digest].embedding_text,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(unique)


def read_embedding_inputs(path: Path) -> list[EmbeddingInput]:
    inputs: list[EmbeddingInput] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row["embedding_text"])
            digest = str(row["embedding_text_sha256"])
            if text_sha256(text) != digest:
                raise TextEmbeddingError(
                    f"embedding_text_sha256 mismatch at {path}:{line_number}"
                )
            inputs.append(EmbeddingInput(digest, text))
    return inputs


def cache_record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["model"]),
        str(record["embedding_text_sha256"]),
        int(record["vector_dim"]),
    )


def _vector(value: object, vector_dim: int) -> list[float]:
    if not isinstance(value, list) or len(value) != vector_dim:
        raise TextEmbeddingError(f"embedding must have dimension {vector_dim}")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise TextEmbeddingError("embedding values must be finite numbers")
    return vector


class TextEmbeddingCache:
    def __init__(
        self, path: Path, *, model: str, model_sha256: str, vector_dim: int
    ) -> None:
        self.path = Path(path)
        self.model = model
        self.model_sha256 = model_sha256
        self.vector_dim = vector_dim
        self._records: dict[str, dict[str, Any]] = {}
        try:
            records, _ = load_records(self.path)
        except CacheRecordError as exc:
            raise TextEmbeddingError(str(exc)) from exc
        for record in records:
            if cache_record_key(record)[0::2] != (model, vector_dim):
                continue
            if record.get("model_sha256") != model_sha256:
                raise TextEmbeddingError(
                    f"Model digest mismatch in {self.path} for "
                    f"{record.get('embedding_text_sha256')}"
                )
            _vector(record.get("embedding"), vector_dim)
            self._records[str(record["embedding_text_sha256"])] = record

    def __len__(self) -> int:
        return len(self._records)

    def digests(self) -> list[str]:
        return sorted(self._records)

    def get(self, digest: str) -> list[float] | None:
        record = self._records.get(digest)
        return None if record is None else _vector(record["embedding"], self.vector_dim)

    def set(self, digest: str, embedding: Sequence[float]) -> None:
        vector = _vector(list(embedding), self.vector_dim)
        record = append_record(
            self.path,
            {
                "model": self.model,
                "model_sha256": self.model_sha256,
                "embedding_text_sha256": digest,
                "vector_dim": self.vector_dim,
                "embedding": vector,
                "created_at": datetime.now(UTC).isoformat(),
            },
            schema=CACHE_SCHEMA,
        )
        self._records[digest] = record

    def missing(self, inputs: Iterable[EmbeddingInput]) -> list[EmbeddingInput]:
        seen: set[str] = set()
        missing: list[EmbeddingInput] = []
        for item in inputs:
            digest = item.embedding_text_sha256
            if digest in seen or digest in self._records:
                continue
            seen.add(digest)
            missing.append(item)
        return missing

    def records_for(self, digests: Iterable[str]) -> list[dict[str, Any]]:
        return [self._records[digest] for digest in digests if digest in self._records]


def embed_missing(
    inputs: Sequence[EmbeddingInput],
    cache: TextEmbeddingCache,
    clients: Sequence[EmbeddingClient],
    *,
    batch_size: int,
    concurrency: int = 1,
    deadline: float = math.inf,
    clock: Callable[[], float] = time.monotonic,
) -> EmbedRun:
    total = len({item.embedding_text_sha256 for item in inputs})
    missing = cache.missing(inputs)
    cached = total - len(missing)
    if not missing:
        return EmbedRun(total=total, cached=cached, embedded=0, stopped_early=False)
    if not clients:
        raise TextEmbeddingError("embedding requires at least one model server")
    size = max(1, batch_size)
    batches = [missing[start : start + size] for start in range(0, len(missing), size)]
    embedded = 0

    async def operation(
        client: EmbeddingClient, _index: int, batch: list[EmbeddingInput]
    ) -> tuple[list[EmbeddingInput], list[list[float]]]:
        vectors = await asyncio.to_thread(
            client.embed,
            [item.embedding_text for item in batch],
            cache.model,
            cache.vector_dim,
        )
        return batch, vectors

    def on_completed(
        completed: list[tuple[int, tuple[list[EmbeddingInput], list[list[float]]]]],
    ) -> None:
        nonlocal embedded
        for _index, (batch, vectors) in completed:
            for item, vector in zip(batch, vectors, strict=True):
                cache.set(item.embedding_text_sha256, vector)
            embedded += len(batch)
        print(
            f"Embedding progress: {cached + embedded}/{total}, "
            f"cached={cached}, embedded={embedded}",
            flush=True,
        )

    scheduled = asyncio.run(
        stream_map_ordered(
            batches,
            list(clients),
            max(1, concurrency),
            operation,
            deadline,
            on_completed=on_completed,
            clock=clock,
        )
    )
    return EmbedRun(
        total=total,
        cached=cached,
        embedded=embedded,
        stopped_early=scheduled.stopped_early,
    )
```

- [ ] **Step 5: Implement the local and Kaggle backends**

Create `seed-pipeline/src/seed_pipeline/embeddings/service.py`:

```python
"""Run text embedding for bundle inputs locally (llama.cpp) or on Kaggle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from seed_pipeline.cache.jsonl_records import merge_records
from seed_pipeline.config.paths import COMPOSE_FILE, GGUF_ROOT, WORK_DIR
from seed_pipeline.embeddings.text_cache import (
    TextEmbeddingCache,
    cache_record_key,
    embed_missing,
    read_embedding_inputs,
)
from seed_pipeline.integrations.kaggle.job_lock import kaggle_cache_lock, kaggle_job_lock
from seed_pipeline.runtime.catalog import ModelKind, require_model
from seed_pipeline.runtime.client import LlamaCppClient
from seed_pipeline.runtime.compose import LlamaCppComposeManager, resolve_server


@dataclass(frozen=True)
class TextEmbeddingRequest:
    inputs_path: Path
    model: str
    cache_path: Path
    force: bool
    dry_run: bool
    budget_seconds: int
    request_timeout_seconds: float
    kaggle_account: str | None = None


@dataclass(frozen=True)
class TextEmbeddingResult:
    cache_path: Path | None
    actions: tuple[str, ...]
    incomplete: bool = False


class TextEmbeddingBackend(Protocol):
    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult: ...


def open_text_cache(path: Path, model: str) -> TextEmbeddingCache:
    spec = require_model(model)
    if spec.kind is not ModelKind.EMBEDDING or spec.vector_dimension is None:
        raise ValueError(f"--model must select an embedding model: {model}")
    return TextEmbeddingCache(
        path, model=model, model_sha256=spec.sha256, vector_dim=spec.vector_dimension
    )


class LocalTextEmbeddingBackend:
    def __init__(
        self,
        *,
        compose_file: Path = COMPOSE_FILE,
        gguf_root: Path = GGUF_ROOT,
        server_mode: str = "compose",
        server_urls: Sequence[str] = (),
    ) -> None:
        self.compose_file = compose_file
        self.gguf_root = gguf_root
        self.server_mode = server_mode
        self.server_urls = list(server_urls)

    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        spec = require_model(request.model)
        if request.dry_run:
            open_text_cache(request.cache_path, request.model)
            return TextEmbeddingResult(None, ("dry-run",))
        if request.force:
            request.cache_path.unlink(missing_ok=True)
        cache = open_text_cache(request.cache_path, request.model)
        inputs = read_embedding_inputs(request.inputs_path)
        if not cache.missing(inputs):
            return TextEmbeddingResult(request.cache_path, ("cached",))
        endpoints = resolve_server(
            self.server_mode,
            self.server_urls,
            spec,
            LlamaCppComposeManager(self.compose_file),
            self.gguf_root,
        )
        run = embed_missing(
            inputs,
            cache,
            [
                LlamaCppClient(endpoint, timeout=request.request_timeout_seconds)
                for endpoint in endpoints
            ],
            batch_size=spec.local_request_batch_size,
        )
        return TextEmbeddingResult(
            request.cache_path,
            (f"cached={run.cached}", f"embedded={run.embedded}"),
            incomplete=run.stopped_early,
        )


class KaggleTextEmbeddingBackend:
    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        with kaggle_job_lock(request.cache_path):
            return self._run_unlocked(request)

    @staticmethod
    def _run_unlocked(request: TextEmbeddingRequest) -> TextEmbeddingResult:
        from seed_pipeline.integrations.kaggle.auto_profile import (
            ensure_runtime_profile,
        )
        from seed_pipeline.integrations.kaggle.models import StageName
        from seed_pipeline.integrations.kaggle.service import run_kaggle_stage

        spec = require_model(request.model)
        resolution = ensure_runtime_profile(
            workload="corpus-embed",
            benchmark_stage=StageName.CORPUS_EMBED_BENCHMARK.value,
            model=request.model,
            input_path=request.inputs_path,
            gguf_root=GGUF_ROOT,
            budget_seconds=request.budget_seconds,
            dry_run=request.dry_run,
            force=request.force,
            kaggle_account=request.kaggle_account,
        )
        if resolution.profile is None:
            return TextEmbeddingResult(
                None, (f"profile={resolution.action}",), incomplete=True
            )
        result = run_kaggle_stage(
            stage=StageName.CORPUS_EMBED,
            model=request.model,
            input_path=request.inputs_path,
            output_dir=WORK_DIR / "kaggle-text-embeddings" / spec.slug,
            gguf_root=GGUF_ROOT,
            force=request.force,
            check_only=request.dry_run,
            budget_seconds=request.budget_seconds,
            runtime_profile=resolution.profile.selected,
            kaggle_account=request.kaggle_account,
        )
        actions = tuple(action.reason for action in result.actions)
        if result.artifact_path is None:
            return TextEmbeddingResult(
                None, actions, incomplete=not result.completion.is_complete
            )
        remote = open_text_cache(result.artifact_path, request.model)
        with kaggle_cache_lock(request.cache_path):
            merge_records(
                request.cache_path,
                remote.records_for(remote.digests()),
                key=cache_record_key,
                equivalent=lambda left, right: left["embedding"] == right["embedding"],
            )
        return TextEmbeddingResult(
            request.cache_path, actions, incomplete=not result.completion.is_complete
        )
```

- [ ] **Step 6: Switch the Kaggle stage, worker, kernel and checkpoint to the text contract**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/stages.py`, inside `CorpusEmbedStage`: set `contract_version: int = 3`, and in the returned `StageJob` use:

```python
            local_cache_path=output_dir / "text_embeddings.jsonl",
            data_filename="text_embeddings.jsonl",
```

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/kernels.py`, the checkpoint filename map entry becomes:

```python
                "corpus-embed": "text_embeddings.jsonl",
```

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/checkpoints.py:23`:

```python
    StageName.CORPUS_EMBED: "text_embedding_cache",
```

Replace `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/corpus_embed.py`:

```python
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from seed_pipeline.embeddings.text_cache import (
    TextEmbeddingCache,
    embed_missing,
    read_embedding_inputs,
)
from seed_pipeline.integrations.kaggle.models import CloudArtifact
from seed_pipeline.integrations.kaggle.workers.runtime import (
    artifact_from_output,
    identity_from_config,
    managed_model_servers,
    resolve_input_file,
    resolve_optional_input_file,
    worker_deadline,
)
from seed_pipeline.integrations.kaggle.workers.telemetry import RuntimeTelemetry
from seed_pipeline.runtime.client import LlamaCppClient

OUTPUT_FILENAME = "text_embeddings.jsonl"


def run_corpus_embed_worker(
    config: dict,
    *,
    command_executor: Callable[[dict], int] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> CloudArtifact:
    identity = identity_from_config(config)
    input_path = resolve_input_file(config, "input")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(config.get("output_path", output_dir / OUTPUT_FILENAME))
    model_name = str(config["model"])
    vector_dimension = int(config["vector_dimension"])
    model_sha256 = str(identity.payload.get("model_sha256") or "")
    runtime_overrides = config.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("runtime_overrides is required")
    telemetry = RuntimeTelemetry("corpus_embed", model_name, output_dir)
    checkpoint_path = resolve_optional_input_file(config, "checkpoint_filename")
    if (
        checkpoint_path is not None
        and checkpoint_path.is_file()
        and not output_path.exists()
    ):
        output_path.write_bytes(checkpoint_path.read_bytes())
    inputs = read_embedding_inputs(input_path)
    if command_executor is None:
        cache = TextEmbeddingCache(
            output_path,
            model=model_name,
            model_sha256=model_sha256,
            vector_dim=vector_dimension,
        )
        server_config = dict(config)
        server_config["runtime_overrides"] = runtime_overrides
        with managed_model_servers(server_config, telemetry=telemetry) as servers:
            embed_missing(
                inputs,
                cache,
                [LlamaCppClient(server.base_url) for server in servers],
                batch_size=int(runtime_overrides["request_batch_size"]),
                concurrency=int(runtime_overrides["concurrency"]),
                deadline=worker_deadline(config, clock),
                clock=clock,
            )
        return_code = 0
    else:
        return_code = command_executor(config)
    if return_code != 0 or not output_path.is_file():
        raise RuntimeError(f"Corpus embedding worker did not produce {output_path}")
    telemetry.close()
    runtime_summary = telemetry.summary()
    telemetry.write_report()
    complete = len(
        TextEmbeddingCache(
            output_path,
            model=model_name,
            model_sha256=model_sha256,
            vector_dim=vector_dimension,
        ).records_for(item.embedding_text_sha256 for item in inputs)
    )
    return artifact_from_output(
        output_path,
        artifact_type="text_embedding_cache",
        identity=identity,
        total=len(inputs),
        complete=complete,
        runtime_summary=runtime_summary,
    )


def main() -> int:
    config = json.loads(
        Path(os.environ["KAGGLE_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
    )
    run_corpus_embed_worker(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Remove the chunk embedding command**

Delete the old chunk backend and its test:

```bash
cd /home/andv/personal/thesis/seed-pipeline
git rm src/seed_pipeline/vector_store/embedding_service.py tests/vector_store/test_embedding_service.py
```

Replace `seed-pipeline/src/seed_pipeline/cli/commands/embed.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from seed_pipeline.cli.options import Backend
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import DEFAULT_EMBEDDING_MODEL
from seed_pipeline.config.paths import (
    PROCESSED_EVALUATION_DIR,
    query_embedding_cache_path,
)
from seed_pipeline.evaluation.query_embedding_service import (
    KaggleQueryEmbeddingBackend,
    LocalQueryEmbeddingBackend,
    QueryEmbeddingBackend,
    QueryEmbeddingRequest,
    QueryEmbeddingStageResult,
)
from seed_pipeline.runtime.catalog import require_model

embed_app = typer.Typer(no_args_is_help=True, add_completion=False)


@embed_app.command("queries")
def embed_queries(
    ctx: typer.Context,
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    evaluation: Annotated[Path, typer.Option("--evaluation")] = PROCESSED_EVALUATION_DIR
    / "section_retrieval_eval.jsonl",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBEDDING_MODEL,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[int, typer.Option("--budget-seconds")] = 21_600,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = 900.0,
    kaggle_account: Annotated[str | None, typer.Option("--kaggle-account")] = None,
) -> None:
    require_model(model)
    resolved_output = output_dir or query_embedding_cache_path(model)
    run_handler(
        state_from_context(ctx),
        lambda: _query_embedding_result(
            query_backend(backend).run(
                QueryEmbeddingRequest(
                    evaluation_path=evaluation,
                    model=model,
                    output_dir=resolved_output,
                    force=force,
                    dry_run=dry_run,
                    budget_seconds=budget_seconds,
                    request_timeout_seconds=request_timeout_seconds,
                    kaggle_account=kaggle_account,
                )
            )
        ),
    )


def query_backend(backend: Backend) -> QueryEmbeddingBackend:
    if backend is Backend.LOCAL:
        return LocalQueryEmbeddingBackend()
    if backend is Backend.KAGGLE:
        return KaggleQueryEmbeddingBackend()
    raise ValueError(f"Unsupported embedding backend: {backend}")


def _query_embedding_result(result: QueryEmbeddingStageResult) -> CommandResult:
    return CommandResult(
        "embed queries",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.cache_path,
        {
            "actions": result.actions,
            "subset_sha256": result.subset_sha256,
            "benchmark_report": getattr(result, "benchmark_report", None),
            "benchmark_levels": getattr(result, "benchmark_levels", 0),
        },
    )
```

`vector_store/ingest_vectors.py` still re-exports `DEFAULT_COMPOSE_FILE`/`DEFAULT_GGUF_ROOT` for `evaluation/query_embedding_service.py` and `evaluation/rerank_service.py`; Task 10 moves those imports to `config.paths` and deletes the package.

- [ ] **Step 8: Run the tests**

Run: `uv run pytest -q tests/embeddings tests/integrations/kaggle tests/cli/test_embed_command.py`
Expected: all pass, including `test_worker_imports_from_source_bundle_without_local_workspace[...corpus_embed]` (the worker imports neither `seed_pipeline.config.paths` nor `pharma_agent`).

- [ ] **Step 9: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 10: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/embeddings seed-pipeline/src/seed_pipeline/integrations/kaggle seed-pipeline/src/seed_pipeline/config/paths.py seed-pipeline/src/seed_pipeline/cli/commands/embed.py seed-pipeline/tests/embeddings seed-pipeline/tests/integrations/kaggle/test_stages.py seed-pipeline/tests/cli/test_embed_command.py
git commit -m "feat(seed): embed text inputs by hash on Kaggle and locally"
```

(`git rm` in Step 7 already staged the deletions.) The commit message ends with the session attribution trailer.

---

### Task 7: `seed bundle embed`

`seed bundle embed` runs the backend `chunk_section` locally to collect unique `(embedding_text_sha256, embedding_text)` pairs, embeds the missing ones with the Task 6 backends, and writes `embeddings/<model_slug>.jsonl` plus an updated manifest through `write_bundle`. Bundle writes go through one helper that validates with `read_bundle` in a staging directory before replacing the target, which Task 4's export also switches to.

Memory note: `KnowledgeBundle.embeddings` holds `list[float]` vectors, so a full bundle with `qwen3-embedding:4b-fp16` (about 25k × 2560 values) needs roughly 2 GB of RAM while this command runs.

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/bundle/io.py`, `seed-pipeline/src/seed_pipeline/bundle/embed.py`
- Modify: `seed-pipeline/src/seed_pipeline/bundle/export.py` (use `write_validated_bundle`)
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py` (add `embed`)
- Create: `seed-pipeline/tests/bundle/test_embed.py`
- Modify: `seed-pipeline/tests/cli/test_bundle_command.py` (embed command test)

**Interfaces:**
- Consumes (P2): shared fixture `backend/tests/fixtures/knowledge_bundle_small/` (embeddings for `fake-embedding-4d`).
- Consumes (P1): `read_bundle`, `write_bundle`, `model_slug`, `KnowledgeBundle.embeddings: dict[str, dict[str, list[float]]]`, `BundleManifest.embeddings: list[BundleEmbeddingFile(model, dims, file)]`, `BundleValidationError`; (Task 4) `iter_section_chunks`, `invalid_bundle`; (Task 6) `EmbeddingInput`, `TextEmbeddingError`, `write_embedding_inputs`, `open_text_cache`, `TextEmbeddingBackend`, `TextEmbeddingRequest`, `TextEmbeddingResult`, `LocalTextEmbeddingBackend`, `KaggleTextEmbeddingBackend`, `BUNDLE_EMBED_WORK_DIR`, `text_embedding_cache_path`.
- Produces:
  - `seed_pipeline.bundle.io.write_validated_bundle(bundle: KnowledgeBundle, output_dir: Path) -> BundleManifest`.
  - `seed_pipeline.bundle.embed`: `BundleEmbedRequest(bundle_dir, model, force=False, dry_run=False, budget_seconds=21_600, request_timeout_seconds=900.0, kaggle_account=None, cache_path=None, work_dir=BUNDLE_EMBED_WORK_DIR)`, `BundleEmbedResult(manifest: BundleManifest | None, inputs: int, vectors: int, embeddings_file: str | None, actions: tuple[str, ...], incomplete: bool)`, `collect_embedding_inputs(bundle) -> list[EmbeddingInput]`, `embed_bundle(request, backend: TextEmbeddingBackend) -> BundleEmbedResult`.
  - CLI: `seed bundle embed --bundle DIR --backend kaggle|local --model MODEL [--cache FILE] [--work-dir DIR] [--force] [--dry-run] [--budget-seconds N] [--request-timeout-seconds S] [--kaggle-account accN]`; exit 3 when the embedding stage is resumable-incomplete.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/bundle/test_embed.py`:

```python
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BundleEmbeddingFile,
    KnowledgeBundle,
    model_slug,
    read_bundle,
)

from seed_pipeline.bundle.embed import (
    BundleEmbedRequest,
    collect_embedding_inputs,
    embed_bundle,
)
from seed_pipeline.bundle.export import ExportRequest, export_bundle
from seed_pipeline.embeddings.service import (
    TextEmbeddingRequest,
    TextEmbeddingResult,
    open_text_cache,
)
from seed_pipeline.embeddings.text_cache import read_embedding_inputs, text_sha256
from seed_pipeline.runtime.catalog import require_model

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
MODEL = "qwen3-embedding:4b-fp16"
DIMS = require_model(MODEL).vector_dimension or 0


def fake_vector(text: str) -> list[float]:
    return [float(len(text) % 7)] * DIMS


class CacheFillingBackend:
    def __init__(self, *, incomplete: bool = False) -> None:
        self.incomplete = incomplete
        self.requests: list[TextEmbeddingRequest] = []

    def run(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.requests.append(request)
        if self.incomplete:
            return TextEmbeddingResult(None, ("profile=pending",), incomplete=True)
        cache = open_text_cache(request.cache_path, request.model)
        for item in read_embedding_inputs(request.inputs_path):
            if cache.get(item.embedding_text_sha256) is None:
                cache.set(item.embedding_text_sha256, fake_vector(item.embedding_text))
        return TextEmbeddingResult(request.cache_path, ("fake",))


def _bundle_dir(tmp_path: Path) -> Path:
    export_bundle(
        ExportRequest(
            rag_final_dir=FIXTURE,
            glossary_path=FIXTURE / "term_glossary.json",
            mappings_path=FIXTURE / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    return tmp_path / "bundle"


def _request(tmp_path: Path, bundle_dir: Path) -> BundleEmbedRequest:
    return BundleEmbedRequest(
        bundle_dir=bundle_dir,
        model=MODEL,
        cache_path=tmp_path / "cache.jsonl",
        work_dir=tmp_path / "work",
    )


def test_collect_embedding_inputs_is_unique_and_hashed(tmp_path: Path) -> None:
    bundle: KnowledgeBundle = read_bundle(_bundle_dir(tmp_path))

    inputs = collect_embedding_inputs(bundle)

    assert len(inputs) == 6
    assert [item.embedding_text_sha256 for item in inputs] == sorted(
        item.embedding_text_sha256 for item in inputs
    )
    assert all(
        item.embedding_text_sha256 == text_sha256(item.embedding_text) for item in inputs
    )


def test_embed_bundle_writes_embeddings_file_and_manifest(tmp_path: Path) -> None:
    bundle_dir = _bundle_dir(tmp_path)
    backend = CacheFillingBackend()

    result = embed_bundle(_request(tmp_path, bundle_dir), backend)

    assert not result.incomplete
    assert (result.inputs, result.vectors) == (6, 6)
    expected_file = f"embeddings/{model_slug(MODEL)}.jsonl"
    assert result.embeddings_file == expected_file
    assert (bundle_dir / expected_file).is_file()
    assert result.manifest is not None
    assert result.manifest.embeddings == [
        BundleEmbeddingFile(model=MODEL, dims=DIMS, file=expected_file)
    ]
    reloaded = read_bundle(bundle_dir)
    for item in collect_embedding_inputs(reloaded):
        assert reloaded.embeddings[MODEL][item.embedding_text_sha256] == fake_vector(
            item.embedding_text
        )
    assert backend.requests[0].inputs_path == (
        tmp_path / "work" / require_model(MODEL).slug / "embedding_inputs.jsonl"
    )


def test_incomplete_embedding_leaves_the_bundle_untouched(tmp_path: Path) -> None:
    bundle_dir = _bundle_dir(tmp_path)

    result = embed_bundle(
        _request(tmp_path, bundle_dir), CacheFillingBackend(incomplete=True)
    )

    assert result.incomplete
    assert result.manifest is None
    assert not (bundle_dir / "embeddings").exists()
    assert read_bundle(bundle_dir).embeddings == {}


BACKEND_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "tests"
    / "fixtures"
    / "knowledge_bundle_small"
)


def test_shared_backend_fixture_bundle_is_read_and_fully_embedded() -> None:
    bundle = read_bundle(BACKEND_FIXTURE)

    inputs = collect_embedding_inputs(bundle)

    assert inputs
    assert {item.embedding_text_sha256 for item in inputs} <= set(
        bundle.embeddings["fake-embedding-4d"]
    )
```

Append to `seed-pipeline/tests/cli/test_bundle_command.py`:

```python
def test_bundle_embed_command_uses_the_selected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import seed_pipeline.cli.commands.bundle as bundle_command
    from tests.bundle.test_embed import CacheFillingBackend

    _export(tmp_path / "bundle")
    created: list[CacheFillingBackend] = []

    def kaggle_backend() -> CacheFillingBackend:
        backend = CacheFillingBackend()
        created.append(backend)
        return backend

    monkeypatch.setattr(bundle_command, "KaggleTextEmbeddingBackend", kaggle_backend)

    result = runner.invoke(
        app,
        [
            "--json",
            "bundle",
            "embed",
            "--bundle",
            str(tmp_path / "bundle"),
            "--backend",
            "kaggle",
            "--model",
            "qwen3-embedding:4b-fp16",
            "--cache",
            str(tmp_path / "cache.jsonl"),
            "--work-dir",
            str(tmp_path / "work"),
            "--kaggle-account",
            "acc2",
        ],
    )

    assert result.exit_code == 0, result.output
    details = json.loads(result.output)["details"]
    assert (details["inputs"], details["vectors"]) == (6, 6)
    assert created[0].requests[0].kaggle_account == "acc2"
```

Add `import pytest` to the imports of `tests/cli/test_bundle_command.py`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/bundle/test_embed.py tests/cli/test_bundle_command.py`
Expected: collection error `ModuleNotFoundError: No module named 'seed_pipeline.bundle.embed'`.

- [ ] **Step 3: Add the validated writer and use it in the export**

Create `seed-pipeline/src/seed_pipeline/bundle/io.py`:

```python
"""Write knowledge bundles through a validated staging directory."""

import shutil
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BundleManifest,
    KnowledgeBundle,
    read_bundle,
    write_bundle,
)


def write_validated_bundle(bundle: KnowledgeBundle, output_dir: Path) -> BundleManifest:
    """Write to `.<name>.next`, validate with `read_bundle`, then replace `output_dir`."""
    output_dir = Path(output_dir)
    staging = output_dir.with_name(f".{output_dir.name}.next")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest = write_bundle(bundle, staging)
        read_bundle(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest
```

In `seed-pipeline/src/seed_pipeline/bundle/export.py`, replace the lines from `staging = output_dir.with_name(...)` to the `return` with:

```python
    manifest = write_validated_bundle(bundle, output_dir)
    return ExportResult(manifest=manifest, skipped_sections=tuple(skipped))
```

remove `import shutil` and the `read_bundle`, `write_bundle` names from the `pharma_agent.domain.corpus.bundle` import, and add:

```python
from seed_pipeline.bundle.io import write_validated_bundle
```

- [ ] **Step 4: Implement the bundle embedding**

Create `seed-pipeline/src/seed_pipeline/bundle/embed.py`:

```python
"""Pre-compute bundle embeddings keyed by sha256(embedding_text) and store them in the bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pharma_agent.domain.corpus.bundle import (
    BundleManifest,
    KnowledgeBundle,
    model_slug,
    read_bundle,
)

from seed_pipeline.bundle.chunks import iter_section_chunks
from seed_pipeline.bundle.io import write_validated_bundle
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from seed_pipeline.config.paths import BUNDLE_EMBED_WORK_DIR, text_embedding_cache_path
from seed_pipeline.embeddings.service import (
    TextEmbeddingBackend,
    TextEmbeddingRequest,
    open_text_cache,
)
from seed_pipeline.embeddings.text_cache import (
    EmbeddingInput,
    TextEmbeddingError,
    write_embedding_inputs,
)
from seed_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class BundleEmbedRequest:
    bundle_dir: Path
    model: str
    force: bool = False
    dry_run: bool = False
    budget_seconds: int = DEFAULT_BUDGET_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    kaggle_account: str | None = None
    cache_path: Path | None = None
    work_dir: Path = BUNDLE_EMBED_WORK_DIR


@dataclass(frozen=True)
class BundleEmbedResult:
    manifest: BundleManifest | None
    inputs: int
    vectors: int
    embeddings_file: str | None
    actions: tuple[str, ...]
    incomplete: bool


def collect_embedding_inputs(bundle: KnowledgeBundle) -> list[EmbeddingInput]:
    unique: dict[str, EmbeddingInput] = {}
    for item in iter_section_chunks(bundle):
        for draft in item.drafts:
            unique.setdefault(
                draft.embedding_text_sha256,
                EmbeddingInput(draft.embedding_text_sha256, draft.embedding_text),
            )
    return [unique[digest] for digest in sorted(unique)]


def embed_bundle(
    request: BundleEmbedRequest, backend: TextEmbeddingBackend
) -> BundleEmbedResult:
    spec = require_model(request.model)
    bundle = read_bundle(request.bundle_dir)
    inputs = collect_embedding_inputs(bundle)
    inputs_path = request.work_dir / spec.slug / "embedding_inputs.jsonl"
    write_embedding_inputs(inputs, inputs_path)
    cache_path = request.cache_path or text_embedding_cache_path(request.model)
    stage = backend.run(
        TextEmbeddingRequest(
            inputs_path=inputs_path,
            model=request.model,
            cache_path=cache_path,
            force=request.force,
            dry_run=request.dry_run,
            budget_seconds=request.budget_seconds,
            request_timeout_seconds=request.request_timeout_seconds,
            kaggle_account=request.kaggle_account,
        )
    )
    if request.dry_run or stage.incomplete or stage.cache_path is None:
        return BundleEmbedResult(
            manifest=None,
            inputs=len(inputs),
            vectors=0,
            embeddings_file=None,
            actions=stage.actions,
            incomplete=stage.incomplete,
        )
    cache = open_text_cache(stage.cache_path, request.model)
    vectors: dict[str, list[float]] = {}
    for item in inputs:
        vector = cache.get(item.embedding_text_sha256)
        if vector is None:
            raise TextEmbeddingError(
                f"Embedding cache {stage.cache_path} is missing {item.embedding_text_sha256}"
            )
        vectors[item.embedding_text_sha256] = vector
    updated = bundle.model_copy(
        update={"embeddings": {**bundle.embeddings, request.model: vectors}}
    )
    manifest = write_validated_bundle(updated, request.bundle_dir)
    return BundleEmbedResult(
        manifest=manifest,
        inputs=len(inputs),
        vectors=len(vectors),
        embeddings_file=f"embeddings/{model_slug(request.model)}.jsonl",
        actions=stage.actions,
        incomplete=False,
    )
```

Add to `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py` (imports merged into the existing block):

```python
from seed_pipeline.bundle.embed import BundleEmbedRequest, embed_bundle
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from seed_pipeline.config.enums import Backend
from seed_pipeline.config.paths import BUNDLE_EMBED_WORK_DIR
from seed_pipeline.embeddings.service import (
    KaggleTextEmbeddingBackend,
    LocalTextEmbeddingBackend,
    TextEmbeddingBackend,
)


def embed_command(
    *,
    bundle: Path,
    backend: Backend,
    model: str,
    cache: Path | None = None,
    work_dir: Path = BUNDLE_EMBED_WORK_DIR,
    force: bool = False,
    dry_run: bool = False,
    budget_seconds: int = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    kaggle_account: str | None = None,
) -> CommandResult:
    adapter: TextEmbeddingBackend = (
        LocalTextEmbeddingBackend()
        if backend is Backend.LOCAL
        else KaggleTextEmbeddingBackend()
    )
    try:
        result = embed_bundle(
            BundleEmbedRequest(
                bundle_dir=bundle,
                model=model,
                force=force,
                dry_run=dry_run,
                budget_seconds=budget_seconds,
                request_timeout_seconds=request_timeout_seconds,
                kaggle_account=kaggle_account,
                cache_path=cache,
                work_dir=work_dir,
            ),
            adapter,
        )
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    return CommandResult(
        command="bundle embed",
        status=CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        artifact=bundle,
        details={
            "inputs": result.inputs,
            "vectors": result.vectors,
            "embeddings_file": result.embeddings_file,
            "actions": list(result.actions),
        },
    )


@bundle_app.command("embed")
def embed(
    ctx: typer.Context,
    bundle: Annotated[Path, typer.Option("--bundle", file_okay=False, resolve_path=True)],
    backend: Annotated[Backend, typer.Option("--backend")],
    model: Annotated[str, typer.Option("--model")],
    cache: Annotated[Path | None, typer.Option("--cache", dir_okay=False)] = None,
    work_dir: Annotated[Path, typer.Option("--work-dir")] = BUNDLE_EMBED_WORK_DIR,
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
    run_handler(
        state_from_context(ctx),
        lambda: embed_command(
            bundle=bundle,
            backend=backend,
            model=model,
            cache=cache,
            work_dir=work_dir,
            force=force,
            dry_run=dry_run,
            budget_seconds=budget_seconds,
            request_timeout_seconds=request_timeout_seconds,
            kaggle_account=kaggle_account,
        ),
    )
```

The monkeypatched name in the CLI test is the module attribute `KaggleTextEmbeddingBackend`, which `embed_command` looks up at call time.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/bundle tests/cli/test_bundle_command.py`
Expected: all pass. P1's `write_bundle` builds `manifest.embeddings` from `bundle.embeddings` (sorted by model, `file = embeddings/<model_slug>.jsonl`, `dims` from the vectors) and rejects vectors of mixed length before writing anything.

- [ ] **Step 6: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/bundle seed-pipeline/src/seed_pipeline/cli/commands/bundle.py seed-pipeline/tests/bundle/test_embed.py seed-pipeline/tests/cli/test_bundle_command.py
git commit -m "feat(seed): add seed bundle embed"
```

The commit message ends with the session attribution trailer.

---

### Task 8: `seed evaluation build` reads chunks from the bundle

The evaluation dataset builders (`build_section_retrieval_eval.build_jsonl`, `patient_query_generation.build_patient_queries`) read chunk rows with the fields of the old unified contract (`chunk_id`, `section_id`, `chunk_index`, `chunk_text`, `chunk_role`, `chunk_content_type`, `hydrate_strategy`, `title`, `colloquial_mapping`...). Before `rag-final/chunks.jsonl` is removed, `seed evaluation build --bundle` derives the same rows from `chunk_section` output and writes them to a work file, so the 2 500-line builder stays unchanged and gold labels keep the old `chunk_id` format (`gold_chunk_label`).

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/bundle/evaluation_chunks.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/evaluation.py:1-89` (imports, `evaluation_build_command`, `evaluation_build`)
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py` (add `EVALUATION_CHUNKS_PATH`)
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/patient_query_generation.py:16`, `seed-pipeline/src/seed_pipeline/evaluation/build_section_retrieval_eval.py:35` (default chunks path)
- Create: `seed-pipeline/tests/bundle/test_evaluation_chunks.py`
- Modify: `seed-pipeline/tests/cli/test_evaluation_command.py` (append a build test)

**Interfaces:**
- Consumes: (P1) `read_bundle`, `BundleValidationError`, `BlockKind`, `DocumentKind`, `hydrate_strategy_for`; (Task 4) `iter_section_chunks`, `gold_chunk_label`, `invalid_bundle`, `export_bundle`; (Task 5) `draft_view`, `DEFAULT_BUNDLE_DIR`; `EvaluationBuildRequest(sections_path, chunks_path, output_dir, patient_query_count, evaluation_row_count)`, `EvaluationBuildResult(patient_queries_path, evaluation_path, patient_query_count, evaluation_row_count)`.
- Produces: `seed_pipeline.bundle.evaluation_chunks.CHUNK_ROLES`, `evaluation_chunk_rows(bundle) -> list[dict[str, Any]]`, `write_evaluation_chunks(bundle, path) -> int`; `seed_pipeline.config.paths.EVALUATION_CHUNKS_PATH = WORK_DIR / "evaluation-chunks" / "chunks.jsonl"`; CLI `seed evaluation build [--sections FILE] [--bundle DIR] [--chunks-output FILE] [--output-dir DIR] ...` (the `--chunks` option is removed).

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/bundle/test_evaluation_chunks.py`:

```python
import json
from pathlib import Path

from pharma_agent.domain.corpus.bundle import read_bundle

from seed_pipeline.bundle.evaluation_chunks import (
    evaluation_chunk_rows,
    write_evaluation_chunks,
)
from seed_pipeline.bundle.export import ExportRequest, export_bundle

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
FIELDS_READ_BY_THE_BUILDERS = (
    "chunk_id",
    "section_id",
    "chunk_index",
    "title",
    "section",
    "source",
    "content_type",
    "context_header",
    "context_path",
    "chunk_text",
    "embedding_text",
    "hydrate_strategy",
    "chunk_role",
    "chunk_content_type",
    "term_annotations",
)


def _bundle_dir(tmp_path: Path) -> Path:
    export_bundle(
        ExportRequest(
            rag_final_dir=FIXTURE,
            glossary_path=FIXTURE / "term_glossary.json",
            mappings_path=FIXTURE / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    return tmp_path / "bundle"


def test_rows_reproduce_the_old_unified_chunk_fields(tmp_path: Path) -> None:
    rows = evaluation_chunk_rows(read_bundle(_bundle_dir(tmp_path)))
    old = [
        json.loads(line)
        for line in (FIXTURE / "chunks.jsonl").read_text("utf-8").splitlines()
    ]

    by_id = {row["chunk_id"]: row for row in rows}
    assert sorted(by_id) == sorted(chunk["chunk_id"] for chunk in old)
    for chunk in old:
        row = by_id[chunk["chunk_id"]]
        for field in FIELDS_READ_BY_THE_BUILDERS:
            assert row[field] == chunk.get(field, []), (chunk["chunk_id"], field)
        assert row["table_id"] == chunk.get("table_id", "")


def test_write_evaluation_chunks_writes_one_row_per_chunk(tmp_path: Path) -> None:
    path = tmp_path / "work" / "chunks.jsonl"

    count = write_evaluation_chunks(read_bundle(_bundle_dir(tmp_path)), path)

    assert count == 6
    assert len(path.read_text("utf-8").splitlines()) == 6
```

Append to `seed-pipeline/tests/cli/test_evaluation_command.py`:

```python
def test_evaluation_build_derives_chunks_from_the_bundle(tmp_path, monkeypatch):
    from seed_pipeline.bundle.export import ExportRequest, export_bundle
    from seed_pipeline.evaluation.build_dataset import EvaluationBuildResult

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "rag_final_small"
    export_bundle(
        ExportRequest(
            rag_final_dir=fixture,
            glossary_path=fixture / "term_glossary.json",
            mappings_path=fixture / "colloquial_mappings.json",
            output_dir=tmp_path / "bundle",
        )
    )
    captured = {}

    def fake_build(request):
        captured["request"] = request
        return EvaluationBuildResult(
            tmp_path / "patient_queries.json", tmp_path / "evaluation.jsonl", 0, 0
        )

    monkeypatch.setattr(evaluation_command, "build_evaluation_dataset", fake_build)
    result = runner.invoke(
        app,
        [
            "--json",
            "evaluation",
            "build",
            "--sections",
            str(fixture / "sections.jsonl"),
            "--bundle",
            str(tmp_path / "bundle"),
            "--chunks-output",
            str(tmp_path / "chunks.jsonl"),
            "--output-dir",
            str(tmp_path / "evaluation"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].chunks_path == tmp_path / "chunks.jsonl"
    assert len((tmp_path / "chunks.jsonl").read_text("utf-8").splitlines()) == 6
    assert json.loads(result.stdout)["details"]["chunks"] == 6
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/bundle/test_evaluation_chunks.py tests/cli/test_evaluation_command.py`
Expected: collection error `ModuleNotFoundError: No module named 'seed_pipeline.bundle.evaluation_chunks'`; the CLI test fails with exit code 2 (`No such option: --bundle`).

- [ ] **Step 3: Implement the evaluation chunk rows**

Create `seed-pipeline/src/seed_pipeline/bundle/evaluation_chunks.py`:

```python
"""Chunk rows for the evaluation dataset builders, derived from the backend chunker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pharma_agent.domain.corpus.bundle import BlockKind, DocumentKind, KnowledgeBundle
from pharma_agent.domain.corpus.hydrate import hydrate_strategy_for

from seed_pipeline.bundle.chunks import gold_chunk_label, iter_section_chunks
from seed_pipeline.bundle.parity import draft_view

CHUNK_ROLES: dict[BlockKind, str] = {
    BlockKind.PROSE: "prose",
    BlockKind.TABLE: "table",
    BlockKind.INDEX_ENTRIES: "index_entry",
    BlockKind.LIST: "appendix_list",
}
CONTENT_TYPES: dict[DocumentKind, str] = {
    DocumentKind.DRUG_MONOGRAPH: "drug_monograph",
    DocumentKind.GENERAL_MONOGRAPH: "general_monograph",
    DocumentKind.LEAFLET: "brand_page",
}


def evaluation_chunk_rows(bundle: KnowledgeBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in iter_section_chunks(bundle):
        strategy = hydrate_strategy_for(item.section).value
        for draft in item.drafts:
            view = draft_view(draft, strategy)
            rows.append(
                {
                    "chunk_id": gold_chunk_label(draft.section_key, draft.ordinal),
                    "section_id": draft.section_key,
                    "chunk_index": draft.ordinal,
                    "title": item.document.title,
                    "section": item.section.heading,
                    "source": item.document.source.title,
                    "content_type": CONTENT_TYPES[item.document.kind],
                    "context_header": draft.context_header,
                    "context_path": list(item.section.context_path),
                    "chunk_text": draft.chunk_text,
                    "embedding_text": draft.embedding_text,
                    "hydrate_strategy": strategy,
                    "chunk_role": CHUNK_ROLES[draft.kind],
                    "chunk_content_type": (
                        "table" if draft.kind is BlockKind.TABLE else "paragraph"
                    ),
                    "table_id": draft.table_key or "",
                    "start_page": draft.start_page,
                    "end_page": draft.end_page,
                    "colloquial_mapping": view["colloquial_mapping"],
                    "term_annotations": view["term_annotations"],
                }
            )
    return rows


def write_evaluation_chunks(bundle: KnowledgeBundle, path: Path) -> int:
    rows = evaluation_chunk_rows(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)
```

In `seed-pipeline/src/seed_pipeline/config/paths.py`, after the Task 5/6 work constants:

```python
EVALUATION_CHUNKS_PATH = WORK_DIR / "evaluation-chunks" / "chunks.jsonl"
```

In `seed-pipeline/src/seed_pipeline/evaluation/patient_query_generation.py:16` and `seed-pipeline/src/seed_pipeline/evaluation/build_section_retrieval_eval.py:35`, replace the default with:

```python
DEFAULT_CHUNKS_PATH = EVALUATION_CHUNKS_PATH
```

and add `EVALUATION_CHUNKS_PATH` to each module's existing `from seed_pipeline.config.paths import (...)` block.

- [ ] **Step 4: Switch the CLI to `--bundle`**

In `seed-pipeline/src/seed_pipeline/cli/commands/evaluation.py`, replace the imports down to `evaluation_app = ...` and the two build functions with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pharma_agent.domain.corpus.bundle import BundleValidationError, read_bundle

from seed_pipeline.bundle.evaluation_chunks import write_evaluation_chunks
from seed_pipeline.cli.commands.bundle import invalid_bundle
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.paths import (
    DEFAULT_BUNDLE_DIR,
    EVALUATION_CHUNKS_PATH,
    PROCESSED_EVALUATION_DIR,
    RAG_FINAL_SECTIONS_PATH,
    retrieval_run_roots,
)
from seed_pipeline.evaluation.build_dataset import (
    EvaluationBuildRequest,
    build_evaluation_dataset,
)
from seed_pipeline.evaluation.rejudge_service import (
    RejudgeRequest,
    run_rejudging,
)

evaluation_app = typer.Typer(no_args_is_help=True, add_completion=False)


def evaluation_build_command(
    *,
    sections: Path = RAG_FINAL_SECTIONS_PATH,
    bundle: Path = DEFAULT_BUNDLE_DIR,
    chunks_output: Path = EVALUATION_CHUNKS_PATH,
    output_dir: Path = PROCESSED_EVALUATION_DIR,
    patient_query_count: int = 500,
    evaluation_row_count: int = 10_000,
) -> CommandResult:
    try:
        knowledge = read_bundle(bundle)
    except BundleValidationError as exc:
        raise invalid_bundle(exc) from exc
    chunk_count = write_evaluation_chunks(knowledge, chunks_output)
    result = build_evaluation_dataset(
        EvaluationBuildRequest(
            sections_path=sections,
            chunks_path=chunks_output,
            output_dir=output_dir,
            patient_query_count=patient_query_count,
            evaluation_row_count=evaluation_row_count,
        )
    )
    return CommandResult(
        command="evaluation build",
        status=CommandStatus.COMPLETE,
        artifact=result.evaluation_path,
        details={
            "chunks": chunk_count,
            "patient_queries": str(result.patient_queries_path),
            "patient_query_count": result.patient_query_count,
            "evaluation_row_count": result.evaluation_row_count,
        },
    )


@evaluation_app.command("build")
def evaluation_build(
    ctx: typer.Context,
    sections: Annotated[Path, typer.Option("--sections")] = RAG_FINAL_SECTIONS_PATH,
    bundle: Annotated[Path, typer.Option("--bundle", file_okay=False)] = DEFAULT_BUNDLE_DIR,
    chunks_output: Annotated[
        Path, typer.Option("--chunks-output", dir_okay=False)
    ] = EVALUATION_CHUNKS_PATH,
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = PROCESSED_EVALUATION_DIR,
    patient_query_count: Annotated[int, typer.Option("--patient-query-count")] = 500,
    evaluation_row_count: Annotated[
        int, typer.Option("--evaluation-row-count")
    ] = 10_000,
) -> None:
    run_handler(
        state_from_context(ctx),
        lambda: evaluation_build_command(
            sections=sections,
            bundle=bundle,
            chunks_output=chunks_output,
            output_dir=output_dir,
            patient_query_count=patient_query_count,
            evaluation_row_count=evaluation_row_count,
        ),
    )
```

The `rejudge-current` command and `_rejudge_current` below stay unchanged.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/bundle/test_evaluation_chunks.py tests/cli/test_evaluation_command.py`
Expected: all pass.

- [ ] **Step 6: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/bundle/evaluation_chunks.py seed-pipeline/src/seed_pipeline/cli/commands/evaluation.py seed-pipeline/src/seed_pipeline/config/paths.py seed-pipeline/src/seed_pipeline/evaluation/patient_query_generation.py seed-pipeline/src/seed_pipeline/evaluation/build_section_retrieval_eval.py seed-pipeline/tests/bundle/test_evaluation_chunks.py seed-pipeline/tests/cli/test_evaluation_command.py
git commit -m "feat(seed): build the evaluation dataset from bundle chunks"
```

The commit message ends with the session attribution trailer.

---

### Task 9: `seed retrieve` through the backend `RetrievalService`

Spec §10.4: the private Qdrant retrievers are replaced by `build_retrieval_service(settings, *, database=None, embedder=None)` (overview §3.4) running on dev Postgres and Qdrant with an imported, published release. `seed retrieve` loads the backend settings (default `../backend/.env`), overrides the retrieval block (`mode` = `bm25`, `dense` or `hybrid`, K values, `collections = [collection]`, rerank protocol `none` with `top_n = max_candidates = candidate_k`, Langfuse keys cleared so evaluation runs are not traced) and writes the same candidate artifact as before, so `seed rerank` and `seed metrics` keep working unchanged.

Query vectors never come from an embedding endpoint during evaluation. For `dense` and `hybrid`, `seed retrieve` checks that the `seed embed queries` cache (`QueryEmbeddingCache`, keyed by `(model, query_id, query_hash)`) holds a vector for every evaluation row, then injects a `CachedQueryEmbedder` that answers `embed(texts)` by `query_hash(text)` and raises `QueryEmbeddingMissing` for any text it does not know. `bm25` is sparse only and gets no embedder. The run identity records `query_embeddings_sha256` (the validated cache subset, as before), `release_id` (from a probe query; every later hit must carry the same release), `chunker_version` and the backend `embedding_model`.

Candidates use the gold chunk label as `chunk_id`, `section_key` as `section_id`, `ordinal` as `chunk_index`, and `Hit.embedding_text` as `document_text`: the reranker text of the old benchmark and of the backend reranker (overview §3.4 keeps `embedding_text` on `Hit`).

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py`, `seed-pipeline/src/seed_pipeline/evaluation/cached_query_embedder.py`, `seed-pipeline/src/seed_pipeline/evaluation/candidate_text.py`
- Delete: `seed-pipeline/src/seed_pipeline/evaluation/retrieval_service.py`, `seed-pipeline/src/seed_pipeline/evaluation/retrievers.py`, `seed-pipeline/tests/evaluation/test_retrieval_service.py`, `seed-pipeline/tests/evaluation/test_retrievers.py`
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/retrieval_candidate_artifact.py:18`, `seed-pipeline/src/seed_pipeline/evaluation/rerankers.py:8,113-115`, `seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py:62`
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py:16-27` (`RunIdentity`)
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py` (whole file)
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py` (add `BACKEND_ENV_FILE`)
- Create: `seed-pipeline/tests/evaluation/test_backend_retrieval.py`
- Modify: `seed-pipeline/tests/cli/test_retrieve_command.py` (whole file)

**Interfaces:**
- Consumes (P1, P3 as pinned in overview §3.4): `CHUNKER_VERSION`; `Hit` (`chunk_version_id`, `release_id`, `collection_id`, `document_key`, `section_key`, `section_revision_id`, `ordinal`, `hydrate_strategy`, `source`, `title`, `section`, `start_page`, `end_page`, `context_header`, `chunk_text`, `embedding_text`, `kind`, `table_key`, `colloquial_mapping`, `term_annotations`, `fusion_score`, `rerank_score`, `matched_queries`), `HydrateStrategy`, `Query(text, origin)`, `QueryOrigin.INITIAL`, `RetrievedItem(hit, chunks)`, `SearchResult(items, queries, error)`, `RetrievalService.search(queries, rerank_query, known_scores=None) -> SearchResult`, `build_retrieval_service(settings, *, database=None, embedder=None) -> RetrievalStack` (`service`, `aclose`), `Settings` (`retrieval.mode` in `hybrid|dense|bm25`, `prefetch_k`, `rrf_k`, `candidate_k`, `qdrant_collection`, `collections`, `embedding.model`, `embedding.dimension`, `rerank.protocol|top_n|max_candidates`, `langfuse.public_key|secret_key`). The injected embedder has the members of P2's `pharma_agent.domain.corpus.ports.Embedder` (`model`, `dimension`, `async embed(texts)`), which also covers the retriever's embed-only protocol.
- Consumes (seed): `QueryEmbeddingCache(path, vector_dim, model_sha256)` with `get(model, query_id, query_text)`, `set(...)`, `validate_subset(rows, model) -> ValidatedSubset(total, complete, missing, sha256)`, `QueryEmbeddingCacheError`, `query_hash`, `query_embedding_cache_path(model)`, `require_model`, `load_query_rows`, `build_candidate_artifact`, `CandidateArtifact`, `CandidateArtifactReader`, `RunIdentity`, `RunWorkspace`, `sha256_file`, `gold_chunk_label`, `COLLECTION_KEY`.
- Produces:
  - `seed_pipeline.evaluation.cached_query_embedder`: `QueryEmbeddingMissing(QueryEmbeddingCacheError)`, `CachedQueryEmbedder(vectors: Mapping[str, Sequence[float]], *, model: str, dimension: int, source: Path)` with `model`, `dimension`, `async embed(texts) -> list[list[float]]`, `cached_query_embedder(cache, rows, *, model, dimension) -> tuple[CachedQueryEmbedder, str]` (embedder, subset sha256).
  - `seed_pipeline.evaluation.backend_retrieval`: `RETRIEVERS = ("bm25", "dense", "hybrid")`, `SearchService` protocol (`async search(queries, rerank_query) -> SearchResult`), `RetrievalBackend` protocol (`service`, `async aclose()`), `BackendFactory` protocol (`__call__(settings, *, embedder: CachedQueryEmbedder | None = None) -> RetrievalBackend`), `RetrieveRequest(evaluation_path, run_root, retriever, candidate_k, rrf_k, limit, force, collection="formulary", prefetch_k=None, artifact_root=None, backend_env_file=None, query_embeddings=None)`, `RetrieveResult(workspace, artifact)`, `resolve_prefetch_k`, `evaluation_settings(base, request) -> Settings`, `query_cache_for(request, *, model, dimension) -> QueryEmbeddingCache`, `candidate_from_hit(hit, rank, source) -> RetrievalCandidate`, `BackendCandidateRetriever(runner, service, *, source, release_id)` with `search_batch(rows, limit)`, `run_retrieval(request, *, backend_factory=build_retrieval_service) -> RetrieveResult`.
  - `seed_pipeline.evaluation.candidate_text.candidate_document_text(payload) -> str` (moved unchanged from `retrievers.py`).
  - `RunIdentity.release_id: str | None = None`, `RunIdentity.chunker_version: str | None = None` (old `run.json` files still load).
  - `seed_pipeline.config.paths.BACKEND_ENV_FILE = PROJECT_ROOT.parent / "backend" / ".env"`.
  - CLI: `seed retrieve --run NAME [--evaluation FILE] [--retriever bm25|dense|hybrid] [--candidate-k N] [--prefetch-k N] [--rrf-k N] [--limit N] [--collection KEY] [--query-embeddings FILE] [--backend-env-file FILE] [--force]` (`--model` and `--qdrant-url` are removed: the model and Qdrant come from the backend settings).

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/evaluation/test_backend_retrieval.py`:

```python
import asyncio
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION
from pharma_agent.domain.retrieval.models import (
    Hit,
    HydrateStrategy,
    Query,
    RetrievedItem,
)
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.infrastructure.settings import Settings

from seed_pipeline.evaluation.backend_retrieval import RetrieveRequest, run_retrieval
from seed_pipeline.evaluation.cached_query_embedder import (
    CachedQueryEmbedder,
    QueryEmbeddingMissing,
)
from seed_pipeline.evaluation.query_embedding_cache import QueryEmbeddingCache
from seed_pipeline.evaluation.run_workspace import load_run_record

MODEL = "fake-embedding-4d"
RELEASE = uuid.UUID("11111111-1111-5111-8111-111111111111")
OTHER_RELEASE = uuid.UUID("22222222-2222-5222-8222-222222222222")
COLLECTION = uuid.UUID("33333333-3333-5333-8333-333333333333")
DOSAGE = "drug:paracetamol:lieu-luong-va-cach-dung"
ADULT = "liều paracetamol người lớn"
CHILD = "paracetamol trẻ em"
VECTORS = {ADULT: [0.25, 0.5, 0.75, 1.0], CHILD: [1.0, 0.75, 0.5, 0.25]}


def make_hit(ordinal: int, score: float, release: uuid.UUID = RELEASE) -> Hit:
    return Hit(
        chunk_version_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{DOSAGE}:{ordinal}"),
        release_id=release,
        collection_id=COLLECTION,
        document_key="drug:paracetamol",
        section_key=DOSAGE,
        section_revision_id=uuid.uuid5(uuid.NAMESPACE_URL, DOSAGE),
        ordinal=ordinal,
        hydrate_strategy=HydrateStrategy.FULL_SECTION,
        source="Dược thư Quốc gia Việt Nam",
        title="PARACETAMOL",
        section="Liều lượng và cách dùng",
        start_page=1133,
        end_page=1134,
        context_header="PARACETAMOL\n> Liều lượng và cách dùng",
        chunk_text=f"Đoạn {ordinal}",
        embedding_text=f"PARACETAMOL\n> Liều lượng và cách dùng\n\nĐoạn {ordinal}",
        kind="prose",
        table_key=None,
        colloquial_mapping=None,
        term_annotations=[],
        fusion_score=score,
        rerank_score=None,
        matched_queries=[],
    )


class FakeSearchService:
    def __init__(self, hits: dict[str, list[Hit]]) -> None:
        self.hits = hits
        self.calls: list[str] = []
        self.vectors: list[list[float]] = []
        self.embedder: CachedQueryEmbedder | None = None

    async def search(self, queries: Sequence[Query], rerank_query: str) -> SearchResult:
        self.calls.append(rerank_query)
        if self.embedder is not None:
            self.vectors.extend(await self.embedder.embed([q.text for q in queries]))
        return SearchResult(
            items=[RetrievedItem(hit=hit) for hit in self.hits[rerank_query]],
            queries=list(queries),
        )


@dataclass
class FakeStack:
    service: FakeSearchService
    settings: list[Settings] = field(default_factory=list)
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, stack: FakeStack) -> None:
        self.stack = stack

    def __call__(
        self, settings: Settings, *, embedder: CachedQueryEmbedder | None = None
    ) -> FakeStack:
        self.stack.settings.append(settings)
        self.stack.service.embedder = embedder
        return self.stack


def _evaluation(tmp_path: Path) -> Path:
    path = tmp_path / "evaluation.jsonl"
    rows = [{"query_id": "q1", "query": ADULT}, {"query_id": "q2", "query": CHILD}]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _query_cache(tmp_path: Path, rows: dict[str, str]) -> Path:
    path = tmp_path / "query_embeddings.jsonl"
    cache = QueryEmbeddingCache(path, vector_dim=4)
    for query_id, text in rows.items():
        cache.set(MODEL, query_id, text, VECTORS[text])
    return path


def _request(
    tmp_path: Path,
    *,
    retriever: str = "hybrid",
    prefetch_k: int | None = 50,
    cache_rows: dict[str, str] | None = None,
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
        artifact_root=tmp_path / "heavy-run",
        retriever=retriever,
        candidate_k=2,
        prefetch_k=prefetch_k,
        rrf_k=2,
        limit=None,
        force=False,
        backend_env_file=env_file,
        query_embeddings=_query_cache(tmp_path, rows),
    )


def _hits(child_release: uuid.UUID = RELEASE) -> dict[str, list[Hit]]:
    return {
        ADULT: [make_hit(2, 0.9), make_hit(1, 0.5)],
        CHILD: [make_hit(1, 0.7, release=child_release)],
    }


def test_hybrid_retrieval_uses_cached_query_vectors(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    result = run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))

    assert stack.closed
    settings = stack.settings[0]
    assert settings.retrieval.mode == "hybrid"
    assert (settings.retrieval.candidate_k, settings.retrieval.prefetch_k) == (2, 50)
    assert settings.retrieval.collections == ["formulary"]
    assert settings.retrieval.rerank.protocol == "none"
    assert settings.retrieval.rerank.top_n == 2
    assert settings.retrieval.rerank.max_candidates == 2
    assert settings.langfuse.public_key is None
    assert stack.service.vectors == [VECTORS[ADULT], VECTORS[ADULT], VECTORS[CHILD]]
    assert result.artifact.query_count == 2
    first = json.loads(result.artifact.data_path.read_text("utf-8").splitlines()[0])
    assert [c["chunk_id"] for c in first["candidates"]] == [
        f"{DOSAGE}:chunk-002",
        f"{DOSAGE}:chunk-001",
    ]
    assert first["candidates"][0]["payload"] == {
        "chunk_id": f"{DOSAGE}:chunk-002",
        "section_id": DOSAGE,
        "chunk_index": 2,
    }
    assert first["candidates"][0]["document_text"] == make_hit(2, 0.9).embedding_text
    identity = load_run_record(tmp_path / "run" / "run.json").identity
    assert identity.release_id == str(RELEASE)
    assert identity.chunker_version == CHUNKER_VERSION
    assert identity.embedding_model == MODEL
    assert identity.collection_name == settings.retrieval.qdrant_collection
    assert identity.query_embeddings_sha256 is not None


def test_existing_complete_candidates_are_reused(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))
    run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))
    assert len(stack.service.calls) == 3

    run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))

    assert len(stack.service.calls) == 4


def test_hits_from_another_release_stop_the_run(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits(child_release=OTHER_RELEASE)))

    with pytest.raises(RuntimeError, match="pinned to release"):
        run_retrieval(_request(tmp_path), backend_factory=FakeFactory(stack))
    assert stack.closed


def test_bm25_needs_no_query_vectors(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    run_retrieval(
        _request(tmp_path, retriever="bm25", prefetch_k=None, cache_rows={}),
        backend_factory=FakeFactory(stack),
    )

    assert stack.settings[0].retrieval.mode == "bm25"
    assert stack.service.embedder is None
    identity = load_run_record(tmp_path / "run" / "run.json").identity
    assert (identity.retriever, identity.query_embeddings_sha256) == ("bm25", None)


def test_missing_query_vectors_stop_before_the_backend_starts(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    with pytest.raises(QueryEmbeddingMissing, match="missing 1 of 2"):
        run_retrieval(
            _request(tmp_path, cache_rows={"q1": ADULT}),
            backend_factory=FakeFactory(stack),
        )
    assert stack.settings == []


def test_cached_embedder_rejects_unknown_query_text(tmp_path: Path) -> None:
    embedder = CachedQueryEmbedder(
        {"0" * 64: [0.1, 0.2, 0.3, 0.4]},
        model=MODEL,
        dimension=4,
        source=tmp_path / "cache.jsonl",
    )

    with pytest.raises(QueryEmbeddingMissing, match="seed embed queries"):
        asyncio.run(embedder.embed(["câu hỏi chưa embed"]))


def test_unknown_retriever_and_dense_prefetch_are_rejected(tmp_path: Path) -> None:
    stack = FakeStack(service=FakeSearchService(_hits()))

    with pytest.raises(ValueError, match="bm25, dense or hybrid"):
        run_retrieval(
            _request(tmp_path, retriever="sparse"), backend_factory=FakeFactory(stack)
        )
    with pytest.raises(ValueError, match="--prefetch-k"):
        run_retrieval(
            _request(tmp_path, retriever="dense", prefetch_k=10),
            backend_factory=FakeFactory(stack),
        )
```

Replace `seed-pipeline/tests/cli/test_retrieve_command.py`:

```python
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import seed_pipeline.cli.commands.retrieve as retrieve_command
from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import (
    BACKEND_ENV_FILE,
    HEAVY_RETRIEVAL_EVAL_DIR,
    RETRIEVAL_EVAL_DIR,
)

runner = CliRunner()


def _fake_run(captured: dict):
    def fake_run(request):
        captured["request"] = request
        return SimpleNamespace(
            workspace=SimpleNamespace(
                root=Path("data/retrieval_eval/experiment"),
                identity=SimpleNamespace(release_id="release"),
            ),
            artifact=SimpleNamespace(
                data_path=Path("data/heavy/candidates.jsonl"), query_count=1
            ),
        )

    return fake_run


def test_retrieve_passes_split_roots_and_backend_defaults(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(retrieve_command, "run_retrieval", _fake_run(captured))

    result = runner.invoke(app, ["--json", "retrieve", "--run", "experiment"])

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.run_root == RETRIEVAL_EVAL_DIR / "experiment"
    assert request.artifact_root == HEAVY_RETRIEVAL_EVAL_DIR / "experiment"
    assert request.retriever == "hybrid"
    assert request.rrf_k == 2
    assert request.collection == "formulary"
    assert request.backend_env_file == BACKEND_ENV_FILE
    assert request.query_embeddings is None


def test_retrieve_forwards_bm25_k_values_and_query_cache(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setattr(retrieve_command, "run_retrieval", _fake_run(captured))

    result = runner.invoke(
        app,
        [
            "retrieve",
            "--run",
            "experiment",
            "--retriever",
            "bm25",
            "--candidate-k",
            "30",
            "--rrf-k",
            "60",
            "--query-embeddings",
            str(tmp_path / "queries.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert (request.retriever, request.candidate_k, request.rrf_k) == ("bm25", 30, 60)
    assert request.query_embeddings == tmp_path / "queries.jsonl"


def test_retrieve_no_longer_accepts_qdrant_options():
    result = runner.invoke(
        app, ["retrieve", "--run", "experiment", "--qdrant-url", "http://x"]
    )

    assert result.exit_code == 2
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/evaluation/test_backend_retrieval.py tests/cli/test_retrieve_command.py`
Expected: collection error `ModuleNotFoundError: No module named 'seed_pipeline.evaluation.backend_retrieval'` and `ImportError: cannot import name 'BACKEND_ENV_FILE'`.

- [ ] **Step 3: Move the payload text helper and fix the reranker input**

```bash
cd /home/andv/personal/thesis/seed-pipeline
git rm src/seed_pipeline/evaluation/retrieval_service.py src/seed_pipeline/evaluation/retrievers.py tests/evaluation/test_retrieval_service.py tests/evaluation/test_retrievers.py
```

Create `seed-pipeline/src/seed_pipeline/evaluation/candidate_text.py` (the function body is the one that lived in `retrievers.py`):

```python
from seed_pipeline.corpus.metadata.payload_layers import (
    compact_colloquial_mapping,
    format_colloquial_mapping,
)


def candidate_document_text(payload: dict) -> str:
    embedding_text = str(payload.get("embedding_text") or "").strip()
    if embedding_text:
        return embedding_text
    context_header = str(payload.get("context_header") or "").strip()
    text = str(payload.get("chunk_text") or "").strip()
    visible_text = (
        f"{context_header}\n\n{text}"
        if context_header and text
        else context_header or text
    )
    colloquial_text = format_colloquial_mapping(
        compact_colloquial_mapping(payload), visible_text
    )
    parts = []
    if context_header:
        parts.append(context_header)
    if colloquial_text:
        parts.append(colloquial_text)
    if text:
        parts.append(text)
    return "\n\n".join(parts).strip()
```

In `seed-pipeline/src/seed_pipeline/evaluation/retrieval_candidate_artifact.py` replace the `retrievers` import with `from seed_pipeline.evaluation.candidate_text import candidate_document_text` (line 33 is unchanged).

In `seed-pipeline/src/seed_pipeline/evaluation/rerankers.py` replace the import on line 8 with `from seed_pipeline.evaluation.candidate_text import candidate_document_text`, and make `LlamaCppReranker.rerank` score the text stored in the candidate artifact (candidates from `seed retrieve` carry `document_text = embedding_text` and a payload without text, so the local reranker scores the same text as the old benchmark):

```python
        documents = [
            candidate.document_text or candidate_document_text(candidate.payload)
            for candidate in candidates
        ]
```

In `seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py:62` change the local import to `from seed_pipeline.evaluation.candidate_text import candidate_document_text`.

- [ ] **Step 4: Extend the run identity**

In `seed-pipeline/src/seed_pipeline/evaluation/run_workspace.py`:

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
```

In `seed-pipeline/src/seed_pipeline/config/paths.py`, after `GGUF_ROOT`:

```python
BACKEND_ENV_FILE = PROJECT_ROOT.parent / "backend" / ".env"
```

- [ ] **Step 5: Implement the cached query embedder**

Create `seed-pipeline/src/seed_pipeline/evaluation/cached_query_embedder.py`:

```python
"""Query embedder for evaluation, backed by the `seed embed queries` cache.

It never calls a model: evaluation queries were embedded once (on Kaggle or locally) and
retrieval must use exactly those vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seed_pipeline.evaluation.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingCacheError,
    query_hash,
)


class QueryEmbeddingMissing(QueryEmbeddingCacheError):
    """An evaluation query has no cached vector for the backend embedding model."""


class CachedQueryEmbedder:
    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        *,
        model: str,
        dimension: int,
        source: Path,
    ) -> None:
        for digest, vector in vectors.items():
            if len(vector) != dimension:
                raise QueryEmbeddingCacheError(
                    f"Query vector {digest} in {source} has dimension {len(vector)}, "
                    f"expected {dimension}"
                )
        self._vectors = {digest: list(vector) for digest, vector in vectors.items()}
        self._model = model
        self._dimension = dimension
        self._source = source

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in texts if query_hash(text) not in self._vectors]
        if missing:
            raise QueryEmbeddingMissing(
                f"Query embedding cache {self._source} has no {self._model} vector for "
                f"{len(missing)} query text(s), first {missing[0][:80]!r}; run "
                f"`uv run seed embed queries --model {self._model}` for this evaluation"
            )
        return [list(self._vectors[query_hash(text)]) for text in texts]


def cached_query_embedder(
    cache: QueryEmbeddingCache,
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    dimension: int,
) -> tuple[CachedQueryEmbedder, str]:
    """Check that every row has a cached vector and return the embedder and subset sha256."""
    row_list = [dict(row) for row in rows]
    subset = cache.validate_subset(row_list, model)
    if not subset.is_complete or subset.sha256 is None:
        raise QueryEmbeddingMissing(
            f"Query embedding cache {cache.path} is missing {subset.missing} of "
            f"{subset.total} queries for {model}; run "
            f"`uv run seed embed queries --model {model}` for this evaluation"
        )
    vectors: dict[str, list[float]] = {}
    for row in row_list:
        text = str(row["query"])
        vector = cache.get(model, str(row["query_id"]), text)
        if vector is None:
            raise QueryEmbeddingMissing(
                f"Query embedding cache {cache.path} lost {row['query_id']}"
            )
        vectors[query_hash(text)] = vector
    return (
        CachedQueryEmbedder(vectors, model=model, dimension=dimension, source=cache.path),
        subset.sha256,
    )
```

- [ ] **Step 6: Implement retrieval through the backend**

Create `seed-pipeline/src/seed_pipeline/evaluation/backend_retrieval.py`:

```python
"""Retrieval evaluation through the backend RetrievalService (spec §10.4)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from pharma_agent.domain.corpus.chunking import CHUNKER_VERSION
from pharma_agent.domain.retrieval.models import Hit, Query, QueryOrigin
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.infrastructure.composition import build_retrieval_service
from pharma_agent.infrastructure.settings import Settings

from seed_pipeline.bundle.chunks import gold_chunk_label
from seed_pipeline.bundle.export import COLLECTION_KEY
from seed_pipeline.config.paths import query_embedding_cache_path
from seed_pipeline.evaluation.artifact_contracts import sha256_file
from seed_pipeline.evaluation.cached_query_embedder import (
    CachedQueryEmbedder,
    cached_query_embedder,
)
from seed_pipeline.evaluation.dump_retrieval_candidates import load_query_rows
from seed_pipeline.evaluation.query_embedding_cache import QueryEmbeddingCache
from seed_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifact,
    CandidateArtifactReader,
    build_candidate_artifact,
)
from seed_pipeline.evaluation.retrieval_types import RetrievalCandidate
from seed_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace
from seed_pipeline.runtime.catalog import require_model

RETRIEVERS = ("bm25", "dense", "hybrid")


class SearchService(Protocol):
    async def search(
        self, queries: Sequence[Query], rerank_query: str
    ) -> SearchResult: ...


class RetrievalBackend(Protocol):
    @property
    def service(self) -> SearchService: ...

    async def aclose(self) -> None: ...


class BackendFactory(Protocol):
    def __call__(
        self, settings: Settings, *, embedder: CachedQueryEmbedder | None = None
    ) -> RetrievalBackend: ...


@dataclass(frozen=True)
class RetrieveRequest:
    evaluation_path: Path
    run_root: Path
    retriever: str
    candidate_k: int
    rrf_k: int
    limit: int | None
    force: bool
    collection: str = COLLECTION_KEY
    prefetch_k: int | None = None
    artifact_root: Path | None = None
    backend_env_file: Path | None = None
    query_embeddings: Path | None = None


@dataclass(frozen=True)
class RetrieveResult:
    workspace: RunWorkspace
    artifact: CandidateArtifact


def resolve_prefetch_k(
    retriever: str, candidate_k: int, prefetch_k: int | None
) -> int | None:
    if retriever != "hybrid":
        if prefetch_k is not None:
            raise ValueError("--prefetch-k is only valid with --retriever hybrid")
        return None
    effective = candidate_k if prefetch_k is None else prefetch_k
    if effective < candidate_k:
        raise ValueError("--prefetch-k must be greater than or equal to --candidate-k")
    return effective


def evaluation_settings(base: Settings, request: RetrieveRequest) -> Settings:
    rerank = base.retrieval.rerank.model_copy(
        update={
            "protocol": "none",
            "top_n": request.candidate_k,
            "max_candidates": request.candidate_k,
        }
    )
    retrieval = base.retrieval.model_copy(
        update={
            "mode": request.retriever,
            "candidate_k": request.candidate_k,
            "prefetch_k": request.prefetch_k or request.candidate_k,
            "rrf_k": request.rrf_k,
            "collections": [request.collection],
            "rerank": rerank,
        }
    )
    langfuse = base.langfuse.model_copy(update={"public_key": None, "secret_key": None})
    return base.model_copy(update={"retrieval": retrieval, "langfuse": langfuse})


def query_cache_for(
    request: RetrieveRequest, *, model: str, dimension: int
) -> QueryEmbeddingCache:
    if request.query_embeddings is not None:
        return QueryEmbeddingCache(request.query_embeddings, vector_dim=dimension)
    spec = require_model(model)
    if spec.vector_dimension != dimension:
        raise ValueError(
            f"Backend embedding dimension {dimension} does not match "
            f"{spec.vector_dimension} for {model}"
        )
    return QueryEmbeddingCache(
        query_embedding_cache_path(model), vector_dim=dimension, model_sha256=spec.sha256
    )


def candidate_from_hit(hit: Hit, rank: int, source: str) -> RetrievalCandidate:
    label = gold_chunk_label(hit.section_key, hit.ordinal)
    return RetrievalCandidate(
        chunk_id=label,
        score=hit.fusion_score,
        rank=rank,
        source=source,
        payload={
            "chunk_id": label,
            "section_id": hit.section_key,
            "chunk_index": hit.ordinal,
        },
        document_text=hit.embedding_text,
    )


async def search_hits(service: SearchService, text: str) -> list[Hit]:
    result = await service.search(
        [Query(text=text, origin=QueryOrigin.INITIAL)], rerank_query=text
    )
    if result.error is not None:
        raise RuntimeError(f"backend retrieval failed: {result.error}")
    return [item.hit for item in result.items]


async def current_release_id(service: SearchService, probe_query: str) -> str:
    releases = {str(hit.release_id) for hit in await search_hits(service, probe_query)}
    if len(releases) != 1:
        raise RuntimeError(
            "Expected hits from exactly one published release, found "
            f"{sorted(releases)}; run `pharma-agent corpus import ... --publish` first"
        )
    return releases.pop()


class BackendCandidateRetriever:
    def __init__(
        self,
        runner: asyncio.Runner,
        service: SearchService,
        *,
        source: str,
        release_id: str,
    ) -> None:
        self._runner = runner
        self._service = service
        self._source = source
        self._release_id = release_id

    def search_batch(
        self, rows: Sequence[dict[str, Any]], limit: int
    ) -> list[list[RetrievalCandidate]]:
        return self._runner.run(self._search_rows(rows, limit))

    async def _search_rows(
        self, rows: Sequence[dict[str, Any]], limit: int
    ) -> list[list[RetrievalCandidate]]:
        return list(
            await asyncio.gather(*(self._search_row(row, limit) for row in rows))
        )

    async def _search_row(
        self, row: dict[str, Any], limit: int
    ) -> list[RetrievalCandidate]:
        hits = await search_hits(self._service, str(row["query"]))
        releases = {str(hit.release_id) for hit in hits} - {self._release_id}
        if releases:
            raise RuntimeError(
                f"Query {row['query_id']} returned release {sorted(releases)} but the "
                f"run is pinned to release {self._release_id}; start a new --run"
            )
        return [
            candidate_from_hit(hit, rank, self._source)
            for rank, hit in enumerate(hits[:limit], start=1)
        ]


def run_retrieval(
    request: RetrieveRequest,
    *,
    backend_factory: BackendFactory = build_retrieval_service,
) -> RetrieveResult:
    if request.candidate_k < 1 or request.rrf_k < 1:
        raise ValueError("candidate_k and rrf_k must be >= 1")
    if request.retriever not in RETRIEVERS:
        raise ValueError("retriever must be bm25, dense or hybrid")
    request = replace(
        request,
        prefetch_k=resolve_prefetch_k(
            request.retriever, request.candidate_k, request.prefetch_k
        ),
    )
    rows = load_query_rows(request.evaluation_path, request.limit)
    base = (
        Settings(_env_file=request.backend_env_file)
        if request.backend_env_file is not None
        else Settings()
    )
    settings = evaluation_settings(base, request)
    embedding = settings.retrieval.embedding
    embedder: CachedQueryEmbedder | None = None
    query_embeddings_sha256: str | None = None
    if request.retriever != "bm25":
        embedder, query_embeddings_sha256 = cached_query_embedder(
            query_cache_for(request, model=embedding.model, dimension=embedding.dimension),
            rows,
            model=embedding.model,
            dimension=embedding.dimension,
        )
    stack = backend_factory(settings, embedder=embedder)
    with asyncio.Runner() as runner:
        try:
            release_id = runner.run(
                current_release_id(stack.service, str(rows[0]["query"]))
            )
            identity = RunIdentity(
                evaluation_path=str(request.evaluation_path.resolve()),
                evaluation_sha256=sha256_file(request.evaluation_path),
                collection_name=settings.retrieval.qdrant_collection,
                embedding_model=embedding.model,
                query_embeddings_sha256=query_embeddings_sha256,
                retriever=request.retriever,
                candidate_k=request.candidate_k,
                rrf_k=request.rrf_k,
                limit=request.limit,
                prefetch_k=request.prefetch_k,
                release_id=release_id,
                chunker_version=CHUNKER_VERSION,
            )
            workspace = RunWorkspace.open_or_create(
                request.run_root,
                identity,
                artifact_root=request.artifact_root,
                force=request.force,
            )
            reusable = _reusable_artifact(workspace, request)
            if reusable is not None:
                workspace.record_candidates(reusable)
                return RetrieveResult(workspace, reusable)
            artifact = build_candidate_artifact(
                rows=rows,
                retriever=BackendCandidateRetriever(
                    runner,
                    stack.service,
                    source=request.retriever,
                    release_id=release_id,
                ),
                output_path=workspace.candidates_dir / "candidates.jsonl",
                identity=asdict(identity),
                candidate_k=request.candidate_k,
            )
            workspace.record_candidates(artifact)
            return RetrieveResult(workspace, artifact)
        finally:
            runner.run(stack.aclose())


def _reusable_artifact(
    workspace: RunWorkspace, request: RetrieveRequest
) -> CandidateArtifact | None:
    data_path = workspace.candidates_dir / "candidates.jsonl"
    manifest_path = workspace.candidates_dir / "manifest.json"
    if request.force or not data_path.is_file() or not manifest_path.is_file():
        return None
    reader = CandidateArtifactReader(data_path, manifest_path)
    query_count = 0
    pair_count = 0
    for record in reader:
        query_count += 1
        pair_count += len(record["candidates"])
    if reader.manifest.identity.get("candidate_k") != request.candidate_k:
        return None
    return CandidateArtifact(
        data_path, manifest_path, reader.manifest, query_count, pair_count
    )
```

`build_retrieval_service` satisfies `BackendFactory`: its `embedder` parameter accepts any object with the backend `Embedder` members, which `CachedQueryEmbedder` provides, and its extra `database` keyword has a default.

Replace `seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from seed_pipeline.bundle.export import COLLECTION_KEY
from seed_pipeline.cli.options import positive_int
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RETRIEVER,
    DEFAULT_RRF_K,
)
from seed_pipeline.config.paths import (
    BACKEND_ENV_FILE,
    PROCESSED_EVALUATION_DIR,
    retrieval_run_roots,
)
from seed_pipeline.evaluation.backend_retrieval import RetrieveRequest, run_retrieval


def retrieve(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")],
    evaluation: Annotated[Path, typer.Option("--evaluation")] = PROCESSED_EVALUATION_DIR
    / "section_retrieval_eval.jsonl",
    retriever: Annotated[
        str, typer.Option("--retriever", help="bm25, dense or hybrid")
    ] = DEFAULT_RETRIEVER,
    candidate_k: Annotated[
        int,
        typer.Option("--candidate-k", callback=lambda _c, _p, v: positive_int(str(v))),
    ] = DEFAULT_CANDIDATE_K,
    prefetch_k: Annotated[
        int | None,
        typer.Option(
            "--prefetch-k",
            callback=lambda _c, _p, value: (
                None if value is None else positive_int(str(value))
            ),
        ),
    ] = None,
    rrf_k: Annotated[
        int, typer.Option("--rrf-k", callback=lambda _c, _p, v: positive_int(str(v)))
    ] = DEFAULT_RRF_K,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    collection: Annotated[str, typer.Option("--collection")] = COLLECTION_KEY,
    query_embeddings: Annotated[
        Path | None,
        typer.Option(
            "--query-embeddings",
            dir_okay=False,
            help="Query embedding cache from `seed embed queries` (dense and hybrid)",
        ),
    ] = None,
    backend_env_file: Annotated[
        Path, typer.Option("--backend-env-file", dir_okay=False)
    ] = BACKEND_ENV_FILE,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    metadata_root, artifact_root = retrieval_run_roots(run)
    request = RetrieveRequest(
        evaluation_path=evaluation,
        run_root=metadata_root,
        artifact_root=artifact_root,
        retriever=retriever,
        candidate_k=candidate_k,
        prefetch_k=prefetch_k,
        rrf_k=rrf_k,
        limit=limit,
        force=force,
        collection=collection,
        backend_env_file=backend_env_file,
        query_embeddings=query_embeddings,
    )
    run_handler(state_from_context(ctx), lambda: _run(request))


def _run(request: RetrieveRequest) -> CommandResult:
    result = run_retrieval(request)
    return CommandResult(
        "retrieve",
        CommandStatus.COMPLETE,
        result.workspace.root,
        {
            "artifact": str(result.artifact.data_path),
            "queries": result.artifact.query_count,
            "release_id": result.workspace.identity.release_id,
        },
    )
```

`_run` looks `run_retrieval` up in the module namespace at call time, so the CLI tests' `monkeypatch.setattr(retrieve_command, "run_retrieval", ...)` takes effect.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q tests/evaluation tests/cli`
Expected: all pass. `tests/evaluation/conftest.py` still builds candidates with `OneCandidateRetriever` (which sets `document_text`), so rerank and metrics tests are unaffected.

- [ ] **Step 8: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green (`Settings(_env_file=...)` is the typed pydantic-settings keyword that P2's settings tests also use).

- [ ] **Step 9: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/src/seed_pipeline/evaluation seed-pipeline/src/seed_pipeline/cli/commands/retrieve.py seed-pipeline/src/seed_pipeline/config/paths.py seed-pipeline/tests/evaluation/test_backend_retrieval.py seed-pipeline/tests/cli/test_retrieve_command.py
git commit -m "feat(seed): evaluate retrieval through the backend with cached query vectors"
```

(`git rm` in Step 3 already staged the deletions.) The commit message ends with the session attribution trailer.

---

### Task 10: Remove the old chunk/Qdrant contracts and rewrite the docs

Spec §10.5 and §10.6. After Tasks 4–9 nothing reads `rag-final/chunks.jsonl`, the Qdrant payload contract or the chunk vector cache. This task deletes them, bumps the published build contract to `rag-final-v3` (`sections.jsonl`, `blocks.jsonl`, `manifest.json`, `validation_report.json`) and rewrites the seed-pipeline docs around the knowledge bundle. The migration runbook (Task 11) tells the operator to copy the old `chunks.jsonl` to `data/heavy/migration/` before building with this commit.

**Files:**
- Delete: `seed-pipeline/src/seed_pipeline/corpus/metadata/build_rag_metadata.py`, `seed-pipeline/src/seed_pipeline/corpus/metadata/qdrant_payload_contract.py`, `seed-pipeline/src/seed_pipeline/corpus/metadata/term_enrichment.py`, `seed-pipeline/src/seed_pipeline/vector_store/` (package), `seed-pipeline/src/seed_pipeline/integrations/postgres/` (package with `schema/rag_app_schema.sql`), `seed-pipeline/src/seed_pipeline/cli/commands/vectors.py`, `seed-pipeline/tests/vector_store/` (package)
- Modify: `seed-pipeline/src/seed_pipeline/cli/app.py` (drop `vectors`)
- Modify: `seed-pipeline/src/seed_pipeline/config/defaults.py:9` (drop `DEFAULT_QDRANT_URL`), `seed-pipeline/src/seed_pipeline/config/paths.py` (drop `VECTOR_EMBEDDING_CACHE_DIR`, `RAG_FINAL_CHUNKS_PATH`, `chunk_embedding_bundle_dir`)
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/query_embedding_service.py:28-31` and usages, `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py:33-36` and usages, `seed-pipeline/src/seed_pipeline/evaluation/preload_query_embeddings.py:6,12-13`, `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py:46,209`
- Modify: `seed-pipeline/src/seed_pipeline/corpus/validation/validate_final_rag.py:21-24` (import), `validate_unified_chunks` (delete), `combine_validation_reports`
- Modify: `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py` (imports, `_digest_payload`, tail of `build_candidate`), `seed-pipeline/src/seed_pipeline/orchestration/validation_service.py` (`run_validation`), `seed-pipeline/src/seed_pipeline/artifacts/contract.py` (v3)
- Modify: `seed-pipeline/tests/artifacts/test_contract.py`
- Create: `seed-pipeline/tests/test_removed_contracts.py`
- Rewrite: `seed-pipeline/README.md`, `seed-pipeline/docs/guides/downstream.md`, `seed-pipeline/docs/guides/cli-reference.md`, `seed-pipeline/docs/guides/workflow-local-kaggle.md`, `seed-pipeline/docs/guides/workflow-local-only.md`, `seed-pipeline/data/README.md`

**Interfaces:**
- Consumes: everything produced by Tasks 3–9; `COMPOSE_FILE`, `GGUF_ROOT` (Task 6).
- Produces: `seed_pipeline.artifacts.contract.CONTRACT_SCHEMA_VERSION = "rag-final-v3"`, `CONTRACT_FILES = {"sections.jsonl", "blocks.jsonl", "manifest.json", "validation_report.json"}`, `TRACKED_FILES = ("sections.jsonl", "blocks.jsonl", "validation_report.json")`, manifest keys `section_count`, `block_count` (no `chunk_count`); `combine_validation_reports(*, source_report, deep_report) -> dict`; CLI without `seed vectors` and `seed embed chunks`.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/test_removed_contracts.py`:

```python
import importlib.util

import pytest
from typer.testing import CliRunner

from seed_pipeline.cli.app import app
from seed_pipeline.config import paths

REMOVED_MODULES = (
    "seed_pipeline.vector_store",
    "seed_pipeline.integrations.postgres",
    "seed_pipeline.corpus.metadata.build_rag_metadata",
    "seed_pipeline.corpus.metadata.qdrant_payload_contract",
    "seed_pipeline.corpus.metadata.term_enrichment",
    "seed_pipeline.cli.commands.vectors",
    "seed_pipeline.evaluation.retrievers",
    "seed_pipeline.evaluation.retrieval_service",
)


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_old_chunk_contract_modules_are_gone(module: str) -> None:
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("args", [["vectors", "upload"], ["embed", "chunks"]])
def test_old_commands_are_gone(args: list[str]) -> None:
    assert CliRunner().invoke(app, args).exit_code == 2


def test_paths_no_longer_expose_the_unified_chunk_contract() -> None:
    assert not hasattr(paths, "RAG_FINAL_CHUNKS_PATH")
    assert not hasattr(paths, "VECTOR_EMBEDDING_CACHE_DIR")
```

Replace `seed-pipeline/tests/artifacts/test_contract.py`:

```python
import json
from pathlib import Path

import pytest

from seed_pipeline.artifacts.contract import (
    CONTRACT_SCHEMA_VERSION,
    ContractError,
    build_manifest,
    validate_contract_directory,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _published(tmp_path: Path, *, block_section: str = "drug:a:b") -> Path:
    final_dir = tmp_path / "rag-final"
    final_dir.mkdir()
    _write_jsonl(final_dir / "sections.jsonl", [{"id": "drug:a:b", "text": "Văn bản"}])
    _write_jsonl(
        final_dir / "blocks.jsonl",
        [{"block_id": "block-000001", "section_id": block_section, "text": "Văn bản"}],
    )
    (final_dir / "validation_report.json").write_text('{"ok": true}\n', encoding="utf-8")
    manifest = build_manifest(
        final_dir,
        build_id="build",
        source_pdf_sha256="pdf",
        snapshot_id="snapshot",
        snapshot_sha256="archive",
        curated_input_digests={"glossary": "g"},
        config_digest="config",
    )
    (final_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return final_dir


def test_manifest_tracks_sections_and_blocks(tmp_path: Path) -> None:
    manifest = validate_contract_directory(_published(tmp_path))

    assert manifest["schema_version"] == CONTRACT_SCHEMA_VERSION == "rag-final-v3"
    assert (manifest["section_count"], manifest["block_count"]) == (1, 1)
    assert "chunk_count" not in manifest
    assert set(manifest["files"]) == {
        "sections.jsonl",
        "blocks.jsonl",
        "validation_report.json",
    }


def test_contract_requires_blocks_and_rejects_leftover_chunks(tmp_path: Path) -> None:
    final_dir = _published(tmp_path)
    (final_dir / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ContractError, match="must contain exactly"):
        validate_contract_directory(final_dir)
    (final_dir / "chunks.jsonl").unlink()
    (final_dir / "blocks.jsonl").unlink()
    with pytest.raises(ContractError, match="must contain exactly"):
        validate_contract_directory(final_dir)


def test_contract_rejects_blocks_of_unknown_sections(tmp_path: Path) -> None:
    final_dir = _published(tmp_path, block_section="drug:missing:section")

    with pytest.raises(ContractError, match="unknown sections"):
        validate_contract_directory(final_dir)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/test_removed_contracts.py tests/artifacts/test_contract.py`
Expected: `ImportError: cannot import name 'CONTRACT_SCHEMA_VERSION'`; after that import is added the removal tests fail because `find_spec("seed_pipeline.vector_store")` is not `None` and `seed vectors upload` still exists.

- [ ] **Step 3: Delete the old code**

```bash
cd /home/andv/personal/thesis/seed-pipeline
git rm -r \
  src/seed_pipeline/corpus/metadata/build_rag_metadata.py \
  src/seed_pipeline/corpus/metadata/qdrant_payload_contract.py \
  src/seed_pipeline/corpus/metadata/term_enrichment.py \
  src/seed_pipeline/vector_store \
  src/seed_pipeline/integrations/postgres \
  src/seed_pipeline/cli/commands/vectors.py \
  tests/vector_store
sed -i \
  -e 's/DEFAULT_COMPOSE_FILE/COMPOSE_FILE/g' \
  -e 's/DEFAULT_GGUF_ROOT/GGUF_ROOT/g' \
  src/seed_pipeline/evaluation/query_embedding_service.py \
  src/seed_pipeline/evaluation/rerank_service.py \
  src/seed_pipeline/integrations/kaggle/service.py
```

Then fix the imports by hand:

- `src/seed_pipeline/cli/app.py`: delete `from seed_pipeline.cli.commands.vectors import vectors_app` and `app.add_typer(vectors_app, name="vectors")`.
- `src/seed_pipeline/config/defaults.py`: delete `DEFAULT_QDRANT_URL = "http://localhost:6333"`.
- `src/seed_pipeline/config/paths.py`: delete `VECTOR_EMBEDDING_CACHE_DIR = ...`, `RAG_FINAL_CHUNKS_PATH = ...` and the whole `chunk_embedding_bundle_dir` function.
- `src/seed_pipeline/evaluation/query_embedding_service.py` and `src/seed_pipeline/evaluation/rerank_service.py`: replace the `from seed_pipeline.vector_store.ingest_vectors import (COMPOSE_FILE, GGUF_ROOT,)` block (renamed by the `sed`) with `from seed_pipeline.config.paths import COMPOSE_FILE, GGUF_ROOT`, merged into the module's existing `seed_pipeline.config.paths` import where there is one (`query_embedding_service.py` already imports `WORK_DIR, query_embedding_bundle_dir`; `rerank_service.py` already imports `WORK_DIR, rerank_score_cache_path`).
- `src/seed_pipeline/integrations/kaggle/service.py`: delete the line `GGUF_ROOT = PROJECT_ROOT.parent / "ai-models" / "gguf"` (renamed by the `sed`) and change line 8 to `from seed_pipeline.config.paths import GGUF_ROOT, PROJECT_ROOT` (`PROJECT_ROOT` is still used for `source_root=PROJECT_ROOT / "src"`).
- `src/seed_pipeline/evaluation/preload_query_embeddings.py`: delete lines 12-13 (`DEFAULT_COMPOSE_FILE = ...`, `DEFAULT_GGUF_ROOT = ...`; nothing uses them) and change line 6 to `from seed_pipeline.config.paths import PROCESSED_EVALUATION_DIR` (`PROJECT_ROOT` was only used by those two lines).

- [ ] **Step 4: Bump the build contract to `rag-final-v3`**

In `src/seed_pipeline/artifacts/contract.py` replace the constants, `build_manifest` and `validate_contract_directory`:

```python
CONTRACT_SCHEMA_VERSION = "rag-final-v3"
CONTRACT_FILES = frozenset(
    {"sections.jsonl", "blocks.jsonl", "manifest.json", "validation_report.json"}
)
TRACKED_FILES = ("sections.jsonl", "blocks.jsonl", "validation_report.json")
```

```python
def build_manifest(
    final_dir: Path,
    *,
    build_id: str,
    source_pdf_sha256: str,
    snapshot_id: str,
    snapshot_sha256: str,
    curated_input_digests: dict[str, str],
    config_digest: str,
) -> dict[str, Any]:
    final_dir = Path(final_dir)
    sections = _jsonl_records(final_dir / "sections.jsonl")
    blocks = _jsonl_records(final_dir / "blocks.jsonl")
    report = json.loads((final_dir / "validation_report.json").read_text("utf-8"))
    tracked_files = {
        name: {
            "sha256": sha256_file(final_dir / name),
            "bytes": (final_dir / name).stat().st_size,
        }
        for name in TRACKED_FILES
    }
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "build_id": build_id,
        "source_pdf_sha256": source_pdf_sha256,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "curated_input_digests": dict(sorted(curated_input_digests.items())),
        "config_digest": config_digest,
        "section_count": len(sections),
        "block_count": len(blocks),
        "validation_ok": bool(report.get("ok")),
        "files": tracked_files,
    }


def validate_contract_directory(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        raise ContractError(f"Final contract directory is missing: {path}")
    actual = {entry.name for entry in path.iterdir()}
    if actual != CONTRACT_FILES:
        raise ContractError(
            f"Final contract must contain exactly {sorted(CONTRACT_FILES)}; "
            f"found {sorted(actual)}"
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError("Unsupported final contract schema")
    report = json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("ok") is not True or manifest.get("validation_ok") is not True:
        raise ContractError("Final contract validation report is not successful")
    sections = _jsonl_records(path / "sections.jsonl")
    blocks = _jsonl_records(path / "blocks.jsonl")
    if not sections or not blocks:
        raise ContractError("Final contract JSONL files must not be empty")
    if any(not record.get("id") or not record.get("text") for record in sections):
        raise ContractError("Every section requires id and text")
    if any(not record.get("block_id") or not record.get("section_id") for record in blocks):
        raise ContractError("Every block requires block_id and section_id")
    section_ids = {record["id"] for record in sections}
    if len(section_ids) != len(sections):
        raise ContractError("Final sections contain duplicate IDs")
    if len({record["block_id"] for record in blocks}) != len(blocks):
        raise ContractError("Final blocks contain duplicate IDs")
    orphans = sorted({str(record["section_id"]) for record in blocks} - section_ids)
    if orphans:
        raise ContractError(f"Final blocks reference unknown sections: {orphans[:5]}")
    for name in TRACKED_FILES:
        expected = manifest.get("files", {}).get(name, {})
        file_path = path / name
        if expected.get("sha256") != sha256_file(file_path):
            raise ContractError(f"Final contract checksum mismatch: {name}")
        if int(expected.get("bytes", -1)) != file_path.stat().st_size:
            raise ContractError(f"Final contract size mismatch: {name}")
    if int(manifest.get("section_count", -1)) != len(sections):
        raise ContractError("Final section count mismatch")
    if int(manifest.get("block_count", -1)) != len(blocks):
        raise ContractError("Final block count mismatch")
    return manifest
```

In `src/seed_pipeline/corpus/validation/validate_final_rag.py`: delete the `from seed_pipeline.corpus.metadata.qdrant_payload_contract import (...)` block (lines 21-24), delete the whole `validate_unified_chunks` function, and replace `combine_validation_reports` with:

```python
def combine_validation_reports(
    *,
    source_report: dict[str, Any],
    deep_report: dict[str, Any],
) -> dict[str, Any]:
    parts = (source_report, deep_report)
    return {
        "ok": all(bool(part.get("ok")) for part in parts),
        "metrics": {
            key: value
            for part in parts
            for key, value in (part.get("metrics") or {}).items()
        },
        "errors": [error for part in parts for error in part.get("errors", [])],
        "warnings": [warning for part in parts for warning in part.get("warnings", [])],
        "findings": [finding for part in parts for finding in part.get("findings", [])],
        "stages": {"source_validation": source_report, "deep_audit": deep_report},
    }
```

In `src/seed_pipeline/orchestration/build_corpus.py`: delete the `from seed_pipeline.corpus.metadata.build_rag_metadata import (...)` block and `validate_unified_chunks` from the `validate_final_rag` import; in `_digest_payload` set `"schema_version": "rag-final-v3"`; replace everything in `build_candidate` from the two `shutil.copy2` calls down to the `validation_report.json` write with:

```python
    shutil.copy2(
        paths.source_final_dir / "sections.jsonl",
        paths.candidate_final_dir / "sections.jsonl",
    )
    shutil.copy2(
        paths.canonical_dir / "blocks.jsonl",
        paths.candidate_final_dir / "blocks.jsonl",
    )
    combined = combine_validation_reports(
        source_report=source_report.to_dict(),
        deep_report=deep_report.to_dict(),
    )
    (paths.candidate_final_dir / "validation_report.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

(The source-final `chunks.jsonl` in the build workspace is still produced and read by `validate_final_rag`/`run_deep_audit`; it is a workspace file, not a published contract.)

In `src/seed_pipeline/orchestration/validation_service.py`: drop `validate_unified_chunks` from the import and replace the `try/except/else` inside `run_validation` with:

```python
        try:
            manifest = validate_contract_directory(request.final_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            report = ValidationReport(ok=False, errors=[str(exc)])
        else:
            report = ValidationReport(
                ok=True,
                metrics={
                    "schema_version": manifest["schema_version"],
                    "build_id": manifest["build_id"],
                    "section_count": manifest["section_count"],
                    "block_count": manifest["block_count"],
                },
            )
```

- [ ] **Step 5: Verify nothing references the removed contracts**

```bash
cd /home/andv/personal/thesis
grep -rn -e vector_store -e qdrant_payload_contract -e build_rag_metadata -e term_enrichment \
  -e rag_app_schema -e RAG_FINAL_CHUNKS_PATH -e VECTOR_EMBEDDING_CACHE_DIR -e "vectors upload" \
  -e "embed chunks" -e DEFAULT_QDRANT_URL -e unified_report \
  seed-pipeline/src seed-pipeline/tests seed-pipeline/README.md seed-pipeline/docs
grep -n "COLLECTION_ALIAS" docker-compose.yml backend/.env.example
```

Expected: the first grep prints only matches inside the docs files that Step 6 rewrites; after Step 6 it prints nothing. The second grep prints nothing (docker-compose never set the alias; P3 removed it from `backend/.env.example`).

- [ ] **Step 6: Rewrite the docs**

Replace `seed-pipeline/README.md`:

````markdown
# seed-pipeline

Công cụ offline tạo dữ liệu gốc cho backend `pharma-agent` từ nguồn nội bộ: **Dược thư Quốc gia Việt Nam** (PDF) và snapshot tờ hướng dẫn sử dụng **An Khang**. Đầu ra là một knowledge bundle `knowledge-bundle/v1`; backend nạp bundle bằng `pharma-agent corpus import`. seed-pipeline không ghi thẳng vào Postgres hay Qdrant.

## Bắt đầu

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv sync    # cài luôn backend `pharma-agent` (path dependency, editable)
```

| Trường hợp | Workflow |
| --- | --- |
| Build ở local, Kaggle GPU embed bundle và chấm rerank | [Local + Kaggle GPU](docs/guides/workflow-local-kaggle.md) |
| Không dùng Kaggle; mọi model chạy local | [Local CPU-only](docs/guides/workflow-local-only.md) |
| Chuyển từ corpus-pipeline cũ (một lần, môi trường dev) | [Migration 2026-09](docs/guides/migration-2026-09.md) |

## Luồng dữ liệu

```text
seed build             PDF + snapshot An Khang -> data/heavy/processed/rag-final/ (sections + blocks)
seed validate          kiểm tra contract rag-final-v3
seed bundle export     rag-final -> data/heavy/bundles/formulary/ (knowledge-bundle/v1)
seed bundle embed      chunk_section của backend -> embeddings/<model_slug>.jsonl (Kaggle hoặc local)
pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish   (chạy trong backend/)
seed evaluation build  bundle -> bộ câu hỏi gold (section key)
seed embed queries     bộ gold -> cache vector query (Kaggle hoặc local)
seed retrieve          RetrievalService của backend -> candidate bundle của run
seed rerank            chấm reranker trên candidate bundle (local hoặc Kaggle)
seed metrics           Hit@K, MRR từ artifact đã đóng băng
```

## Mental model cho retrieval evaluation

- `--run` là tên workspace/thí nghiệm dùng chung cho retrieve, rerank và metrics; tên này không chọn thuật toán.
- `seed retrieve` gọi `build_retrieval_service(settings, embedder=...)` của backend trên Postgres và Qdrant dev đã import và publish release. Settings đọc từ `../backend/.env` (`--backend-env-file`). `--retriever` là `bm25`, `dense` hoặc `hybrid`; `--prefetch-k`, `--candidate-k`, `--rrf-k` được đưa vào cấu hình retrieval của backend. Với `dense` và `hybrid`, vector query lấy từ cache của `seed embed queries` (`--query-embeddings`); thiếu query nào thì lệnh dừng và báo, không embed lại qua endpoint.
- Run identity ghi `release_id`, `chunker_version` và `embedding_model`; release khác thì dùng tên run mới.
- `--model` của `seed rerank` là reranker model; `seed metrics` không chạy model.

### Re-judge metrics trên artifacts hiện có

Khi chỉ sửa judgment mà không đổi query text, có thể chạy lại evaluation mà không retrieve hoặc rerank lại. Lệnh mặc định là dry-run:

```bash
uv run seed evaluation rejudge-current \
  --dense-run dense-qwen4b-k30 \
  --hybrid-run hybrid-qwen4b-p50-k30-rrf2
```

Thêm `--apply` để thay evaluation dataset, cập nhật hash, xoá metrics reports cũ của hai run và tạo lại baseline cùng rerank reports từ candidates/rerank scores đang có.

## Prerequisites chung

- `uv`, Docker và Docker Compose.
- Raw PDF và snapshot An Khang dưới `data/heavy/raw/`; mappings, glossary, curated tables, URL lists và danh sách âm tiết dưới `data/resources/`; source manifests dưới `data/manifests/source/`.
- GGUF dưới `../ai-models/gguf/`.
- `backend/.env` cho `seed retrieve` và `pharma-agent corpus import`; `.env` của seed-pipeline cho Kaggle.

## Đầu ra

- `data/heavy/processed/rag-final/`: `sections.jsonl`, `blocks.jsonl`, `manifest.json`, `validation_report.json` (contract `rag-final-v3`); bản sao manifest nhỏ ở `data/manifests/corpus/`.
- `data/heavy/bundles/formulary/`: knowledge bundle bàn giao cho backend.
- `data/heavy/cache/text_embeddings/<model-slug>.jsonl`: cache embedding theo `sha256(embedding_text)`, dùng lại giữa các lần build.
- `data/heavy/cache/query_embeddings/`, `data/heavy/cache/rerank_scores/`: cache của `seed embed queries` và `seed rerank`.
- `data/retrieval_eval/<run>/` và `data/heavy/retrieval_eval/<run>/`: run metadata, candidates, rerank variants, reports.

Chính sách bàn giao và artifact: [Downstream](docs/guides/downstream.md).

## Cấu trúc project

- `src/seed_pipeline/orchestration/`: build và atomic publish `rag-final`.
- `src/seed_pipeline/corpus/`: trích xuất PDF, crawl An Khang, canonical blocks, bảng Docling, validation.
- `src/seed_pipeline/bundle/`: export, parity, embed và evaluation chunks dựa trên `pharma_agent.domain.corpus`.
- `src/seed_pipeline/embeddings/`: cache embedding theo hash và backend local/Kaggle.
- `src/seed_pipeline/integrations/kaggle/`: reconciliation, checkpoint và worker trên Kaggle (worker không import backend).
- `src/seed_pipeline/evaluation/`: dataset, retrieval qua backend, rerank, metrics.
- `src/seed_pipeline/cli/`: entrypoint `uv run seed`.

## CLI

Entrypoint duy nhất là `uv run seed`. Xem [CLI reference](docs/guides/cli-reference.md); `uv run seed COMMAND --help` cho default thực tế.

## Kiểm tra

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check --min-severity warn
uv run pytest -q
uv run pytest -q -m data     # cần build thật dưới data/heavy (xem migration guide)
```
````

Replace `seed-pipeline/docs/guides/downstream.md`:

```markdown
# Downstream và Data Artifact Policy

## 1. Bàn giao cho backend

1. Đầu ra bàn giao duy nhất của seed-pipeline là knowledge bundle `knowledge-bundle/v1` do `seed bundle export` tạo và `seed bundle embed` bổ sung vector: `manifest.json`, `documents.jsonl`, `sections.jsonl`, `glossary.json`, `colloquial_mappings.json`, `embeddings/<model_slug>.jsonl`. Định dạng và quy tắc validate: spec `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §5 và model `pharma_agent.domain.corpus.bundle`.
2. Document key: `drug:<slug>`, `general:<slug>`, `leaflet:ankhang:<category>:<slug>` (`source.url` của tờ hướng dẫn trỏ về `https://www.nhathuocankhang.com/<category>/<slug>`). Section key giữ nguyên `section_id` cũ để bộ gold evaluation dùng tiếp.
3. Block `kind` (`prose`, `table`, `list`, `index_entries`) và section `retrieval` (`default`, `index_only`) thay cho các ID viết cứng trước đây (`BRAND_INDEX_SECTION_ID`, `APPENDIX_LIST_SECTION_IDS`).
4. Chia chunk, `context_header`, `embedding_text`, thuật ngữ và colloquial mapping do `chunk_section` của backend tính. seed-pipeline import backend như thư viện và không có chunker riêng.
5. Vector được tính trước theo `(model, sha256(embedding_text))`. Backend chỉ dùng vector có model và số chiều khớp setting; phần thiếu backend tự embed qua endpoint đã cấu hình. Kernel Kaggle chỉ nhận text và hash, không import backend.
6. Chạy `uv run seed validate` trước `seed bundle export`. Export và embed ghi bundle vào thư mục tạm, validate bằng `read_bundle`, rồi mới thay thư mục đích.
7. Hydrate (`full_section`, `chunk_window`, `search_only`) là policy của backend (`pharma_agent.domain.corpus.hydrate`), không nằm trong bundle.
8. Đổi thuật toán chunk trong backend thì backend tăng `CHUNKER_VERSION`; import lại bundle tạo release mới.
9. Evaluation retrieval chạy qua `RetrievalService` của backend nên đo đúng retrieval lúc chạy; vector query lấy từ cache của `seed embed queries`, không embed lại; gold label vẫn theo section key.
10. Corpus lúc chạy nằm trong schema `corpus` của backend (Postgres là nguồn chính, Qdrant là index dẫn xuất), nạp bằng `pharma-agent corpus import <bundle_dir> --collection formulary --publish`. seed-pipeline không ghi thẳng vào Postgres hay Qdrant.

## 2. Data Artifact Policy

- **Source / provenance inputs**: inputs nhỏ trong `data/resources/`, source manifests trong `data/manifests/source/`, PDF và snapshot lớn trong `data/heavy/raw/`.
- **Rebuildable workspace**: `data/heavy/.work/` trong lúc build (tự xoá sau publish; workspace lỗi giữ ở `.work/failed/`), cùng input embedding `data/heavy/.work/bundle-embed/` và chunk evaluation `data/heavy/.work/evaluation-chunks/`.
- **Final reproducible contracts**: `data/heavy/processed/rag-final/` (contract `rag-final-v3`), snapshot manifest trong `data/manifests/corpus/`, bundle trong `data/heavy/bundles/`.
- **Caches**: `data/heavy/cache/text_embeddings/`, `query_embeddings/`, `rerank_scores/`; mỗi record có checksum.
- **Local experiment outputs**: metadata/report nhỏ trong `data/retrieval_eval/`; candidates, rerank bundles và per-query reports trong `data/heavy/retrieval_eval/`.
```

Replace `seed-pipeline/docs/guides/cli-reference.md`:

````markdown
# CLI reference

Public entrypoint:

```text
uv run seed
```

Các command chính:

```text
seed doctor --backend local|kaggle
seed build
seed validate
seed bundle export --output DIR
seed bundle parity --bundle DIR --old-chunks FILE
seed bundle embed --bundle DIR --backend local|kaggle --model MODEL
seed evaluation build [--bundle DIR]
seed evaluation rejudge-current --dense-run NAME --hybrid-run NAME
seed embed queries --backend local|kaggle
seed retrieve --run NAME
seed rerank --run NAME --backend local|kaggle --model MODEL
seed metrics --run NAME
```

Global `--json` in envelope máy đọc được, `--debug` bật traceback. `uv run seed COMMAND --help` cho default thực tế.

## Bundle

| Command | Việc |
| --- | --- |
| `seed bundle export --output DIR [--rag-final-dir DIR] [--glossary FILE] [--mappings FILE] [--force]` | Đọc `rag-final/{sections,blocks,manifest}` và resources, xuất `knowledge-bundle/v1`, validate bằng `read_bundle` trước khi thay `DIR` |
| `seed bundle parity --bundle DIR --old-chunks FILE [--report FILE] [--max-chars N]` | So `chunk_section` của backend với `chunks.jsonl` của build cũ (chunk_text, trang, hydrate_strategy, embedding_text, term_annotations, colloquial_mapping, thứ tự); exit 1 nếu lệch |
| `seed bundle embed --bundle DIR --backend local\|kaggle --model MODEL [--cache FILE] [--work-dir DIR] [--kaggle-account accN] [--dry-run] [--force]` | Gom cặp `(embedding_text_sha256, embedding_text)` duy nhất, embed phần thiếu, ghi `embeddings/<model_slug>.jsonl` và manifest; exit 3 nếu Kaggle chưa xong (chạy lại để resume) |

`--model` của `bundle embed` phải trùng `PHARMA_RETRIEVAL__EMBEDDING__MODEL` của backend, ví dụ `qwen3-embedding:4b-fp16`. Cache nằm ở `data/heavy/cache/text_embeddings/<model-slug>.jsonl`; Kaggle dùng dataset input `seed-pipeline-bundle`, stage `corpus-embed` contract version 3.

## Run, retriever và model

`--run` dùng cùng một tên dưới hai cây: metadata ở `data/retrieval_eval/<run>/` và payload lớn ở `data/heavy/retrieval_eval/<run>/`. Retrieve, rerank và metrics mở lại cùng workspace và artifact đã đóng băng.

`seed retrieve` dựng retrieval stack của backend bằng `build_retrieval_service(settings, embedder=...)`:

- Settings đọc từ `--backend-env-file` (mặc định `../backend/.env`), gồm Postgres, Qdrant, tên và số chiều của model embedding.
- Lệnh ghi đè `retrieval.mode`, `candidate_k`, `prefetch_k`, `rrf_k`, `collections = [--collection]` (mặc định `formulary`), tắt rerank (`protocol = none`) và Langfuse.
- `dense` và `hybrid` dùng vector query có sẵn trong cache của `seed embed queries` (`--query-embeddings`, mặc định `data/heavy/cache/query_embeddings/<model-slug>.jsonl` theo `PHARMA_RETRIEVAL__EMBEDDING__MODEL`). Thiếu vector của query nào thì lệnh dừng trước khi retrieve; evaluation không bao giờ embed query qua endpoint. `bm25` không cần vector query.
- Phải có release đã import và publish; release lấy từ query đầu tiên và mọi hit sau phải cùng release.
- Run identity: evaluation path/sha256, collection Qdrant, `embedding_model`, `query_embeddings_sha256`, `retriever`, K, `limit`, `release_id`, `chunker_version`.

| Retriever | Query embeddings | Qdrant | Candidate semantics |
| --- | --- | --- | --- |
| `bm25` | Không dùng | BM25 sparse | `candidate-k` là retrieval limit |
| `dense` | Bắt buộc, từ cache | dense | `candidate-k` là retrieval limit |
| `hybrid` | Bắt buộc, từ cache | dense + BM25 sparse, RRF | `prefetch-k` mỗi nhánh, `candidate-k` sau RRF |

```bash
uv run seed retrieve \
  --run backend-hybrid-qwen4b-p50-k30-rrf2 \
  --retriever hybrid \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 2
```

Candidate `chunk_id` là nhãn vị trí `<section_key>:chunk-<ordinal:03d>`, `document_text` là `embedding_text` của chunk (dùng cho rerank).

## Rerank và metrics

`seed rerank` chỉ đọc complete candidate bundle của run; với Kaggle, stage chỉ load reranker và chấm candidate pairs. Mỗi lần rerank hoàn tất tạo một score variant bất biến dưới run.

```bash
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account acc2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Metrics gồm Hit@3/5/10/30, MRR, và với `answer_mode=multi_required`: Multi-section Recall@K, Multi-all-hit@K. `--top-k` không được lớn hơn `candidate-k`.

`seed embed queries --backend local|kaggle --model MODEL` tạo cache vector query (`data/heavy/cache/query_embeddings/<model-slug>.jsonl`) mà `seed retrieve` dùng cho `dense` và `hybrid`; `--model` phải trùng model embedding của backend. Candidate `document_text` là `embedding_text` của hit, đúng văn bản reranker của benchmark cũ chấm.

## Runtime profiling trên Kaggle

Stage production tự benchmark workload một lần nếu chưa có profile hợp lệ và lưu dưới `data/heavy/runtime_kaggle_profiles/`; profile mất hiệu lực khi model, runtime, topology hoặc search space đổi. Lỗi benchmark dừng pipeline.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Stage completed successfully |
| 1 | Runtime or validation failure (bao gồm parity lệch) |
| 2 | Invalid CLI usage |
| 3 | Resumable incomplete stage |
| 130 | Interrupted by the operator |
````

Replace `seed-pipeline/docs/guides/workflow-local-kaggle.md`:

````markdown
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
uv sync
uv run seed doctor --backend kaggle --kaggle-account acc1
docker compose -f ../docker-compose.yml up -d postgres qdrant llama-embedding
```

Production stage ở profile mode tự tìm checkpoint tương thích trong mọi profile `accN` và mirror sang owner đích trước khi submit kernel.

## 2. Build, validate và export bundle

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/heavy/bundles/formulary --force
```

## 3. Embed bundle trên Kaggle

```bash
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle \
  --model qwen3-embedding:4b-fp16 --kaggle-account acc1 --dry-run
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle \
  --model qwen3-embedding:4b-fp16 --kaggle-account acc1
```

Lệnh stream log kernel, tải artifact về và merge vào `data/heavy/cache/text_embeddings/qwen3_embedding_4b_fp16.jsonl`. Exit code 3 nghĩa là kernel chưa xong hoặc hết budget: chạy lại đúng lệnh để resume từ checkpoint. Khi cache đủ, lệnh ghi `embeddings/qwen3_embedding_4b_fp16.jsonl` vào bundle. `Ctrl-C` chỉ dừng theo dõi; kernel vẫn chạy và lần chạy sau tự attach.

## 4. Import vào backend

```bash
cd ../backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish
uv run pharma-agent corpus releases --collection formulary
cd ../seed-pipeline
```

## 5. Evaluation dataset và retrieval

```bash
uv run seed evaluation build --bundle data/heavy/bundles/formulary
uv run seed embed queries --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
uv run seed retrieve --run backend-bm25-k30 --retriever bm25 --candidate-k 30
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid \
  --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

Thêm `--limit 50` và tên run `...-smoke50` để chạy thử trước.

## 6. Rerank trên Kaggle và metrics

```bash
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account acc2 --dry-run
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account acc2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Kaggle không truy cập Postgres/Qdrant và không tính metrics; rerank chỉ nhận candidate JSONL cùng manifest.

## Troubleshooting

- Credential/owner lỗi: sửa `.env`, chạy lại `seed doctor --backend kaggle`.
- `checksum mismatch` hoặc `no mounted file found`: dataset input chưa `READY` hoặc lệch; chạy lại để reconciler publish đúng version.
- `Expected hits from exactly one published release`: chưa import/publish release, hoặc `retrieval.collections` trỏ sai collection.
- `pinned to release`: release hiện hành đổi giữa chừng; dùng tên run mới.
- `Query embedding cache ... is missing`: chạy lại `seed embed queries --backend kaggle` với đúng file evaluation và model embedding của backend.
- `--top-k` lớn hơn `candidate-k`: retrieve lại với candidate depth đủ lớn.

Chi tiết lệnh: [CLI reference](cli-reference.md). Chính sách bàn giao: [Downstream](downstream.md).
````

Replace `seed-pipeline/docs/guides/workflow-local-only.md`:

````markdown
# Local-only workflow

Mọi model chạy local qua llama.cpp trong `../docker-compose.yml`; không dùng Kaggle.

## 1. Kiểm tra môi trường

```bash
uv sync
uv run seed doctor --backend local
docker compose -f ../docker-compose.yml up -d postgres qdrant llama-embedding llama-reranker
```

## 2. Build, validate, export và embed

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/heavy/bundles/formulary --force
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend local --model qwen3-embedding:4b-fp16
```

Embed local chạy trên CPU nên chậm; lệnh ghi cache sau mỗi lô, chạy lại để tiếp tục. Có thể bỏ bước embed: `pharma-agent corpus import` tự embed phần thiếu qua endpoint của backend.

## 3. Import vào backend

```bash
cd ../backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish
cd ../seed-pipeline
```

## 4. Evaluation

```bash
uv run seed evaluation build --bundle data/heavy/bundles/formulary
uv run seed embed queries --backend local --model qwen3-embedding:4b-fp16
uv run seed retrieve --run backend-bm25-k30 --retriever bm25 --candidate-k 30
uv run seed metrics --run backend-bm25-k30 --top-k 30
uv run seed retrieve --run backend-dense-qwen4b-k30 --retriever dense --candidate-k 30
uv run seed metrics --run backend-dense-qwen4b-k30 --top-k 30
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid \
  --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend local --model qwen3-reranker:4b-fp16
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

Mỗi baseline cần ít nhất 30 candidates/query để tính Hit@30. Thêm `--limit 50` với tên run riêng để smoke test; bỏ `--limit` làm đổi run identity.

## Troubleshooting

- `Expected hits from exactly one published release`: import và publish bundle trước.
- `Query embedding cache ... is missing`: chạy `seed embed queries --backend local` với đúng file evaluation và model embedding của backend.
- Run identity xung đột: chọn `--run` mới thay vì ghi đè artifact cũ.

Chi tiết lệnh: [CLI reference](cli-reference.md). Chính sách bàn giao: [Downstream](downstream.md).
````

Replace `seed-pipeline/data/README.md`:

````markdown
# Data layout

`data/heavy/` is the local payload boundary. It is ignored by Git and contains raw binaries, the published build (`processed/rag-final/`), knowledge bundles (`bundles/`), caches, build workspaces, migration copies (`migration/`) and retrieval candidates/rerank/per-query artifacts. Archive it as one unit when moving data to Drive:

```bash
zip -r seed-pipeline-heavy.zip data/heavy
```

Restore by unpacking the archive at the project root so that the directory is again `data/heavy/`.

Tracked data is intentionally small:

- `resources/`: maintained mappings, glossary, curation inputs, URL lists, and syllable data.
- `manifests/`: source and corpus provenance snapshots.
- `retrieval_eval/`: run metadata, candidate-manifest snapshots, and Markdown summaries. Large run payloads with the same run name are under `heavy/retrieval_eval/`.

`heavy/runtime_kaggle_profiles/` is generated for a particular runtime and is included in the heavy archive; it can be regenerated when the profile is missing or invalid.
````

- [ ] **Step 7: Run the tests and the reference check**

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv run pytest -q tests/test_removed_contracts.py tests/artifacts/test_contract.py
cd .. && grep -rn -e vector_store -e qdrant_payload_contract -e build_rag_metadata -e term_enrichment \
  -e rag_app_schema -e RAG_FINAL_CHUNKS_PATH -e VECTOR_EMBEDDING_CACHE_DIR -e "vectors upload" \
  -e "embed chunks" -e DEFAULT_QDRANT_URL -e unified_report \
  seed-pipeline/src seed-pipeline/tests seed-pipeline/README.md seed-pipeline/docs
```

Expected: tests pass; grep prints nothing.

- [ ] **Step 8: Run the full check**

`cd seed-pipeline && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 9: Commit**

```bash
cd /home/andv/personal/thesis
git add -A seed-pipeline/src seed-pipeline/tests seed-pipeline/README.md seed-pipeline/docs seed-pipeline/data/README.md
git commit -m "refactor(seed): remove the unified chunk and Qdrant upload contracts"
```

The commit message ends with the session attribution trailer.

---

### Task 11: Migration runbook and documented-command check

Spec §11 steps 1–7 as an operator runbook with exact commands, plus a test that every `uv run seed ...` line in the seed-pipeline docs still parses against the CLI (so the runbook and guides cannot drift from the code). Acceptance values come from the tracked reports of run `hybrid-qwen4b-p50-k30-rrf2` (`data/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/reports/`, 10 000 queries): baseline RRF Hit@10 95.67%, MRR 0.7242; with `qwen3-reranker:4b-fp16` (the backend default reranker) Hit@10 96.98%, MRR 0.8060.

**Files:**
- Create: `seed-pipeline/docs/guides/migration-2026-09.md`
- Create: `seed-pipeline/tests/test_docs.py`
- Modify: `seed-pipeline/README.md` (already links the runbook since Task 10; no change needed if the link is present)

**Interfaces:**
- Consumes: CLI from Tasks 1–10; backend CLI (P2): `pharma-agent migrate`, `pharma-agent corpus import <bundle_dir> --collection <key> [--publish]`, `pharma-agent corpus releases [--collection <key>]`, `pharma-agent check`.
- Produces: `docs/guides/migration-2026-09.md`; `tests/test_docs.py::test_documented_seed_commands_parse`, `test_migration_runbook_covers_every_spec_step`.

- [ ] **Step 1: Write the failing test**

Create `seed-pipeline/tests/test_docs.py`:

```python
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from seed_pipeline.cli.app import app
from seed_pipeline.config.paths import PROJECT_ROOT

GUIDES = PROJECT_ROOT / "docs" / "guides"
DOCS = (PROJECT_ROOT / "README.md", *sorted(GUIDES.glob("*.md")))


def _code_lines(text: str) -> list[str]:
    lines: list[str] = []
    inside = False
    pending = ""
    for raw in text.splitlines():
        if raw.lstrip().startswith("```"):
            inside = not inside
            continue
        if not inside:
            continue
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        lines.append((pending + raw).strip())
        pending = ""
    return lines


def _seed_commands() -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for doc in DOCS:
        for line in _code_lines(doc.read_text(encoding="utf-8")):
            if line.startswith("uv run seed "):
                commands.append((doc.name, shlex.split(line, comments=True)[3:]))
    return commands


@pytest.mark.parametrize(("doc", "args"), _seed_commands())
def test_documented_seed_commands_parse(doc: str, args: list[str]) -> None:
    result = CliRunner().invoke(app, [*args, "--help"])

    assert result.exit_code == 0, (doc, args, result.output)


def test_migration_runbook_covers_every_spec_step() -> None:
    text = (GUIDES / "migration-2026-09.md").read_text(encoding="utf-8")

    for step in range(1, 8):
        assert f"\n## {step}. " in text
    for required in (
        "uv run seed bundle parity",
        "uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle",
        "pharma-agent corpus import",
        "hybrid-qwen4b-p50-k30-rrf2",
        "95.98%",
        "0.7960",
    ):
        assert required in text, required
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd /home/andv/personal/thesis/seed-pipeline && uv run pytest -q tests/test_docs.py`
Expected: `test_migration_runbook_covers_every_spec_step` FAILS with `FileNotFoundError: .../docs/guides/migration-2026-09.md`; the parametrized command checks pass (they cover the Task 10 docs).

- [ ] **Step 3: Write the runbook**

Create `seed-pipeline/docs/guides/migration-2026-09.md`:

````markdown
# Chuyển đổi 2026-09: corpus-pipeline → seed-pipeline và corpus platform

Runbook một lần cho môi trường dev theo spec `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §11. Postgres và Qdrant dev bị xoá và dựng lại; không có backfill. Mọi lệnh `uv run seed` chạy trong `seed-pipeline/`, lệnh `pharma-agent` chạy trong `backend/`.

Cần sẵn: bản archive `data/heavy/` (raw PDF, snapshot An Khang, bộ gold `processed/evaluation/section_retrieval_eval.jsonl`) giải nén vào `seed-pipeline/data/heavy/`, `seed-pipeline/.env` có profile Kaggle, `backend/.env` có Postgres, Qdrant và embedding endpoint.

## 1. Backend

Backend đã có schema `corpus`, domain corpus, `pharma-agent corpus import`, layout Qdrant mới, `PostgresHydrator`, audit và health check (plan P1–P3 đã merge).

```bash
cd /home/andv/personal/thesis/backend
uv sync
uv run pytest -q
uv run pytest -q -m integration
```

## 2. seed-pipeline: đổi tên và export

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv sync
uv run seed doctor --backend kaggle --kaggle-account acc1
```

Export chạy trong bước 3, trên cả build cũ lẫn build mới.

## 3. Kiểm tra khớp chunk (bắt buộc 100%)

Parity so `chunk_section` của backend với `chunks.jsonl` do code cũ sinh ra. Code cũ bị xoá ở commit `refactor(seed): remove the unified chunk and Qdrant upload contracts`, nên build cũ chạy trên commit ngay trước đó:

```bash
cd /home/andv/personal/thesis
PRE_REMOVAL=$(git log --format=%H -1 --grep="evaluate retrieval through the backend with cached query vectors")
git switch --detach "$PRE_REMOVAL"
cd seed-pipeline
uv sync
```

```bash
uv run seed build
uv run seed validate
mkdir -p data/heavy/migration
cp data/heavy/processed/rag-final/chunks.jsonl data/heavy/migration/rag-final-chunks.jsonl
uv run seed bundle export --output data/heavy/bundles/formulary --force
uv run seed bundle parity --bundle data/heavy/bundles/formulary --old-chunks data/heavy/migration/rag-final-chunks.jsonl --report data/heavy/migration/parity-pre-removal.json
uv run pytest -q -m data tests/bundle/test_parity_data.py
```

Kết quả phải là `mismatches=0` và test `data` pass. Nếu lệch, `parity-pre-removal.json` liệt kê section, ordinal, field, giá trị cũ và mới; sửa chunker trong backend (P1) hoặc quy tắc export (`seed_pipeline/bundle/export.py`) rồi chạy lại, không chuyển sang bước 4.

Quay lại nhánh làm việc, build lại theo contract `rag-final-v3` và kiểm tra lần nữa với cùng bản sao `chunks.jsonl`:

```bash
cd /home/andv/personal/thesis
git switch -
cd seed-pipeline
uv sync
```

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/heavy/bundles/formulary --force
uv run seed bundle parity --bundle data/heavy/bundles/formulary --old-chunks data/heavy/migration/rag-final-chunks.jsonl --report data/heavy/migration/parity-final.json
```

## 4. Embedding

```bash
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1 --dry-run
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
```

Exit code 3 là kernel chưa xong: chạy lại đúng lệnh để resume. Khi xong, bundle có `embeddings/qwen3_embedding_4b_fp16.jsonl` và `manifest.json` liệt kê file này. `--model` phải trùng `PHARMA_RETRIEVAL__EMBEDDING__MODEL` của backend.

## 5. Reset dev và import

Dừng backend nếu đang chạy, xoá collection Qdrant cũ (alias đi cùng collection), dựng lại Postgres từ migration rồi import:

```bash
cd /home/andv/personal/thesis
docker compose stop backend
for name in $(curl -s http://localhost:6333/collections | python3 -c 'import json, sys; print("\n".join(c["name"] for c in json.load(sys.stdin)["result"]["collections"] if c["name"].startswith("thesis_chunks_")))'); do
  curl -s -X DELETE "http://localhost:6333/collections/$name"
done
docker compose rm -sf postgres
docker volume rm thesis_postgres_data
docker compose up -d postgres qdrant llama-embedding llama-reranker
```

```bash
cd /home/andv/personal/thesis/backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish
uv run pharma-agent corpus releases --collection formulary
uv run pharma-agent check
```

`corpus releases` phải cho một release `ready` là release hiện hành, số chunk bằng số chunk đã kiểm ở bước 3; import không gọi embedding endpoint cho chunk nào vì mọi vector có trong bundle.

## 6. Kiểm tra chất lượng

Dùng đúng bộ gold của run cũ để kết quả so được:

```bash
cd /home/andv/personal/thesis/seed-pipeline
sha256sum data/heavy/processed/evaluation/section_retrieval_eval.jsonl
```

Hash phải là `b26d9fd71a6dd28a65b571449b612c13510b7e1568bddbcfdb5156e21ac8b1a0` (ghi trong `data/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/run.json`). Không chạy lại `seed evaluation build` trong bước này.

```bash
uv run seed embed queries --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2-smoke50 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --limit 50
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account acc2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

`seed retrieve` không embed query qua endpoint: vector lấy từ `data/heavy/cache/query_embeddings/qwen3_embedding_4b_fp16.jsonl` do `seed embed queries` ghi (resume theo checkpoint, chỉ query mới hoặc đổi text mới chạy inference). Khi cache không đổi, `query_embeddings_sha256` trong `run.json` của run mới bằng `d9769fab90929314e959fb7abb75286bcc1a6436a1607d1b18fff6480d716b42` của run cũ; thiếu query nào thì `seed retrieve` dừng với `Query embedding cache ... is missing`.

So report `data/retrieval_eval/backend-hybrid-qwen4b-p50-k30-rrf2/reports/rerank/qwen3_reranker_4b_fp16/*/*/report.md` với ngưỡng (giảm tối đa 1 điểm phần trăm; MRR tối đa 0.01):

| Chỉ số | `hybrid-qwen4b-p50-k30-rrf2` + `qwen3-reranker:4b-fp16` | Ngưỡng chấp nhận |
| --- | ---: | ---: |
| Hit@10 | 96.98% | ≥ 95.98% |
| MRR | 0.8060 | ≥ 0.7960 |

Tham khảo baseline không rerank (`reports/baseline/*/report.md`): Hit@10 95.67% (≥ 94.67%), MRR 0.7242 (≥ 0.7142). Nếu dùng reranker khác, lấy số của đúng reranker đó trong `data/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/<model-slug>/`.

Không đạt ngưỡng thì dừng ở đây: kiểm tra `parity-final.json`, `corpus releases`, collection Qdrant `chunks_current` và so từng query giữa candidates cũ (`data/heavy/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/candidates/`) và mới theo `section_id`.

## 7. Dọn dẹp

Code và tài liệu cũ đã bị xoá ở commit `refactor(seed): remove the unified chunk and Qdrant upload contracts`. Sau khi bước 6 đạt:

```bash
cd /home/andv/personal/thesis/seed-pipeline
rm -rf data/heavy/cache/vector_embeddings data/heavy/.work/kaggle-chunk-embeddings
rm -rf data/heavy/migration
cd /home/andv/personal/thesis
grep -n "COLLECTION_ALIAS" docker-compose.yml backend/.env.example
```

`grep` không được in gì (`docker-compose.yml` không đặt alias; backend dùng `retrieval.qdrant_collection = "chunks_current"`). Corpus lúc chạy từ nay nạp bằng `pharma-agent corpus import` như bước 5; ghi Hit@10 và MRR đạt được vào mô tả PR của đợt chuyển đổi.
````

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/test_docs.py`
Expected: all pass (every `uv run seed ...` line in README and guides, including the runbook, exits 0 with `--help`).

- [ ] **Step 5: Run the full check**

`uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
cd /home/andv/personal/thesis
git add seed-pipeline/docs/guides/migration-2026-09.md seed-pipeline/tests/test_docs.py
git commit -m "docs(seed): add the 2026-09 migration runbook"
```

The commit message ends with the session attribution trailer.

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
| --- | --- |
| §3 folder `seed-pipeline/`, `.pre-commit-config.yaml`, root and backend README | 1 |
| §3 `pharma-agent = { path = "../backend", editable = true }`, backend never imports `seed_pipeline`, `qdrant-client>=1.19,<2`, shared ruff/pyrefly/pytest standard | 2 (worker import guard also asserts no `pharma_agent` in Kaggle kernels) |
| §5.1 bundle files, §5.3 validation before write | 4 (`write_bundle` + `read_bundle` in staging), 7 (`embeddings/<model_slug>.jsonl`, manifest) |
| §5.2 document keys, section keys unchanged, block `kind` and `retrieval` from `BRAND_INDEX_SECTION_ID`/`APPENDIX_LIST_SECTION_IDS`, An Khang `source.url` | 4 |
| §5.1 glossary copied, colloquial mappings with `section_keys` | 4 |
| §7.3 parity 100% on chunk_text, pages, hydrate_strategy, embedding_text, term_annotations, colloquial_mapping | 5 (unit on fixture, `data`-marked real check, CLI), 11 step 3 |
| §10.1 rename package/CLI/project/Kaggle slug/workspace prefixes/docs | 1 |
| §10.2 source processing, validation and deep audit kept | 3, 10 (only the unified-chunk validator is removed) |
| §10.3 `seed bundle export --output` | 3 (publish blocks), 4 |
| §10.3 `seed bundle embed --bundle --backend kaggle\|local --model`, hash checkpoints, kernel receives text only | 6, 7 |
| §10.4 `seed evaluation build` kept | 8 (chunk rows now derived from the bundle) |
| §10.4 `seed retrieve` through `build_retrieval_service(settings, embedder=...)`, modes `bm25`/`dense`/`hybrid` and K through backend retrieval config, query vectors from the `seed embed queries` cache (never re-embedded), run identity `release_id`/`chunker_version`/`embedding_model`, gold labels by section key | 9 |
| §10.4 `seed embed queries` and `seed rerank` keep working; local rerank text identical to the old benchmark | 6 (query command kept), 9 (`CachedQueryEmbedder` reads that cache; `document_text = Hit.embedding_text`) |
| §10.5 removals: `compile_unified_chunks`, `build_qdrant_payload`, `qdrant_payload_contract.py`, `vector_store/`, `rag-final/chunks.jsonl`, `vectors upload`, `embed chunks`, `rag_app_schema.sql` | 6 (`embed chunks`), 9 (old retrievers), 10 (rest) |
| §10.6 `downstream.md` rewritten, item 10 points to schema `corpus` and `pharma-agent corpus import` | 10 |
| §11 steps 1–7 with commands and the Hit@10/MRR ≤ 1 pp acceptance vs `hybrid-qwen4b-p50-k30-rrf2` + rerank | 11 |
| §12 contract row: seed tests validate exported bundles with backend models; shared fixture checked on both sides | 4, 7 (`backend/tests/fixtures/knowledge_bundle_small/`) |

### Names checked against the overview (§2, §3)

- Bundle: `BUNDLE_SCHEMA_VERSION`, `DocumentKind`, `BlockKind` (`PROSE`, `TABLE`, `LIST`, `INDEX_ENTRIES`), `RetrievalMode` (`DEFAULT`, `INDEX_ONLY`), `SourceInfo`, `DocumentRecord`, `BlockRecord`, `SectionRecord`, `GlossaryEntry`, `ColloquialMappingRecord` (`section_keys`), `BundleCollection`, `BundleGenerator`, `BundleEmbeddingFile`, `BundleManifest`, `KnowledgeBundle.embeddings`, `BundleValidationError.problems`, `model_slug`, `read_bundle`, `write_bundle`.
- Chunking and enrichment: `CHUNKER_VERSION`, `MAX_CHUNK_CHARS`, `ChunkDraft` fields (`section_key`, `ordinal`, `kind`, `chunk_text`, `context_header`, `embedding_text`, `embedding_text_sha256`, `start_page`, `end_page`, `table_key`, `term_annotations`, `colloquial`), `chunk_section(document, section, glossary, mappings, *, max_chars)`, `compose_embedding_text(*, context_header, chunk_text, colloquial, terms)`, `hydrate_strategy_for(section)`.
- Retrieval: `build_retrieval_service(settings, *, database=None, embedder=None) -> RetrievalStack` (`service`, `aclose`), `RetrievalSettings.mode` (`hybrid`, `dense`, `bm25`), `RetrievalSettings.qdrant_collection`, `RetrievalSettings.collections`, `Hit` fields of overview §3.4 (`release_id`, `section_key`, `ordinal`, `start_page: int | None`, `embedding_text`, `kind`, `table_key`, `colloquial_mapping`, `term_annotations`, `fusion_score`...). The injected embedder exposes P2's `Embedder` members (`model`, `dimension`, `embed`).
- Backend CLI: `pharma-agent corpus import <bundle_dir> --collection <key> [--publish]`, `pharma-agent corpus releases`. Fixture `backend/tests/fixtures/knowledge_bundle_small/` with `fake-embedding-4d`.
- Every task ends with the seed-pipeline check command from the Global Constraints; Task 1 also runs the backend check because it edits backend files. No task adds ignores, `# noqa`, `# type: ignore` or pyrefly suppressions, and no production code gains a test-only flag (tests inject fakes through `backend_factory` and monkeypatched module attributes).

### Placeholder and type scan

- No "TBD", "TODO" or "similar to Task N"; every code step shows the full code or the exact replaced lines.
- Types used across tasks match: `ExportRequest`/`export_bundle` (4) reused in 5, 7, 8; `iter_section_chunks`/`gold_chunk_label` (4) in 5, 7, 8, 9; `draft_view` (5) in 8; `TextEmbeddingRequest`/`TextEmbeddingResult`/`open_text_cache` (6) in 7; `DEFAULT_BUNDLE_DIR`/`MIGRATION_DIR` (5) in 8, 11; `COMPOSE_FILE`/`GGUF_ROOT` (6) in 10; `invalid_bundle` (4) in 5, 7, 8.

### Decisions and cross-plan alignment

- P1 rules followed: `colloquial_mappings.json` is a JSON array; uncurated leaflets get a record keyed by their slug and parity compares their `colloquial_mapping` without `key`; leaflet blocks are the cleaned text with paragraphs joined by a blank line and pipe tables over 3 000 characters as separate `table` blocks; `pharma_agent.domain.corpus.hydrate` is public; `write_bundle` fills `manifest.embeddings` from `bundle.embeddings`.
- A curated colloquial slug resolved by several leaflets stays one record so that `seed bundle export` guarantees unique keys (P2's primary key `(release_id, position)` does not); those leaflets share merged `product_names`, the export test asserts key uniqueness, and the parity report lists such leaflets if the full build has any.
- Evaluation injects `CachedQueryEmbedder` into `build_retrieval_service`; `bm25` runs without an embedder. The backend settings (`PHARMA_RETRIEVAL__EMBEDDING__MODEL`, `__DIMENSION`) choose which cached vectors are read.
- This revision follows overview §3.4 for the `embedder` keyword, the `bm25` mode and `Hit.embedding_text`; the P3 and P2 plan texts read during the revision still showed `build_retrieval_service(settings, *, database=None)`, modes `hybrid|dense`, `Hit` without `embedding_text` and `open_corpus_services(settings)`, and are expected to catch up with the overview.
