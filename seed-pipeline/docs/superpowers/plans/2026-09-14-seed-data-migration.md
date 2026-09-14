# Seed Data Migration, Evaluation Rebuild and Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, with checkpoints) for this plan: most steps operate on real data and stacks, several need the user's confirmation, and nothing here is suitable for parallel subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every piece of `seed-pipeline/data/` into the new five-folder layout with leaflet identifiers, rebuild the corpus and all nine evaluation runs with the new CLI from existing caches (no paid or GPU recomputation), prove the numbers against the old reports, delete the legacy data and migration-only code, and store the result in the private Kaggle archive.

**Architecture:** Phase 2 runs one-off scripts from `data/work/migration/` (Git-ignored, never committed) that read the old formats in `data/heavy/` read-only, apply one fixed identity mapping and write through the current seal and manifest helpers. Phase 3 uses only the public CLI (`pharma-agent corpus import`, `seed retrieve`, `seed rerank`, `seed metrics`) on a separate `thesis-eval` Compose project; the one OpenAI run is imported from its old candidates. Phase 4 deletes legacy data and code after explicit confirmation, then runs `seed data push` and a fresh-clone `seed data pull`.

**Tech Stack:** uv, seed-pipeline CLI and library, backend `pharma-agent` CLI, Docker Compose, Postgres 17, Qdrant 1.19, zstandard, Kaggle CLI.

**Spec:** `seed-pipeline/docs/superpowers/specs/2026-09-14-seed-data-layout-design.md` §7, §8, §10, §11, §12, §13.

**Depends on:** plans `2026-09-14-seed-data-layout-code.md` and `2026-09-14-seed-data-archive.md`, fully implemented and green on branch `seed-data-layout`.

## Global Constraints

- `data/heavy/`, `data/retrieval_eval/`, `data/manifests/corpus/` are only read until Task 12, and Task 12 deletes them only after the user confirms.
- Every script stops at the first mismatch (`STOP: …`, non-zero exit) and prints counts; never "fix" a mismatch by editing data by hand. Scripts refuse to overwrite their outputs; delete the output explicitly to rerun a step.
- Identity mapping (spec §7.1), used by every script through `identity_map.py`:
  - `brand:ankhang:X`, `leaflet:ankhang:X` → `leaflet:X` (section keys, document keys, `section_id`, `chunk_id` with its `:chunk-NNN` suffix);
  - `query_id` `ankhang-…` → `leaflet-…`;
  - labels `ankhang` → `leaflet`, `ankhang_<x>` → `leaflet_<x>` (`eval_group`, `source_family`, `source_subcategory`, `eval_tags`);
  - notes: `An Khang`, `ankhang` → `leaflet`.
  Query text is never changed (query hashes and query vectors stay valid).
- No step may start a model server or a Kaggle job. `seed bundle embed` must report `cached`, and `seed rerank --dry-run` must report `missing_pairs=0`; otherwise stop and ask the user.
- Secrets are never printed. `seed-pipeline/.env` and `backend/.env` are used as they are; copying `.env` into the verification clone uses `cp` only.
- Command locations: `seed …` runs in `seed-pipeline/` (`uv --directory seed-pipeline run seed …` from the repo root), `pharma-agent …` in `backend/`, `docker compose …` at the repo root. Scripts run as `uv --directory seed-pipeline run python data/work/migration/<script>.py`.
- **Commit** only the files a task names, after `git status --short` and `git diff --cached --name-status` show no data payload (only moves, manifests, `run.json`, `report.md`, docs and code). Messages end with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N4DN578zgtYyAbaKdYXRX4
  ```

## Script Map

| Script (`seed-pipeline/data/work/migration/`) | Task | Reads | Writes |
| --- | --- | --- | --- |
| `identity_map.py` | 1 | — | — (shared helpers, self-check) |
| `migrate_sources.py` | 2 | `heavy/raw/`, moved snapshot manifest, `sources/leaflets/urls/drug_urls.txt` | `sources/duoc-thu-quoc-gia-viet-nam.pdf`, `sources/leaflets/html/`, `sources/leaflets/manifest.json` |
| `migrate_gold.py` | 3 | `heavy/processed/evaluation/` | `evaluation/gold/` |
| `migrate_caches.py` | 4 | `heavy/cache/`, `heavy/runtime_kaggle_profiles/`, `evaluation/gold/` | `cache/` |
| `map_old_chunks.py` | 5 | `heavy/migration/rag-final-chunks.jsonl` | `work/migration/rag-final-chunks.jsonl` |
| `import_openai_run.py` | 9 | `heavy/retrieval_eval/dense-text-embedding-3-large-k30/candidates/` | `evaluation/runs/dense-text-embedding-3-large-k30/` |
| `compare_metrics.py` | 10 | `evaluation/runs/*/reports/` | `work/migration/comparison.md` |

---

## Phase 2: Move the data

### Task 1: Preconditions and the identity map

**Files:**
- Create: `seed-pipeline/data/work/migration/identity_map.py` (ignored by Git)

- [ ] **Step 1: Check the starting point**

Run from the repo root:

```bash
git status --short
git log --oneline -1
uv --directory seed-pipeline run pytest -q
df -h /home/andv/personal/thesis
```

Expected: a clean tree on `seed-data-layout` with plans A and B merged, tests passing, and at least 30 GB free (new layout ≈ 15 GB next to the 18 GB legacy folder). If a Kaggle job from the old layout is still running, stop and ask the user.

- [ ] **Step 2: Create `identity_map.py`**

```python
"""Map source-named identifiers to leaflet identifiers (spec §7.1). One-off, not committed."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

DATA = Path(__file__).resolve().parents[2]
SECTION_PREFIXES = ("brand:ankhang:", "leaflet:ankhang:")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"STOP: {message}")


def map_key(value: str) -> str:
    for prefix in SECTION_PREFIXES:
        if value.startswith(prefix):
            return "leaflet:" + value.removeprefix(prefix)
    return value


def map_query_id(value: str) -> str:
    if value.startswith("ankhang-"):
        return "leaflet-" + value.removeprefix("ankhang-")
    return value


def map_label(value: str) -> str:
    if value == "ankhang":
        return "leaflet"
    if value.startswith("ankhang_"):
        return "leaflet_" + value.removeprefix("ankhang_")
    return value


def map_note(value: str) -> str:
    return value.replace("An Khang", "leaflet").replace("ankhang", "leaflet")


def names_source(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return "ankhang" in text or "an khang" in text


class InjectiveMap:
    """Registers old -> new identifier pairs and stops when two old ones collide."""

    def __init__(self, what: str) -> None:
        self.what = what
        self.targets: dict[str, str] = {}

    def __call__(self, old: str, new: str) -> str:
        previous = self.targets.setdefault(new, old)
        if previous != old:
            fail(f"{self.what}: {previous!r} and {old!r} both map to {new!r}")
        return new


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    assert map_key("brand:ankhang:thuoc-a:x:chunk-001") == "leaflet:thuoc-a:x:chunk-001"
    assert map_key("leaflet:ankhang:thuoc-a:x") == "leaflet:thuoc-a:x"
    assert map_key("drug:paracetamol:lieu-dung") == "drug:paracetamol:lieu-dung"
    assert map_query_id("ankhang-alias-any-0007") == "leaflet-alias-any-0007"
    assert map_query_id("patient-004") == "patient-004"
    assert map_label("ankhang_brand") == "leaflet_brand"
    assert map_label("ankhang") == "leaflet"
    assert (
        map_note("An Khang patient query for brand_x")
        == "leaflet patient query for brand_x"
    )
    assert (
        map_note("coverage ankhang subsection brand_x")
        == "coverage leaflet subsection brand_x"
    )
    ids = InjectiveMap("probe")
    ids("a", "x")
    try:
        ids("b", "x")
    except SystemExit:
        pass
    else:
        raise AssertionError("collision was not detected")
    print("identity_map self-check passed")
```

- [ ] **Step 3: Run the self-check**

Run: `uv --directory seed-pipeline run python data/work/migration/identity_map.py`
Expected: `identity_map self-check passed`.

---

### Task 2: Sources

**Files:**
- Move (Git): `seed-pipeline/data/resources/*` → `seed-pipeline/data/sources/`, `seed-pipeline/data/manifests/source/ankhang-2026-07-24-97f5b5c43eee.manifest.json` → `seed-pipeline/data/sources/leaflets/manifest.json`
- Create: `seed-pipeline/data/work/migration/migrate_sources.py`
- Produces (ignored): `data/sources/duoc-thu-quoc-gia-viet-nam.pdf`, `data/sources/leaflets/html/<category>/<slug>.html`

- [ ] **Step 1: Move the tracked small files**

Run from the repo root:

```bash
cd seed-pipeline/data
mkdir -p sources/leaflets/urls
git mv resources/colloquial_mappings.json resources/term_glossary.json resources/vietnamese_valid_syllables.json sources/
git mv resources/curation sources/curation
git mv resources/ankhang/all_urls.txt resources/ankhang/drug_urls.txt sources/leaflets/urls/
git mv manifests/source/ankhang-2026-07-24-97f5b5c43eee.manifest.json sources/leaflets/manifest.json
cd ../..
git status --short seed-pipeline/data
```

Expected: only `R` (renamed) entries, and `data/resources/` is empty.

- [ ] **Step 2: Create `migrate_sources.py`**

```python
"""Copy the formulary PDF, unpack leaflet HTML and write the leaflet manifest (spec §7.2)."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import zstandard
from identity_map import DATA, fail, sha256_file

from seed_pipeline.corpus.sources.leaflet_source import (
    LeafletFile,
    LeafletManifest,
    verify_leaflet_source,
    write_leaflet_manifest,
)

OLD_PDF = DATA / "heavy/raw/duoc-thu-quoc-gia-viet-nam.pdf"
OLD_SNAPSHOT = (
    DATA / "heavy/raw/ankhang/snapshots/ankhang-2026-07-24-97f5b5c43eee.tar.zst"
)
RAG_MANIFEST = DATA / "manifests/corpus/manifest.json"
LEAFLETS = DATA / "sources/leaflets"
SITEMAP_URL = "https://www.nhathuocankhang.com/sitemap-sanpham.xml"
CRAWLED_ON = "2026-07-24"


def copy_pdf() -> None:
    expected = json.loads(RAG_MANIFEST.read_text(encoding="utf-8"))["source_pdf_sha256"]
    if sha256_file(OLD_PDF) != expected:
        fail(f"{OLD_PDF} does not match source_pdf_sha256")
    target = DATA / "sources/duoc-thu-quoc-gia-viet-nam.pdf"
    shutil.copy2(OLD_PDF, target)
    if sha256_file(target) != expected:
        fail("the copied PDF does not match source_pdf_sha256")
    print("pdf: copied and verified")


def unpack_html(old_manifest: dict[str, Any]) -> None:
    if sha256_file(OLD_SNAPSHOT) != old_manifest["archive_sha256"]:
        fail("the snapshot archive does not match its manifest")
    expected = {str(item["path"]): item for item in old_manifest["files"]}
    html = LEAFLETS / "html"
    if html.exists() and any(html.iterdir()):
        fail(f"{html} is not empty; delete it to rerun")
    seen: set[str] = set()
    with (
        OLD_SNAPSHOT.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            name = PurePosixPath(member.name)
            record = expected.get(member.name)
            if (
                record is None
                or not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or len(name.parts) != 2
                or member.name in seen
            ):
                fail(f"unexpected snapshot member {member.name!r}")
            payload = archive.extractfile(member)
            if payload is None:
                fail(f"cannot read snapshot member {member.name!r}")
            data = payload.read()
            if (
                len(data) != int(record["size"])
                or hashlib.sha256(data).hexdigest() != record["sha256"]
            ):
                fail(f"{member.name} does not match the snapshot manifest")
            target = html / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            seen.add(member.name)
    if seen != set(expected):
        fail(f"the snapshot lacks {len(set(expected) - seen)} manifest files")
    print(f"html: {len(seen)} files unpacked and verified")


def write_manifest(old_manifest: dict[str, Any]) -> None:
    drug_urls = LEAFLETS / "urls/drug_urls.txt"
    if sha256_file(drug_urls) != old_manifest["url_list_sha256"]:
        fail("drug_urls.txt does not match url_list_sha256 of the old manifest")
    url_by_path: dict[str, str] = {}
    for url in drug_urls.read_text(encoding="utf-8").split():
        parts = [part for part in urlparse(url).path.split("/") if part]
        if len(parts) == 2:
            url_by_path[f"{parts[0]}/{parts[1]}.html"] = url
    files = tuple(
        LeafletFile(
            str(item["path"]),
            int(item["size"]),
            str(item["sha256"]),
            url_by_path.get(str(item["path"])),
        )
        for item in sorted(old_manifest["files"], key=lambda item: str(item["path"]))
    )
    write_leaflet_manifest(
        LEAFLETS / "manifest.json",
        LeafletManifest(
            SITEMAP_URL, CRAWLED_ON, old_manifest["url_list_sha256"], files
        ),
    )
    source = verify_leaflet_source(LEAFLETS)
    without_url = sum(1 for item in files if item.source_url is None)
    print(
        f"manifest: {source.file_count} files verified, {without_url} without source_url"
    )


def main() -> None:
    old_manifest = json.loads((LEAFLETS / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("schema_version") != "ankhang-snapshot-v1":
        fail("sources/leaflets/manifest.json is not the moved snapshot manifest")
    copy_pdf()
    unpack_html(old_manifest)
    write_manifest(old_manifest)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `uv --directory seed-pipeline run python data/work/migration/migrate_sources.py`
Expected: `pdf: copied and verified`, `html: 2434 files unpacked and verified`, `manifest: 2434 files verified, N without source_url` (report N to the user; any N is acceptable because `source_url` is provenance only).

---

### Task 3: Gold evaluation set

**Files:**
- Create: `seed-pipeline/data/work/migration/migrate_gold.py`
- Produces (ignored): `data/evaluation/gold/section_retrieval_eval.jsonl`, `data/evaluation/gold/patient_queries.json`

- [ ] **Step 1: Create `migrate_gold.py`**

```python
"""Map the gold set and patient queries to leaflet identifiers (spec §7.1, §7.2)."""

from __future__ import annotations

import json
from typing import Any

from identity_map import (
    DATA,
    InjectiveMap,
    fail,
    jsonl,
    map_key,
    map_label,
    map_note,
    map_query_id,
    names_source,
    sha256_file,
)

OLD_GOLD = DATA / "heavy/processed/evaluation/section_retrieval_eval.jsonl"
OLD_PATIENTS = DATA / "heavy/processed/evaluation/patient_queries.json"
OLD_GOLD_SHA256 = "b26d9fd71a6dd28a65b571449b612c13510b7e1568bddbcfdb5156e21ac8b1a0"
GOLD = DATA / "evaluation/gold"

sections = InjectiveMap("section id")
chunks = InjectiveMap("chunk id")
queries = InjectiveMap("query id")


def map_row(row: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(row)
    mapped["query_id"] = queries(row["query_id"], map_query_id(row["query_id"]))
    for field in ("eval_group", "source_family", "source_subcategory"):
        if isinstance(row.get(field), str):
            mapped[field] = map_label(row[field])
    if isinstance(row.get("eval_tags"), list):
        mapped["eval_tags"] = [map_label(tag) for tag in row["eval_tags"]]
    if isinstance(row.get("notes"), str):
        mapped["notes"] = map_note(row["notes"])
    for field, register in (
        ("expected_section_id", sections),
        ("expected_chunk_id", chunks),
    ):
        if isinstance(row.get(field), str):
            mapped[field] = register(row[field], map_key(row[field]))
    for field, register in (
        ("expected_section_ids", sections),
        ("expected_chunk_ids", chunks),
    ):
        if isinstance(row.get(field), list):
            mapped[field] = [register(value, map_key(value)) for value in row[field]]
    if names_source(mapped):
        fail(f"{row['query_id']} still names the source after mapping")
    return mapped


def main() -> None:
    if sha256_file(OLD_GOLD) != OLD_GOLD_SHA256:
        fail("the old gold set is not the one the old reports used")
    GOLD.mkdir(parents=True, exist_ok=True)
    target = GOLD / "section_retrieval_eval.jsonl"
    if target.exists():
        fail(f"{target} exists; delete it to rerun")
    rows = [map_row(row) for row in jsonl(OLD_GOLD)]
    if len({row["query_id"] for row in rows}) != len(rows):
        fail("the gold set has duplicate query ids")
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    patients: list[dict[str, Any]] = []
    for row in json.loads(OLD_PATIENTS.read_text(encoding="utf-8")):
        mapped = dict(row)
        if isinstance(row.get("expected_section_id"), str):
            mapped["expected_section_id"] = sections(
                row["expected_section_id"], map_key(row["expected_section_id"])
            )
        if isinstance(row.get("notes"), str):
            mapped["notes"] = map_note(row["notes"])
        if names_source(mapped):
            fail(f"patient query {row.get('query')!r} still names the source")
        patients.append(mapped)
    patients_path = GOLD / "patient_queries.json"
    patients_path.write_text(
        json.dumps(patients, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"gold: {len(rows)} rows, sha256 {sha256_file(target)}")
    print(f"patients: {len(patients)} rows, sha256 {sha256_file(patients_path)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv --directory seed-pipeline run python data/work/migration/migrate_gold.py`
Expected: `gold: 10000 rows, sha256 …` and `patients: 500 rows, sha256 …`. Record both sha256 values for the Phase 3 commit message.

---

### Task 4: Caches and Kaggle profiles

**Files:**
- Create: `seed-pipeline/data/work/migration/migrate_caches.py`
- Produces (ignored): `data/cache/text_embeddings/<model>.jsonl` ×5, `data/cache/query_embeddings/<model>.jsonl` ×5, `data/cache/rerank_scores/<model>.jsonl` ×5, `data/cache/kaggle_profiles/rerank/<model>.json` ×4

- [ ] **Step 1: Create `migrate_caches.py`**

```python
"""Rewrite legacy caches into data/cache with leaflet identifiers (spec §7.2)."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from identity_map import DATA, InjectiveMap, fail, jsonl, map_key, map_query_id

from seed_pipeline.cache.jsonl_records import seal_record, verify_record
from seed_pipeline.evaluation.query_hash import query_hash
from seed_pipeline.runtime.catalog import EMBEDDING_MODELS, RERANKER_MODELS, ModelSpec
from seed_pipeline.runtime.runtime_profiles import RuntimeProfile

HEAVY = DATA / "heavy"
CACHE = DATA / "cache"
TEXT_COUNT = 24946


def sealed_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield verify_record(json.loads(line), path=path, line_number=number)


class SealedWriter:
    def __init__(self, target: Path, schema: str) -> None:
        if target.exists():
            fail(f"{target} exists; delete it to rerun")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.schema = schema
        self.temporary = target.with_name(f".{target.name}.migrating")
        self.handle = self.temporary.open("w", encoding="utf-8")
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        body = {
            k: v
            for k, v in record.items()
            if k not in ("cache_schema", "record_sha256")
        }
        sealed = seal_record(body, self.schema)
        self.handle.write(json.dumps(sealed, ensure_ascii=False, sort_keys=True) + "\n")
        self.count += 1

    def close(self) -> None:
        self.handle.close()
        self.temporary.replace(self.target)


def check_vector(record: dict[str, Any], spec: ModelSpec, path: Path) -> None:
    if (
        record.get("model") != spec.name
        or record.get("model_sha256") != spec.sha256
        or int(record.get("vector_dim", -1)) != spec.vector_dimension
        or len(record["embedding"]) != spec.vector_dimension
    ):
        fail(f"{path} has a record that does not match catalog {spec.name}")


def migrate_text_embeddings() -> None:
    for name, spec in EMBEDDING_MODELS.items():
        if name == "qwen3-embedding:4b-fp16":
            source = HEAVY / "cache/text_embeddings" / f"{spec.slug}.jsonl"
            digest_field = "embedding_text_sha256"
        else:
            source = HEAVY / "cache/vector_embeddings" / f"{spec.slug}.jsonl"
            digest_field = "text_hash"
        writer = SealedWriter(
            CACHE / "text_embeddings" / f"{spec.slug}.jsonl", "text-embedding-v1"
        )
        vectors: dict[str, str] = {}
        for record in sealed_records(source):
            check_vector(record, spec, source)
            digest = str(record[digest_field])
            fingerprint = hashlib.sha256(
                json.dumps(record["embedding"]).encode()
            ).hexdigest()
            if digest in vectors:
                if vectors[digest] != fingerprint:
                    fail(f"{source}: two different vectors for text {digest}")
                continue
            vectors[digest] = fingerprint
            writer.write(
                {
                    "model": spec.name,
                    "model_sha256": spec.sha256,
                    "embedding_text_sha256": digest,
                    "vector_dim": spec.vector_dimension,
                    "embedding": record["embedding"],
                    "created_at": record["created_at"],
                }
            )
        writer.close()
        if writer.count != TEXT_COUNT:
            fail(f"{name}: {writer.count} texts, expected {TEXT_COUNT}")
        print(f"cache/text_embeddings/{spec.slug}.jsonl: {writer.count} texts")


def migrate_query_embeddings(gold_rows: list[dict[str, Any]]) -> None:
    for name, spec in EMBEDDING_MODELS.items():
        source = HEAVY / "cache/query_embeddings" / f"{spec.slug}.jsonl"
        writer = SealedWriter(
            CACHE / "query_embeddings" / f"{spec.slug}.jsonl", "query-embedding-v2"
        )
        ids = InjectiveMap(f"{spec.slug} query id")
        present: set[tuple[str, str]] = set()
        for record in sealed_records(source):
            check_vector(record, spec, source)
            mapped = dict(record)
            mapped["query_id"] = ids(
                record["query_id"], map_query_id(record["query_id"])
            )
            key = (mapped["query_id"], str(record["query_hash"]))
            if key in present:
                continue
            present.add(key)
            writer.write(mapped)
        writer.close()
        missing = [
            row["query_id"]
            for row in gold_rows
            if (row["query_id"], query_hash(str(row["query"]))) not in present
        ]
        if missing:
            fail(
                f"{name}: {len(missing)} gold queries have no vector, e.g. {missing[:3]}"
            )
        print(f"cache/query_embeddings/{spec.slug}.jsonl: {writer.count} queries")


def migrate_rerank_scores() -> None:
    for name, spec in RERANKER_MODELS.items():
        if spec.rerank_contract is None:
            fail(f"{name} has no scoring contract in the catalog")
        contract = (spec.sha256, spec.rerank_contract.sha256, spec.reranker_protocol)
        writer = SealedWriter(
            CACHE / "rerank_scores" / f"{spec.slug}.jsonl", "rerank-score-v2"
        )
        scores: dict[bytes, float] = {}
        matched = 0
        for source in sorted(
            (HEAVY / "cache/rerank_scores" / spec.slug).glob("*.jsonl")
        ):
            first = next(sealed_records(source), None)
            if (
                first is None
                or (
                    first["model_sha256"],
                    first["request_contract_sha256"],
                    first["protocol"],
                )
                != contract
            ):
                print(
                    f"skip {source.relative_to(DATA)}: not the catalog contract of {name}"
                )
                continue
            matched += 1
            for record in sealed_records(source):
                if (
                    record["reranker"] != name
                    or (
                        record["model_sha256"],
                        record["request_contract_sha256"],
                        record["protocol"],
                    )
                    != contract
                ):
                    fail(f"{source} mixes scoring contracts")
                mapped = dict(record)
                mapped["query_id"] = map_query_id(record["query_id"])
                mapped["chunk_id"] = map_key(record["chunk_id"])
                key = hashlib.sha256(
                    "\0".join(
                        (
                            mapped["query_id"],
                            str(record["query_hash"]),
                            mapped["chunk_id"],
                            str(record["document_hash"]),
                        )
                    ).encode()
                ).digest()
                if key in scores:
                    if scores[key] != record["score"]:
                        fail(f"{source}: conflicting scores for one query-chunk pair")
                    continue
                scores[key] = record["score"]
                writer.write(mapped)
        writer.close()
        if matched != 1:
            fail(
                f"{name}: expected one cache file with the catalog contract, found {matched}"
            )
        print(f"cache/rerank_scores/{spec.slug}.jsonl: {writer.count} pairs")


def migrate_profiles() -> None:
    root = HEAVY / "runtime_kaggle_profiles"
    for workload in sorted(path for path in root.iterdir() if path.is_dir()):
        for model in sorted(path for path in workload.iterdir() if path.is_dir()):
            files = sorted(model.glob("*.json"))
            if len(files) != 1:
                fail(f"{model} has {len(files)} profiles, expected 1")
            RuntimeProfile.from_dict(json.loads(files[0].read_text(encoding="utf-8")))
            target = CACHE / "kaggle_profiles" / workload.name / f"{model.name}.json"
            if target.exists():
                fail(f"{target} exists; delete it to rerun")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(files[0], target)
            print(f"cache/kaggle_profiles/{workload.name}/{model.name}.json")


def main() -> None:
    gold_rows = list(jsonl(DATA / "evaluation/gold/section_retrieval_eval.jsonl"))
    migrate_text_embeddings()
    migrate_query_embeddings(gold_rows)
    migrate_rerank_scores()
    migrate_profiles()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it in the background**

Run with `run_in_background: true` (it parses about 8 GB of JSON): `uv --directory seed-pipeline run python data/work/migration/migrate_caches.py`
Expected when it exits: five `text_embeddings` lines with `24946 texts`, five `query_embeddings` lines, one `skip heavy/cache/rerank_scores/bge_reranker_v2_gemma_f16/… not the catalog contract` line, five `rerank_scores` lines and four `kaggle_profiles/rerank/…` lines, exit code 0.

- [ ] **Step 3: Confirm the new cache loaders accept the files**

Run:

```bash
uv --directory seed-pipeline run python -c "
from seed_pipeline.config.paths import query_embedding_cache_path, rerank_score_cache_path, text_embedding_cache_path
from seed_pipeline.embeddings.text_cache import TextEmbeddingCache
from seed_pipeline.evaluation.query_embedding_cache import QueryEmbeddingCache
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.runtime.catalog import EMBEDDING_MODELS, RERANKER_MODELS
for name, spec in EMBEDDING_MODELS.items():
    QueryEmbeddingCache(query_embedding_cache_path(name), vector_dim=spec.vector_dimension, model_sha256=spec.sha256)
    print('query ok', name)
for name, spec in RERANKER_MODELS.items():
    RerankScoreCache(rerank_score_cache_path(name), model_sha256=spec.sha256, request_contract_sha256=spec.rerank_contract.sha256)
    print('rerank ok', name)
print('text files', sorted(p.name for p in text_embedding_cache_path('qwen3-embedding:4b-fp16').parent.iterdir()))
"
```

Expected: ten `ok` lines and five text cache file names. (Text caches are exercised end to end by `seed bundle embed` in Task 5.)

---

### Task 5: Rebuild the corpus, embed from cache and check parity

**Files:**
- Create: `seed-pipeline/data/work/migration/map_old_chunks.py`
- Produces: `data/corpus/rag-final/` (manifest and validation report tracked), `data/corpus/formulary/` (manifest tracked)

- [ ] **Step 1: Build, validate and export**

Run in `seed-pipeline/`:

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/corpus/formulary --force
```

Expected: each ends with `status=complete`; `data/corpus/rag-final/manifest.json` has a `leaflet_source` entry with `file_count` 2434.

- [ ] **Step 2: Embed every model from the migrated caches**

Run in `seed-pipeline/`, one model at a time:

```bash
uv run seed bundle embed --bundle data/corpus/formulary --backend local --model embeddinggemma:300m
uv run seed bundle embed --bundle data/corpus/formulary --backend local --model bge-m3:567m-fp16
uv run seed bundle embed --bundle data/corpus/formulary --backend local --model qwen3-embedding:0.6b-fp16
uv run seed bundle embed --bundle data/corpus/formulary --backend local --model qwen3-embedding:4b-fp16
uv run seed bundle embed --bundle data/corpus/formulary --backend local --model qwen3-embedding:8b-fp16
```

Expected: every run reports the action `cached` and `data/corpus/formulary/embeddings/` holds five `<model>.jsonl` files. If any run reports texts to embed or tries to reach a model server, stop and ask the user.

- [ ] **Step 3: Map the old chunks and check parity**

Create `data/work/migration/map_old_chunks.py`:

```python
"""Copy the pre-migration chunks with leaflet identifiers for `seed bundle parity`."""

from __future__ import annotations

import json
from typing import Any

from identity_map import DATA, InjectiveMap, fail, jsonl, map_key

SOURCE = DATA / "heavy/migration/rag-final-chunks.jsonl"
TARGET = DATA / "work/migration/rag-final-chunks.jsonl"
chunk_ids = InjectiveMap("chunk id")


def map_strings(value: Any) -> Any:
    if isinstance(value, str):
        return map_key(value)
    if isinstance(value, list):
        return [map_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: map_strings(item) for key, item in value.items()}
    return value


def main() -> None:
    if TARGET.exists():
        fail(f"{TARGET} exists; delete it to rerun")
    count = 0
    with TARGET.open("w", encoding="utf-8") as handle:
        for record in jsonl(SOURCE):
            mapped = map_strings(record)
            chunk_ids(record["chunk_id"], mapped["chunk_id"])
            handle.write(json.dumps(mapped, ensure_ascii=False) + "\n")
            count += 1
    print(f"old chunks mapped: {count}")


if __name__ == "__main__":
    main()
```

Run:

```bash
uv --directory seed-pipeline run python data/work/migration/map_old_chunks.py
uv --directory seed-pipeline run seed bundle parity --bundle data/corpus/formulary --old-chunks data/work/migration/rag-final-chunks.jsonl --report data/work/migration/parity.json
```

Expected: `old chunks mapped: 24955`, then parity exit code 0 with 0 mismatches.

- [ ] **Step 4: Check the bundle carries no source name**

Run:

```bash
grep -rci 'ankhang\|an khang' seed-pipeline/data/corpus/formulary --include='*.json' --include='documents.jsonl' --include='sections.jsonl' | grep -v ':0$'
uv --directory seed-pipeline run python -c "
import json
docs = [json.loads(line) for line in open('data/corpus/formulary/documents.jsonl', encoding='utf-8')]
leaflets = [d for d in docs if d['key'].startswith('leaflet:')]
print('leaflet documents', len(leaflets), 'with url', sum(1 for d in leaflets if d['source'].get('url')))
"
```

Expected: the `grep` prints nothing; the Python check prints `with url 0`.

- [ ] **Step 5: Commit Phase 2**

```bash
git add -A seed-pipeline/data/resources seed-pipeline/data/manifests/source seed-pipeline/data/sources seed-pipeline/data/corpus
git status --short seed-pipeline/data
git diff --cached --name-status
git commit -m "chore(seed-data): move sources and rebuild the corpus in the new layout"
```

Expected staged entries: renames from `resources/` and `manifests/source/`, the rewritten `sources/leaflets/manifest.json`, and `corpus/rag-final/{manifest.json,validation_report.json}` plus `corpus/formulary/manifest.json`. No `.html`, `.pdf` or `.jsonl`.

---

### Task 6: Re-import the app stack

**Files:** none tracked.

- [ ] **Step 1: Import and publish the new bundle**

Run in `backend/` (uses `backend/.env`, the stack that serves the app):

```bash
uv run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish
uv run pharma-agent corpus gc --collection formulary --keep 1
uv run pharma-agent corpus releases --collection formulary
```

Expected: a new published release; gc reports one retired release and non-zero `sections deleted` and `documents deleted`; `releases` lists only the new release as current.

- [ ] **Step 2: Remove the pre-migration Qdrant collections (confirm first)**

Run: `curl -s http://localhost:6333/collections` and `curl -s http://localhost:6333/aliases`.
Show the user every collection whose name starts with `thesis_chunks_` and ask for confirmation. After a yes, delete each one:

```bash
curl -s -X DELETE "http://localhost:6333/collections/<name>"
```

Expected: the listing then shows only the backend's `chunks_<model>` collections and the `chunks_current` alias.

- [ ] **Step 3: Check the corpus schema**

Run from the repo root:

```bash
docker compose exec -T postgres pg_dump -U thesis -d thesis --schema=corpus --data-only | grep -ci 'ankhang'
```

Expected: `0`. Then run `uv run pharma-agent check` in `backend/` and ask one leaflet question through the app to confirm citations still render.

---

## Phase 3: Rebuild the evaluation runs

### Task 7: Evaluation stack and retrieval runs

**Files:** produces `data/evaluation/runs/<run>/` for eight runs.

- [ ] **Step 1: Start the evaluation stack**

Run from the repo root:

```bash
POSTGRES_PORT=5434 QDRANT_HTTP_PORT=6335 QDRANT_GRPC_PORT=6336 docker compose -p thesis-eval up -d --wait postgres qdrant
env PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis PHARMA_QDRANT__URL=http://localhost:6335 uv --directory backend run pharma-agent migrate
```

Expected: both services healthy; migrations applied. Every command in Tasks 7–8 that talks to the backend gets the same two variables plus the model variables shown below; the app stack on 5433/6333 is not touched.

- [ ] **Step 2: Dense runs for the four smaller models**

For each row, run the three commands from the repo root, in this order, finishing one model before starting the next:

| Model | Dimension | Run |
| --- | ---: | --- |
| `embeddinggemma:300m` | 768 | `dense-gemma300m-k30` |
| `bge-m3:567m-fp16` | 1024 | `dense-bge-m3-k30` |
| `qwen3-embedding:0.6b-fp16` | 1024 | `dense-qwen06b-k30` |
| `qwen3-embedding:8b-fp16` | 4096 | `dense-qwen8b-k30` |

```bash
EVAL="PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis PHARMA_QDRANT__URL=http://localhost:6335 PHARMA_RETRIEVAL__EMBEDDING__MODEL=<model> PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=<dimension>"
env $EVAL uv --directory backend run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish
env $EVAL uv --directory seed-pipeline run seed retrieve --run <run> --retriever dense --candidate-k 30
uv --directory seed-pipeline run seed metrics --run <run> --top-k 30
```

Expected: import publishes a release for that model; retrieve prints `queries=10000`; metrics writes `data/evaluation/runs/<run>/reports/baseline/top30-window3/`.

- [ ] **Step 3: The 4B model: dense, BM25 and both hybrids**

```bash
EVAL="PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis PHARMA_QDRANT__URL=http://localhost:6335 PHARMA_RETRIEVAL__EMBEDDING__MODEL=qwen3-embedding:4b-fp16 PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=2560"
env $EVAL uv --directory backend run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish
env $EVAL uv --directory seed-pipeline run seed retrieve --run dense-qwen4b-k30 --retriever dense --candidate-k 30
env $EVAL uv --directory seed-pipeline run seed retrieve --run bm25-qwen4b-k30 --retriever bm25 --candidate-k 30
env $EVAL uv --directory seed-pipeline run seed retrieve --run hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2
env $EVAL uv --directory seed-pipeline run seed retrieve --run hybrid-qwen4b-p50-k30-rrf60 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 60
for run in dense-qwen4b-k30 bm25-qwen4b-k30 hybrid-qwen4b-p50-k30-rrf2 hybrid-qwen4b-p50-k30-rrf60; do
  uv --directory seed-pipeline run seed metrics --run "$run" --top-k 30
done
```

Expected: four more runs with baseline reports.

---

### Task 8: Rerank variants from the migrated score caches

**Files:** produces `data/evaluation/runs/hybrid-qwen4b-p50-k30-rrf2/rerank/<model>/` ×5.

- [ ] **Step 1: Dry-run every reranker**

```bash
for model in bge-reranker-v2-gemma:f16 bge-reranker-v2-m3:f16 qwen3-reranker:0.6b-fp16 qwen3-reranker:4b-fp16 qwen3-reranker:8b-fp16; do
  uv --directory seed-pipeline run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --model "$model" --dry-run
done
```

Expected: each prints `missing_pairs=0`. If any model reports more, stop, tell the user the model and the count, and continue only with their decision (CPU or Kaggle run, or skip that variant).

- [ ] **Step 2: Publish the variants and their reports**

```bash
for model in bge-reranker-v2-gemma:f16 bge-reranker-v2-m3:f16 qwen3-reranker:0.6b-fp16 qwen3-reranker:4b-fp16 qwen3-reranker:8b-fp16; do
  uv --directory seed-pipeline run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --model "$model"
done
uv --directory seed-pipeline run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

Expected: each rerank prints `scored=0` and an artifact under `rerank/<model>/`; metrics lists five reranked reports.

---

### Task 9: Import the OpenAI run from its old candidates

**Files:**
- Create: `seed-pipeline/data/work/migration/import_openai_run.py`
- Produces: `data/evaluation/runs/dense-text-embedding-3-large-k30/`

- [ ] **Step 1: Create `import_openai_run.py`**

```python
"""Import the paid OpenAI dense run from its old candidates (spec §8.2), origin imported."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

from identity_map import (
    DATA,
    fail,
    jsonl,
    map_key,
    map_query_id,
    names_source,
    sha256_file,
)

from seed_pipeline.evaluation.artifact_contracts import ArtifactManifest, write_json
from seed_pipeline.evaluation.retrieval_candidate_artifact import (
    CandidateArtifactReader,
)
from seed_pipeline.evaluation.run_workspace import RunIdentity, RunWorkspace

RUN = "dense-text-embedding-3-large-k30"
OLD = DATA / "heavy/retrieval_eval" / RUN / "candidates"
GOLD = DATA / "evaluation/gold/section_retrieval_eval.jsonl"
TARGET = DATA / "evaluation/runs" / RUN


def main() -> None:
    old_manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(OLD / "candidates.jsonl") != old_manifest["data_sha256"]:
        fail("the old candidates do not match their manifest")
    if TARGET.exists():
        fail(f"{TARGET} exists; delete it to rerun")
    old = old_manifest["identity"]
    identity = RunIdentity(
        evaluation_path="evaluation/gold/section_retrieval_eval.jsonl",
        evaluation_sha256=sha256_file(GOLD),
        collection_name=str(old["collection_name"]),
        embedding_model=str(old["embedding_model"]),
        query_embeddings_sha256=None,
        retriever=str(old["retriever"]),
        candidate_k=int(old["candidate_k"]),
        rrf_k=int(old["rrf_k"]),
        limit=None,
        prefetch_k=None,
    )
    workspace = RunWorkspace.open_or_create(TARGET, identity, origin="imported")
    gold_ids = {row["query_id"] for row in jsonl(GOLD)}
    data_path = workspace.candidates_dir / "candidates.jsonl"
    data_path.parent.mkdir(parents=True)
    queries = pairs = 0
    with data_path.open("w", encoding="utf-8") as handle:
        for record in jsonl(OLD / "candidates.jsonl"):
            record["query_id"] = map_query_id(record["query_id"])
            if record["query_id"] not in gold_ids:
                fail(f"candidate query {record['query_id']} is not in the gold set")
            for item in record["candidates"]:
                item["chunk_id"] = map_key(item["chunk_id"])
                payload = item.get("payload") or {}
                for field in ("chunk_id", "section_id"):
                    if isinstance(payload.get(field), str):
                        payload[field] = map_key(payload[field])
            identifiers = {
                "query_id": record["query_id"],
                "candidates": [
                    {"chunk_id": item["chunk_id"], "payload": item.get("payload")}
                    for item in record["candidates"]
                ],
            }
            if names_source(identifiers):
                fail(f"{record['query_id']} still names the source after mapping")
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            queries += 1
            pairs += len(record["candidates"])
    if queries != len(gold_ids):
        fail(f"{queries} candidate queries for {len(gold_ids)} gold queries")
    manifest = ArtifactManifest.create(
        artifact_type="retrieval_candidates",
        data_path=data_path,
        record_count=queries,
        identity={
            **asdict(identity),
            "pair_count": pairs,
            "candidate_k": identity.candidate_k,
        },
    )
    manifest_path = data_path.with_name("manifest.json")
    write_json(manifest_path, manifest.to_dict())
    for _ in CandidateArtifactReader(data_path, manifest_path):
        pass
    workspace.record_candidates(SimpleNamespace(data_path=data_path))
    print(f"{RUN}: {queries} queries, {pairs} pairs imported")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Import and score it**

```bash
uv --directory seed-pipeline run python data/work/migration/import_openai_run.py
uv --directory seed-pipeline run seed metrics --run dense-text-embedding-3-large-k30 --top-k 30
```

Expected: `10000 queries, 300000 pairs imported`, then a baseline report. `run.json` shows `"origin": "imported"`.

---

### Task 10: Compare with the old numbers and commit Phase 3

**Files:**
- Create: `seed-pipeline/data/work/migration/compare_metrics.py`
- Produces: `data/work/migration/comparison.md`

- [ ] **Step 1: Create `compare_metrics.py`**

```python
"""Compare rebuilt reports with the pre-migration numbers (spec §8.3)."""

from __future__ import annotations

import re
import sys

from identity_map import DATA

from seed_pipeline.runtime.catalog import require_model

RUNS = DATA / "evaluation/runs"
IMPORTED = {"dense-text-embedding-3-large-k30"}
EXPECTED = [
    ("bm25-qwen4b-k30", None, 77.98, 0.5550),
    ("dense-gemma300m-k30", None, 83.12, 0.5879),
    ("dense-bge-m3-k30", None, 90.95, 0.7333),
    ("dense-qwen06b-k30", None, 91.85, 0.7042),
    ("dense-qwen4b-k30", None, 95.21, 0.7784),
    ("dense-qwen8b-k30", None, 95.14, 0.7948),
    ("dense-text-embedding-3-large-k30", None, 94.85, 0.7506),
    ("hybrid-qwen4b-p50-k30-rrf60", None, 92.45, 0.6784),
    ("hybrid-qwen4b-p50-k30-rrf2", None, 95.67, 0.7242),
    ("hybrid-qwen4b-p50-k30-rrf2", "bge-reranker-v2-gemma:f16", 95.30, 0.7570),
    ("hybrid-qwen4b-p50-k30-rrf2", "bge-reranker-v2-m3:f16", 95.04, 0.7982),
    ("hybrid-qwen4b-p50-k30-rrf2", "qwen3-reranker:0.6b-fp16", 96.18, 0.7823),
    ("hybrid-qwen4b-p50-k30-rrf2", "qwen3-reranker:4b-fp16", 96.98, 0.8060),
    ("hybrid-qwen4b-p50-k30-rrf2", "qwen3-reranker:8b-fp16", 89.75, 0.4797),
]
HIT10 = re.compile(r"^\| Hit@10 \| ([0-9.]+)% \|$", re.MULTILINE)
MRR = re.compile(r"^\| MRR \| ([0-9.]+) \|$", re.MULTILINE)


def read_report(run: str, model: str | None) -> tuple[float, float]:
    base = RUNS / run / "reports"
    folder = (
        base / "baseline"
        if model is None
        else base / "rerank" / require_model(model).slug
    )
    text = (folder / "top30-window3" / "report.md").read_text(encoding="utf-8")
    hit, mrr = HIT10.search(text), MRR.search(text)
    if hit is None or mrr is None:
        raise SystemExit(f"STOP: no Hit@10 or MRR row in {folder}")
    return float(hit.group(1)), float(mrr.group(1))


def main() -> int:
    lines = [
        "| Run / variant | Hit@10 old | Hit@10 new | MRR old | MRR new | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    failures = 0
    for run, model, old_hit, old_mrr in EXPECTED:
        new_hit, new_mrr = read_report(run, model)
        if run in IMPORTED:
            ok = abs(new_hit - old_hit) <= 0.005 and abs(new_mrr - old_mrr) <= 0.00005
        else:
            ok = new_hit >= old_hit - 1.0 and new_mrr >= old_mrr - 0.01
        failures += not ok
        label = run if model is None else f"rrf2 + {model}"
        lines.append(
            f"| {label} | {old_hit:.2f}% | {new_hit:.2f}% | {old_mrr:.4f} | "
            f"{new_mrr:.4f} | {'pass' if ok else 'FAIL'} |"
        )
    table = "\n".join(lines) + "\n"
    (DATA / "work/migration/comparison.md").write_text(table, encoding="utf-8")
    print(table)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the comparison**

Run: `uv --directory seed-pipeline run python data/work/migration/compare_metrics.py`
Expected: 14 rows, all `pass`, exit code 0. On any `FAIL`: stop, do not delete anything, and compare the old candidates (mapped with `map_key`) with the new ones per query by `section_id` for that run; report the findings to the user.

- [ ] **Step 3: Stop the evaluation stack**

Run from the repo root: `docker compose -p thesis-eval down -v`
Expected: the `thesis-eval` containers and volumes are gone; the app stack keeps running.

- [ ] **Step 4: Commit Phase 3**

```bash
git add seed-pipeline/data/evaluation/runs
git diff --cached --name-status
git commit -F - <<'EOF'
chore(seed-data): rebuild the evaluation runs with the new CLI

Rebuilt from migrated caches on the thesis-eval stack; the OpenAI run is
imported from its old candidates. Gold set sha256: <from Task 3>.

<paste data/work/migration/comparison.md>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4DN578zgtYyAbaKdYXRX4
EOF
```

Replace the two `<…>` markers with the recorded gold sha256 and the table before running. Expected staged files: `run.json`, `candidates/manifest.json`, `rerank/*/manifest.json`, `reports/**/manifest.json` and `reports/**/report.md` only.

---

## Phase 4: Clean up and archive

### Task 11: Remove migration-only code and documents

**Files:**
- Delete: `seed-pipeline/src/seed_pipeline/bundle/parity.py`, `seed-pipeline/tests/bundle/test_parity.py`, `seed-pipeline/tests/bundle/test_parity_data.py`, `seed-pipeline/docs/guides/migration-2026-09.md`
- Modify: `seed-pipeline/src/seed_pipeline/bundle/evaluation_chunks.py`, `seed-pipeline/src/seed_pipeline/bundle/__init__.py`, `seed-pipeline/src/seed_pipeline/cli/commands/bundle.py`, `seed-pipeline/src/seed_pipeline/config/paths.py`, `seed-pipeline/pyproject.toml`, `seed-pipeline/tests/cli/test_bundle_command.py`, `seed-pipeline/tests/test_docs.py`, `seed-pipeline/tests/test_removed_contracts.py`, `seed-pipeline/README.md`, `seed-pipeline/docs/guides/cli-reference.md`
- Create: `seed-pipeline/docs/guides/evaluation.md`

- [ ] **Step 1: Write the failing removed-contract tests**

In `tests/test_removed_contracts.py` append `"seed_pipeline.bundle.parity",` to `REMOVED_MODULES`, `"MIGRATION_DIR",` to `REMOVED_PATH_NAMES`, and `["bundle", "parity"],` to the command parametrization.

Run: `uv --directory seed-pipeline run pytest -q tests/test_removed_contracts.py`
Expected: FAIL for the module, the path name and the command.

- [ ] **Step 2: Remove parity and the migration runbook**

- Move `draft_view` (lines 82–100 of `bundle/parity.py`) into `bundle/evaluation_chunks.py` above its first use, replace `from seed_pipeline.bundle.parity import draft_view` with `from pharma_agent.domain.corpus.chunking import ChunkDraft` (merge into an existing import from that module if there is one).
- `git rm` `bundle/parity.py`, `tests/bundle/test_parity.py`, `tests/bundle/test_parity_data.py`, `docs/guides/migration-2026-09.md`.
- `cli/commands/bundle.py`: delete `parity_command` and the `@bundle_app.command("parity")` function with their now-unused imports.
- `tests/cli/test_bundle_command.py`: delete `test_bundle_parity_command_passes_on_the_fixture` and `test_bundle_parity_command_fails_on_a_changed_chunk`.
- `config/paths.py`: delete `MIGRATION_DIR`.
- `pyproject.toml` `[tool.pytest.ini_options]`: delete the `data` marker line and change `addopts` to `["--import-mode=importlib", "-m", "not integration"]` (no other test uses the `data` marker).
- `bundle/__init__.py` docstring: `"""Knowledge-bundle export and embeddings built on the backend corpus domain."""`.
- `tests/test_docs.py`: delete `test_migration_runbook_covers_every_spec_step`.
- `README.md`: delete the row linking `docs/guides/migration-2026-09.md` (line 17) and change "export, parity, embed và evaluation chunks" to "export, embed và evaluation chunks" (line 74).
- `docs/guides/cli-reference.md`: delete the `seed bundle parity` lines (16 and 33) and "(bao gồm parity lệch)" (line 90).

- [ ] **Step 3: Write the evaluation guide**

Create `docs/guides/evaluation.md`:

````markdown
# Đánh giá retrieval

Guide này dựng lại các số liệu retrieval của luận văn từ dữ liệu trong archive. Mọi lệnh `uv run seed` chạy trong `seed-pipeline/`, lệnh `pharma-agent` chạy trong `backend/`, lệnh `docker compose` chạy ở thư mục gốc repo.

## 1. Dữ liệu

```bash
uv run seed data pull --kaggle-account acc1
```

Lệnh tải bundle `data/corpus/formulary` (embedding của 5 model), bộ gold `data/evaluation/gold`, cache vector query và điểm rerank, kiểm sha256 từng file.

## 2. Môi trường riêng cho đánh giá

```bash
POSTGRES_PORT=5434 QDRANT_HTTP_PORT=6335 QDRANT_GRPC_PORT=6336 docker compose -p thesis-eval up -d --wait postgres qdrant
export PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis
export PHARMA_QDRANT__URL=http://localhost:6335
```

Không cần llama.cpp: vector chunk có sẵn trong bundle, vector query có sẵn trong cache, `seed retrieve` tắt rerank. Xong việc: `docker compose -p thesis-eval down -v`.

## 3. Chạy một model

Mỗi model được import và publish rồi mới chạy retrieval, vì `seed retrieve` đọc release đang publish. Chạy 4 model nhỏ trước, model `qwen3-embedding:4b-fp16` sau cùng.

```bash
export PHARMA_RETRIEVAL__EMBEDDING__MODEL=qwen3-embedding:4b-fp16
export PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=2560
uv run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish
uv run seed retrieve --run dense-qwen4b-k30 --retriever dense --candidate-k 30
uv run seed retrieve --run bm25-qwen4b-k30 --retriever bm25 --candidate-k 30
uv run seed retrieve --run hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

| Model | Chiều vector | Run dense |
| --- | ---: | --- |
| `embeddinggemma:300m` | 768 | `dense-gemma300m-k30` |
| `bge-m3:567m-fp16` | 1024 | `dense-bge-m3-k30` |
| `qwen3-embedding:0.6b-fp16` | 1024 | `dense-qwen06b-k30` |
| `qwen3-embedding:8b-fp16` | 4096 | `dense-qwen8b-k30` |
| `qwen3-embedding:4b-fp16` | 2560 | `dense-qwen4b-k30`, `bm25-qwen4b-k30`, `hybrid-qwen4b-p50-k30-rrf2`, `hybrid-qwen4b-p50-k30-rrf60` |

## 4. Rerank

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --dry-run
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

`--dry-run` in `missing_pairs=N`. Khi `N` bằng 0, rerank chỉ đọc cache điểm và không khởi động model; khi `N` lớn hơn 0 cần chạy model (CPU hoặc Kaggle) cho các cặp còn thiếu.

## 5. Run `dense-text-embedding-3-large-k30`

Run này dùng embedding API trả phí nên không chạy lại. Candidates được nhập từ lần chạy gốc (`origin: imported` trong `run.json`) và chỉ tính lại metrics.

## 6. Kết quả tham chiếu

| Run / biến thể | Hit@10 | MRR |
| --- | ---: | ---: |
| `bm25-qwen4b-k30` | 77,98% | 0,5550 |
| `dense-gemma300m-k30` | 83,12% | 0,5879 |
| `dense-bge-m3-k30` | 90,95% | 0,7333 |
| `dense-qwen06b-k30` | 91,85% | 0,7042 |
| `dense-qwen4b-k30` | 95,21% | 0,7784 |
| `dense-qwen8b-k30` | 95,14% | 0,7948 |
| `dense-text-embedding-3-large-k30` | 94,85% | 0,7506 |
| `hybrid-qwen4b-p50-k30-rrf60` | 92,45% | 0,6784 |
| `hybrid-qwen4b-p50-k30-rrf2` | 95,67% | 0,7242 |
| rrf2 + `bge-reranker-v2-gemma:f16` | 95,30% | 0,7570 |
| rrf2 + `bge-reranker-v2-m3:f16` | 95,04% | 0,7982 |
| rrf2 + `qwen3-reranker:0.6b-fp16` | 96,18% | 0,7823 |
| rrf2 + `qwen3-reranker:4b-fp16` | 96,98% | 0,8060 |
| rrf2 + `qwen3-reranker:8b-fp16` | 89,75% | 0,4797 |

Đây là số của lần đánh giá trước khi chuyển layout; lần dựng lại chấp nhận Hit@10 thấp hơn tối đa 1 điểm phần trăm và MRR thấp hơn tối đa 0,01 (xem commit "rebuild the evaluation runs" để có số mới).
````

After Task 10 passes, replace the reference table with the new numbers from `data/work/migration/comparison.md` and keep the sentence about the old numbers pointing to that commit. Add a row `| Đánh giá retrieval (dựng lại số liệu luận văn) | [Evaluation](docs/guides/evaluation.md) |` where the migration row was in `README.md`.

- [ ] **Step 4: Run the seed gate and commit**

Run in `seed-pipeline/`: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: PASS, including `tests/test_docs.py` for every `uv run seed` line of the new guide.

```bash
git add -A seed-pipeline/src seed-pipeline/tests seed-pipeline/pyproject.toml seed-pipeline/README.md seed-pipeline/docs/guides
git commit -m "refactor(seed): remove migration-only parity code and add the evaluation guide"
```

---

### Task 12: Delete the legacy data (confirm first)

**Files:**
- Delete (ignored): `seed-pipeline/data/heavy/`
- Delete (tracked): `seed-pipeline/data/retrieval_eval/`, `seed-pipeline/data/manifests/`
- Modify: `seed-pipeline/.gitignore`, `seed-pipeline/tests/test_repository_data_policy.py`

- [ ] **Step 1: Ask the user**

Show `du -sh seed-pipeline/data/heavy seed-pipeline/data/retrieval_eval seed-pipeline/data/manifests`, the Phase 3 comparison table and the list of Git-tracked files that will be removed (`git ls-files seed-pipeline/data/retrieval_eval seed-pipeline/data/manifests`). Ask for an explicit yes before deleting. Stop here without a yes.

- [ ] **Step 2: Delete and update the ignore rules**

```bash
rm -rf seed-pipeline/data/heavy
git rm -r -q seed-pipeline/data/retrieval_eval seed-pipeline/data/manifests
find seed-pipeline/data/resources seed-pipeline/data/manifests -depth -type d -empty -delete 2>/dev/null || true
```

In `seed-pipeline/.gitignore` delete the two lines

```gitignore
# Pre-migration data folder; the data migration removes this rule with the folder.
data/heavy/
```

and in `tests/test_repository_data_policy.py` delete the row `("data/heavy/probe.bin", True),`.

- [ ] **Step 3: Check and commit**

Run: `git status --short seed-pipeline/data | head` (expected: only deletions) and `uv --directory seed-pipeline run pytest -q tests/test_repository_data_policy.py` (expected: PASS).

```bash
git add -A seed-pipeline/data seed-pipeline/.gitignore seed-pipeline/tests/test_repository_data_policy.py
git diff --cached --name-status | grep -v '^D' || true
git commit -m "chore(seed-data): delete the pre-migration data layout"
```

Expected: the `grep` shows only `.gitignore` and the policy test as non-deletions.

---

### Task 13: Push the archive and prove a fresh clone restores it

**Files:** none tracked.

- [ ] **Step 1: Ask the user before the first push**

Show `du -sh` of every top-level folder in `seed-pipeline/data/`, the number of files `git ls-files --others --ignored --exclude-standard -- seed-pipeline/data | grep -v '^seed-pipeline/data/work/' | wc -l` will archive, and the Kaggle profile that will own the dataset (`KAGGLE_ACCOUNT_DEFAULT` unless the user names one). Ask for an explicit yes and the account. Stop here without a yes.

- [ ] **Step 2: Push**

Run in `seed-pipeline/` with `run_in_background: true`: `uv run seed data push --kaggle-account <account> --message "Seed data after the layout migration"`
Expected: `status=complete`, `dataset=<owner>/seed-pipeline-data`, `version=1` (or the next version), the part count. Then poll `kaggle datasets status <owner>/seed-pipeline-data` through the profile until `READY` (Kaggle processes large uploads after the command returns).

- [ ] **Step 3: Restore into a fresh clone**

```bash
CLONE=/tmp/claude-1000/-home-andv-personal-thesis/23fdb23d-5609-4bf6-942e-616429c195ab/scratchpad/thesis-clone
git clone --branch seed-data-layout /home/andv/personal/thesis "$CLONE"
cp /home/andv/personal/thesis/seed-pipeline/.env "$CLONE/seed-pipeline/.env"
uv --directory "$CLONE" sync
uv --directory "$CLONE/seed-pipeline" run seed data pull --kaggle-account <account>
```

Expected: `written` equals `files` from the push and `unchanged=0`. If the scratchpad disk lacks space for a second copy, ask the user for another location instead of skipping the check.

- [ ] **Step 4: Compare the clone with the working tree**

```bash
uv --directory /home/andv/personal/thesis/seed-pipeline run python - <<'EOF'
import json
from pathlib import Path
from seed_pipeline.evaluation.artifact_contracts import sha256_file
clone = Path("/tmp/claude-1000/-home-andv-personal-thesis/23fdb23d-5609-4bf6-942e-616429c195ab/scratchpad/thesis-clone/seed-pipeline/data")
original = Path("/home/andv/personal/thesis/seed-pipeline/data")
manifest = json.loads((clone / "work/archive/archive-manifest.json").read_text(encoding="utf-8"))
different = [item["path"] for item in manifest["files"] if sha256_file(original / item["path"]) != item["sha256"] or sha256_file(clone / item["path"]) != item["sha256"]]
print("files", len(manifest["files"]), "different", len(different), different[:5])
EOF
rm -rf /tmp/claude-1000/-home-andv-personal-thesis/23fdb23d-5609-4bf6-942e-616429c195ab/scratchpad/thesis-clone
```

Expected: `different 0`. Then tell the user the dataset reference, version, total size and file count, and remind them of the leftovers outside this plan: the stray Kaggle kernel ending in `3abd35cc1939e942`, the dataset `pipeline-input-3abd35cc1939e942`, and quitting ProxyPal so CLIProxyAPI can use port 8317.
