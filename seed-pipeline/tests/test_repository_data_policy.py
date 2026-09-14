import subprocess
from pathlib import Path

import pytest

from seed_pipeline.config.paths import PROJECT_ROOT


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode == 0


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


def test_no_tracked_data_file_exceeds_five_mib():
    output = subprocess.run(
        ["git", "ls-files", "-z", "--", "data"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tracked = [Path(item.decode()) for item in output.split(b"\0") if item]
    oversized = [
        path
        for path in tracked
        if (PROJECT_ROOT / path).stat().st_size > 5 * 1024 * 1024
    ]
    assert oversized == []


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
