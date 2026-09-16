"""llama-reranker from compose.yaml scores a long document with the batched flags."""

import os
import shutil
import subprocess
from collections.abc import Iterator

import pytest
import requests

from pharma_lab.config.paths import COMPOSE_FILE, GGUF_ROOT
from pharma_lab.runtime.catalog import require_model
from pharma_lab.runtime.client import LlamaCppClient, LlamaCppRequestError
from pharma_lab.runtime.compose import LlamaCppComposeManager, SubprocessRunner
from pharma_lab.runtime.runtime_profiles import reranker_candidate

pytestmark = pytest.mark.integration

MODEL = "qwen3-reranker:0.6b-fp16"
PROJECT = "seed-rerank-integration"
PORT = "18435"
# The smallest ubatch the catalog allows: the longest prompt of the corpus is 2,074
# tokens and must still fit in one pass.
LEVEL = reranker_candidate(
    server_slots=4, ubatch=4096, request_batch_size=3, concurrency=1, threads=4
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
