from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pharma_lab.runtime.model_profiles import (
    NATIVE_RERANK_PROTOCOL,
    EmbeddingRuntimeProfile,
    EmbeddingWorkloadProfile,
    RerankContract,
    native_rerank_contract,
)
from pharma_lab.runtime.runtime_profiles import (
    EmbeddingRuntimeSearchSpaces,
    RuntimeCandidate,
    RuntimeSearchSpace,
    reranker_candidate,
)


class ModelKind(StrEnum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    CHAT = "chat"


class ModelTopology(StrEnum):
    REPLICATED_2X1 = "replicated_2x1"
    SHARDED_1X2 = "sharded_1x2"


KAGGLE_RERANK_REQUEST_BATCH_SIZE = 30
LOCAL_RERANK_REQUEST_BATCH_SIZE = 15
# Reranker levels as (-np, -ub). Every slot may hold a whole ubatch, so -c is
# -ub x (slots + 1) and -ub alone bounds the longest prompt the server accepts.
# On a T4 (14,806 MiB free) compute takes about 0.6 MiB per ubatch token and KV
# 0.11 MiB (0.6b) or 0.14 MiB (4b, 8b) per -c token, so the larger models take fewer
# slots.
# A batch carries at most -np whole documents, so a level with too few slots leaves the
# ubatch (and the GPU) half empty; the levels below sweep slots and requests in flight so
# the benchmark can measure which one actually wins.
SMALL_RERANK_LEVELS = ((2, 4096, 2), (4, 4096, 2), (4, 4096, 4))
SMALL_MODEL_KAGGLE_RERANK_LEVELS = (
    (4, 4096, 2),
    (8, 4096, 2),
    (8, 4096, 4),
    (16, 4096, 4),
)
# The 8b model is sharded over both T4s, so its KV costs half as much per GPU.
SHARDED_RERANK_LEVELS = ((2, 4096, 2), (4, 4096, 4), (8, 4096, 4))
# The local CPU machine serves one request of at most 15 candidates at a time.
LOCAL_RERANK_LEVELS = ((4, 4096, 1),)
# Local CPU llama-reranker levels. The backend sends one request with at most 15
# candidates at a time, so a level has a single client request in flight.
LOCAL_RERANK_SEARCH_SPACE = RuntimeSearchSpace(
    tuple(
        reranker_candidate(
            server_slots=server_slots,
            ubatch=ubatch,
            request_batch_size=LOCAL_RERANK_REQUEST_BATCH_SIZE,
            concurrency=concurrency,
            threads=threads,
        )
        for server_slots, ubatch, concurrency in LOCAL_RERANK_LEVELS
        for threads in (8, 12)
    )
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: ModelKind
    canonical_filename: str
    byte_size: int
    sha256: str
    topology: ModelTopology
    kaggle_parallel: int
    kaggle_request_batch_size: int
    local_request_batch_size: int = 1
    kaggle_context_per_slot: int = 2048
    kaggle_logical_batch_size: int = 2048
    kaggle_physical_batch_size: int = 2048
    vector_dimension: int | None = None
    reranker_protocol: str | None = None
    rerank_contract: RerankContract | None = None
    embedding_runtime: EmbeddingRuntimeProfile | None = None
    rerank_search_space: RuntimeSearchSpace | None = None
    local_rerank_search_space: RuntimeSearchSpace | None = None
    embedding_search_space: EmbeddingRuntimeSearchSpaces | None = None
    # Chat models are only served, so their server layout is fixed instead of benchmarked.
    serve_runtime: RuntimeCandidate | None = None

    @property
    def slug(self) -> str:
        return (
            self.name.replace(":", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "_")
        )

    @property
    def gguf_dataset_slug(self) -> str:
        return "vector-cache-gguf-" + self.canonical_filename.removesuffix(
            ".gguf"
        ).replace(".", "-")


def _embedding(
    name: str,
    filename: str,
    size: int,
    sha256: str,
    dimension: int,
    topology: ModelTopology,
    parallel: int,
    batch: int,
) -> ModelSpec:
    benchmark_concurrency = (
        (1, 2) if topology is ModelTopology.SHARDED_1X2 else (1, 2, 4)
    )
    embedding_profile = EmbeddingRuntimeProfile(
        query=EmbeddingWorkloadProfile(
            production_batch_size=batch,
            production_concurrency=1,
            benchmark_batch_sizes=tuple(sorted({max(1, batch // 2), batch, batch * 2})),
            benchmark_concurrency=benchmark_concurrency,
        ),
        corpus=EmbeddingWorkloadProfile(
            production_batch_size=batch,
            production_concurrency=1,
            benchmark_batch_sizes=tuple(sorted({max(1, batch // 2), batch, batch * 2})),
            benchmark_concurrency=benchmark_concurrency,
        ),
        context_per_slot=2048,
        logical_batch_size=2048,
        physical_batch_size=2048,
    )

    def candidates(batch_sizes: tuple[int, ...]) -> RuntimeSearchSpace:
        return RuntimeSearchSpace(
            tuple(
                RuntimeCandidate(
                    server_slots=parallel,
                    concurrency=concurrency,
                    request_batch_size=batch_size,
                    context_per_slot=2048,
                    logical_batch_size=2048,
                    physical_batch_size=2048,
                )
                for batch_size in batch_sizes
                for concurrency in benchmark_concurrency
            )
        )

    return ModelSpec(
        name=name,
        kind=ModelKind.EMBEDDING,
        canonical_filename=filename,
        byte_size=size,
        sha256=sha256,
        topology=topology,
        kaggle_parallel=parallel,
        kaggle_request_batch_size=batch,
        vector_dimension=dimension,
        embedding_runtime=embedding_profile,
        embedding_search_space=EmbeddingRuntimeSearchSpaces(
            query=candidates(tuple(sorted({max(1, batch // 2), batch}))),
            corpus=candidates(tuple(sorted({batch, batch * 2}))),
        ),
    )


def _reranker(
    name: str,
    filename: str,
    size: int,
    sha256: str,
    topology: ModelTopology,
    *,
    levels: tuple[tuple[int, int, int], ...],
) -> ModelSpec:
    search_space = RuntimeSearchSpace(
        tuple(
            reranker_candidate(
                server_slots=server_slots,
                ubatch=ubatch,
                request_batch_size=KAGGLE_RERANK_REQUEST_BATCH_SIZE,
                concurrency=concurrency,
            )
            for server_slots, ubatch, concurrency in levels
        )
    )
    smallest = search_space.candidates[0]
    return ModelSpec(
        name=name,
        kind=ModelKind.RERANKER,
        canonical_filename=filename,
        byte_size=size,
        sha256=sha256,
        topology=topology,
        kaggle_parallel=smallest.server_slots,
        kaggle_request_batch_size=KAGGLE_RERANK_REQUEST_BATCH_SIZE,
        kaggle_context_per_slot=smallest.context_per_slot,
        kaggle_logical_batch_size=smallest.logical_batch_size,
        kaggle_physical_batch_size=smallest.physical_batch_size,
        reranker_protocol=NATIVE_RERANK_PROTOCOL,
        rerank_contract=native_rerank_contract(),
        rerank_search_space=search_space,
        local_rerank_search_space=LOCAL_RERANK_SEARCH_SPACE,
    )


EMBEDDING_MODELS = {
    "embeddinggemma:300m": _embedding(
        "embeddinggemma:300m",
        "embeddinggemma-300m-bf16.gguf",
        612_429_792,
        "95a1f284251f78a1409a9c9e52dd4026c2180b13a90a5ede2a878bb8141fba08",
        768,
        ModelTopology.REPLICATED_2X1,
        4,
        32,
    ),
    "bge-m3:567m-fp16": _embedding(
        "bge-m3:567m-fp16",
        "bge-m3-567m-fp16.gguf",
        1_157_671_200,
        "daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c",
        1024,
        ModelTopology.REPLICATED_2X1,
        2,
        16,
    ),
    "qwen3-embedding:0.6b-fp16": _embedding(
        "qwen3-embedding:0.6b-fp16",
        "qwen3-embedding-0.6b-fp16.gguf",
        1_197_629_632,
        "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340",
        1024,
        ModelTopology.REPLICATED_2X1,
        1,
        8,
    ),
    "qwen3-embedding:4b-fp16": _embedding(
        "qwen3-embedding:4b-fp16",
        "qwen3-embedding-4b-fp16.gguf",
        8_049_889_824,
        "e8b4e85c8fcc26079d27418cf8d6a16df1a09890cba0966324a97280f91e782c",
        2560,
        ModelTopology.REPLICATED_2X1,
        1,
        8,
    ),
    "qwen3-embedding:8b-fp16": _embedding(
        "qwen3-embedding:8b-fp16",
        "qwen3-embedding-8b-fp16.gguf",
        15_141_156_384,
        "9a2dfcc2e867828909456dd52a69e3775b677bdce1816f7cc55f3657393e7e53",
        4096,
        ModelTopology.SHARDED_1X2,
        1,
        4,
    ),
}


RERANKER_MODELS = {
    # The Qwen3 files carry the Vietnamese medical rerank instruction; the instruction
    # decision and the derivation are in docs/guides/evaluation.md.
    "qwen3-reranker:0.6b-fp16": _reranker(
        "qwen3-reranker:0.6b-fp16",
        "qwen3-reranker-0.6b-f16-vimed.gguf",
        1_197_634_336,
        "fa17b7c742ffeeb6f80529e50c9c35a8179c4369baa4a039ef94381c0665f851",
        ModelTopology.REPLICATED_2X1,
        levels=SMALL_MODEL_KAGGLE_RERANK_LEVELS,
    ),
    "qwen3-reranker:4b-fp16": _reranker(
        "qwen3-reranker:4b-fp16",
        "qwen3-reranker-4b-f16-vimed.gguf",
        8_049_922_912,
        "4b428e981efc9a209c674a0af9f1ec527b5570896e0bcc431382c16f1e839c7a",
        ModelTopology.REPLICATED_2X1,
        levels=SMALL_RERANK_LEVELS,
    ),
    "qwen3-reranker:8b-fp16": _reranker(
        "qwen3-reranker:8b-fp16",
        "qwen3-reranker-8b-f16-vimed.gguf",
        15_141_207_776,
        "c6516333e32d4f1d8aad8325d5d5bdb8165f9778e892eed4ca8b0eaf3339a810",
        ModelTopology.SHARDED_1X2,
        levels=SHARDED_RERANK_LEVELS,
    ),
    # XLM-R cross-encoder (568M, 8,192 positions) in the 0.6b size class; its inputs carry
    # no chat template, so it takes the 0.6b levels.
    "bge-reranker-v2-m3:f16": _reranker(
        "bge-reranker-v2-m3:f16",
        "bge-reranker-v2-m3-f16.gguf",
        1_159_774_912,
        "3c2de408d2c0a85a9472dc09f9d5a22c9b73743c6343952c15053299c777c298",
        ModelTopology.REPLICATED_2X1,
        levels=SMALL_MODEL_KAGGLE_RERANK_LEVELS,
    ),
}

CHAT_MODELS = {
    # Qwen3.5-9B (Apache 2.0) converted from the BF16 GGUF to F16, because T4 GPUs have
    # no bfloat16 kernels; the derivation is in docs/guides/e2e-evaluation.md. At 17.9 GB
    # it is split over both T4s. Only 8 of its 32 layers keep a KV cache (32 KiB per
    # token), so four 16k-token slots fit next to the weights on each GPU.
    "qwen3.5:9b-f16": ModelSpec(
        name="qwen3.5:9b-f16",
        kind=ModelKind.CHAT,
        canonical_filename="qwen3.5-9b-f16.gguf",
        byte_size=17_920_697_312,
        sha256="863a67e28486f4c3ad7a30c49614a398d5a07caf8d0067a018d0e5f0abf79f2d",
        topology=ModelTopology.SHARDED_1X2,
        kaggle_parallel=1,
        kaggle_request_batch_size=1,
        serve_runtime=RuntimeCandidate(
            server_slots=4,
            concurrency=4,
            request_batch_size=1,
            context_per_slot=16384,
            logical_batch_size=2048,
            physical_batch_size=512,
        ),
    ),
}

MODEL_CATALOG = {**EMBEDDING_MODELS, **RERANKER_MODELS, **CHAT_MODELS}


def require_model(name: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc
