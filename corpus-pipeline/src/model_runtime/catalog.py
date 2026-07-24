from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelKind(StrEnum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"


class ModelTopology(StrEnum):
    REPLICATED_2X1 = "replicated_2x1"
    SHARDED_1X2 = "sharded_1x2"


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
    local_parallel: int = 1
    local_request_batch_size: int = 1
    kaggle_context_per_slot: int = 2048
    kaggle_logical_batch_size: int = 2048
    kaggle_physical_batch_size: int = 2048
    vector_dimension: int | None = None
    reranker_protocol: str | None = None

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
    )


def _reranker(
    name: str,
    filename: str,
    size: int,
    sha256: str,
    topology: ModelTopology,
    protocol: str,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        kind=ModelKind.RERANKER,
        canonical_filename=filename,
        byte_size=size,
        sha256=sha256,
        topology=topology,
        kaggle_parallel=1,
        kaggle_request_batch_size=1,
        reranker_protocol=protocol,
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
    "qwen3-reranker:0.6b-fp16": _reranker(
        "qwen3-reranker:0.6b-fp16",
        "qwen3-reranker-0.6b-f16.gguf",
        1_197_634_304,
        "fa726a72c1afafe42ae6ca6059c9a78a43f18db7389a8fa04f88bb7f37d0a8aa",
        ModelTopology.REPLICATED_2X1,
        "completion_logprobs",
    ),
    "qwen3-reranker:4b-fp16": _reranker(
        "qwen3-reranker:4b-fp16",
        "qwen3-reranker-4b-f16.gguf",
        8_049_912_256,
        "029cfa267eff0a1c4a19371d8db857b0e515eaa1414c32fc3d4a885df3b142c9",
        ModelTopology.REPLICATED_2X1,
        "completion_logprobs",
    ),
    "qwen3-reranker:8b-fp16": _reranker(
        "qwen3-reranker:8b-fp16",
        "qwen3-reranker-8b-f16.gguf",
        15_141_207_744,
        "a53322f7936010458424a12f0f6d22291547e42fa85c16dd4730244d659cea96",
        ModelTopology.SHARDED_1X2,
        "completion_logprobs",
    ),
    "bge-reranker-v2-m3:f16": _reranker(
        "bge-reranker-v2-m3:f16",
        "bge-reranker-v2-m3-f16.gguf",
        1_159_774_912,
        "3c2de408d2c0a85a9472dc09f9d5a22c9b73743c6343952c15053299c777c298",
        ModelTopology.REPLICATED_2X1,
        "native_rerank",
    ),
    "bge-reranker-v2-gemma:f16": _reranker(
        "bge-reranker-v2-gemma:f16",
        "bge-reranker-v2-gemma-f16.gguf",
        5_018_535_968,
        "640562faf67b49777bd06869a9a915906067a330ffe382825844f226bbf1939c",
        ModelTopology.REPLICATED_2X1,
        "native_rerank",
    ),
}

MODEL_CATALOG = {**EMBEDDING_MODELS, **RERANKER_MODELS}


def require_model(name: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc
