from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


QWEN3_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
DEFAULT_RERANK_INSTRUCTION = (
    "Given a Vietnamese medical retrieval query, retrieve relevant passages "
    "that answer the query"
)


def build_qwen3_yes_no_prompt(
    query: str,
    document: str,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
) -> str:
    return (
        f"<|im_start|>system\n{QWEN3_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@dataclass(frozen=True)
class CompletionScoring:
    positive_token: str
    negative_token: str
    n_predict: int
    temperature: float
    samplers: tuple[str, ...]
    n_probs: int
    min_keep: int
    post_sampling_probs: bool
    logit_bias: float

    def __post_init__(self) -> None:
        if not self.positive_token or not self.negative_token:
            raise ValueError("completion candidate tokens must not be empty")
        if self.positive_token == self.negative_token:
            raise ValueError("completion candidate tokens must differ")
        if self.n_predict < 1 or self.n_probs < 1 or self.min_keep < 1:
            raise ValueError("completion scoring counts must be positive")
        if self.temperature < 0:
            raise ValueError("completion temperature must be non-negative")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "positive_token": self.positive_token,
            "negative_token": self.negative_token,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "samplers": list(self.samplers),
            "n_probs": self.n_probs,
            "min_keep": self.min_keep,
            "post_sampling_probs": self.post_sampling_probs,
            "logit_bias": self.logit_bias,
        }


PromptBuilder = Callable[[str, str, str], str]


@dataclass(frozen=True)
class RerankContract:
    protocol: str
    template_id: str | None
    template_version: str | None
    instruction: str
    scoring: CompletionScoring | None = None

    def __post_init__(self) -> None:
        if self.protocol == "completion_logprobs":
            if not self.template_id or not self.template_version:
                raise ValueError(
                    "completion_logprobs requires a prompt template and version"
                )
            if self.scoring is None:
                raise ValueError("completion_logprobs requires completion scoring")
        elif self.protocol == "native_rerank":
            if self.template_id or self.template_version or self.instruction:
                raise ValueError("native_rerank must not declare a prompt")
            if self.scoring is not None:
                raise ValueError("native_rerank must not declare completion scoring")
        else:
            raise ValueError(f"unsupported rerank protocol: {self.protocol}")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "instruction": self.instruction,
            "scoring": (
                self.scoring.canonical_payload() if self.scoring is not None else None
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    def build_prompt(self, query: str, document: str) -> str:
        if self.template_id != "qwen3_yes_no_v1":
            raise ValueError(f"unsupported prompt template: {self.template_id}")
        return build_qwen3_yes_no_prompt(query, document, self.instruction)


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
        if min(self.context_per_slot, self.logical_batch_size, self.physical_batch_size) < 1:
            raise ValueError("embedding runtime values must be positive")


def qwen3_rerank_contract() -> RerankContract:
    return RerankContract(
        protocol="completion_logprobs",
        template_id="qwen3_yes_no_v1",
        template_version="1",
        instruction=DEFAULT_RERANK_INSTRUCTION,
        scoring=CompletionScoring(
            positive_token="yes",
            negative_token="no",
            n_predict=1,
            temperature=1.0,
            samplers=("temperature",),
            n_probs=2,
            min_keep=2,
            post_sampling_probs=True,
            logit_bias=100.0,
        ),
    )


def native_rerank_contract() -> RerankContract:
    return RerankContract(
        protocol="native_rerank",
        template_id=None,
        template_version=None,
        instruction="",
    )
