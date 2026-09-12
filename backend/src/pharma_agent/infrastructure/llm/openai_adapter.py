"""LlmPort over the OpenAI SDK using Chat Completions, so any OpenAI-compatible server works."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar

from openai import OpenAIError
from pydantic import BaseModel

from pharma_agent.domain.llm.models import ChatMessage, LlmRole, LlmUsage, StreamDelta
from pharma_agent.domain.llm.port import LlmError
from pharma_agent.infrastructure.openai_client import build_async_openai
from pharma_agent.infrastructure.settings import LlmSettings, ResolvedEndpoint

T = TypeVar("T", bound=BaseModel)
ClientFactory = Callable[[ResolvedEndpoint, float, int], Any]


def default_client_factory(
    endpoint: ResolvedEndpoint, timeout: float, max_retries: int
) -> Any:
    return build_async_openai(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        timeout=timeout,
        max_retries=max_retries,
        traced=False,
    )


def langfuse_client_factory(
    endpoint: ResolvedEndpoint, timeout: float, max_retries: int
) -> Any:
    """Every call becomes a Langfuse generation with tokens and cost."""
    return build_async_openai(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        timeout=timeout,
        max_retries=max_retries,
        traced=True,
    )


class OpenAiLlmAdapter:
    def __init__(
        self,
        settings: LlmSettings,
        client_factory: ClientFactory = default_client_factory,
    ) -> None:
        self._settings = settings
        self._factory = client_factory
        self._endpoints: dict[LlmRole, ResolvedEndpoint] = {
            role: settings.resolve(role) for role in LlmRole
        }
        self._clients: dict[tuple[str | None, str], Any] = {}

    def _client(self, role: LlmRole) -> tuple[Any, ResolvedEndpoint]:
        endpoint = self._endpoints[role]
        key = (endpoint.base_url, endpoint.api_key)
        client = self._clients.get(key)
        if client is None:
            client = self._factory(
                endpoint, self._settings.timeout_seconds, self._settings.max_retries
            )
            self._clients[key] = client
        return client, endpoint

    async def structured(
        self, role: LlmRole, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, LlmUsage]:
        client, endpoint = self._client(role)
        try:
            completion = await client.chat.completions.parse(
                model=endpoint.model,
                messages=_to_openai(messages),
                response_format=schema,
                **_options(endpoint),
            )
        except OpenAIError as exc:
            raise LlmError(f"{role.value}: {exc}") from exc
        choice = completion.choices[0]
        parsed = choice.message.parsed
        if parsed is None:
            raise LlmError(
                f"{role.value}: model returned no parsed output (refusal={choice.message.refusal!r})"
            )
        return parsed, _usage(completion.usage)

    async def stream(
        self, role: LlmRole, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamDelta]:
        client, endpoint = self._client(role)
        try:
            stream = await client.chat.completions.create(
                model=endpoint.model,
                messages=_to_openai(messages),
                stream=True,
                stream_options={"include_usage": True},
                **_options(endpoint),
            )
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None and delta.content:
                        yield StreamDelta(text=delta.content)
                if getattr(chunk, "usage", None) is not None:
                    yield StreamDelta(usage=_usage(chunk.usage))
        except OpenAIError as exc:
            raise LlmError(f"{role.value}: {exc}") from exc


def _options(endpoint: ResolvedEndpoint) -> dict[str, Any]:
    """Optional request fields, sent only when set so any OpenAI-compatible server accepts the call."""
    if endpoint.reasoning_effort is None:
        return {}
    return {"reasoning_effort": endpoint.reasoning_effort}


def _to_openai(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


def _usage(usage: Any) -> LlmUsage:
    if usage is None:
        return LlmUsage()
    return LlmUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )
