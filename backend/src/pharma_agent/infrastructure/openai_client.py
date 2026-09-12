"""One place that builds AsyncOpenAI clients for chat and embeddings."""

from openai import AsyncOpenAI


def build_async_openai(
    *,
    api_key: str,
    base_url: str | None,
    timeout: float,
    max_retries: int,
    traced: bool,
) -> AsyncOpenAI:
    """Build a client; `traced` routes calls through Langfuse's OpenAI integration.

    Importing `langfuse.openai` instruments the OpenAI SDK process-wide, so doing it here,
    when the application is built, makes chat and embedding calls traced from the first
    request instead of only after some other code happened to import it.
    """
    if traced:
        from langfuse.openai import AsyncOpenAI as TracedAsyncOpenAI

        return TracedAsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
    return AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
    )
