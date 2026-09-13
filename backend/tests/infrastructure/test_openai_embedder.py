from pharma_agent.infrastructure.retrieval.qdrant_adapter import OpenAiEmbedder


def test_openai_embedder_exposes_model_and_dimension() -> None:
    embedder = OpenAiEmbedder(object(), model="qwen3-embedding:4b-fp16", dimension=2560)
    assert (embedder.model, embedder.dimension) == ("qwen3-embedding:4b-fp16", 2560)
