from importlib.metadata import version


def test_ragas_imports_cleanly_with_the_workspace_openai() -> None:
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        FactualCorrectness,
        Faithfulness,
    )

    assert version("ragas") == "0.4.3"
    assert version("openai").startswith("3.")
    assert all(
        callable(item)
        for item in (
            OpenAIEmbeddings,
            llm_factory,
            AnswerRelevancy,
            FactualCorrectness,
            Faithfulness,
        )
    )
