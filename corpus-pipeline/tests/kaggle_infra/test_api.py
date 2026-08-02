from types import SimpleNamespace

import pytest

from corpus_pipeline.integrations.kaggle.api import KaggleCommandRunner
from corpus_pipeline.integrations.kaggle.errors import KaggleCommandError


def test_runner_treats_error_text_as_failure(monkeypatch):
    monkeypatch.setattr(
        "corpus_pipeline.integrations.kaggle.api.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="error: rejected", stderr=""
        ),
    )
    with pytest.raises(KaggleCommandError):
        KaggleCommandRunner().run_result(["kaggle", "quota"])
