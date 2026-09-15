import io
from datetime import UTC, datetime

from seed_pipeline.config import paths
from seed_pipeline.evaluation.rerank_log import RerankLog, open_rerank_log


def _noon() -> datetime:
    return datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def test_log_appends_timestamped_lines_and_echoes_them(tmp_path) -> None:
    path = tmp_path / "logs" / "rerank" / "model.log"
    echo = io.StringIO()
    log = RerankLog(path, echo=echo, now=_noon)

    log("account=acc3 budget_seconds=21600")
    log("quota table\nacc1 27.58h")

    expected = (
        "2026-09-15T12:00:00+00:00 account=acc3 budget_seconds=21600\n"
        "2026-09-15T12:00:00+00:00 quota table\n"
        "2026-09-15T12:00:00+00:00 acc1 27.58h\n"
    )
    assert path.read_text(encoding="utf-8") == expected
    assert echo.getvalue() == expected


def test_log_keeps_lines_written_by_earlier_commands(tmp_path) -> None:
    path = tmp_path / "model.log"

    RerankLog(path, now=_noon)("first")
    RerankLog(path, now=_noon)("second")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == ["first", "second"]


def test_open_rerank_log_targets_the_model_log(isolated_logs_dir) -> None:
    log = open_rerank_log("qwen3-reranker:0.6b-fp16", echo=False)

    assert log.path == (
        isolated_logs_dir.root / "rerank" / "qwen3_reranker_0_6b_fp16.log"
    )
    assert log.path.parent.parent == paths.LOGS_DIR
