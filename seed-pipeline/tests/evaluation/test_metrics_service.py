from collections import defaultdict

from seed_pipeline.evaluation.metrics_service import (
    accumulate_breakdown_metrics,
    markdown_breakdown_tables,
)


def test_accumulate_breakdown_metrics_groups_each_dimension_independently():
    breakdowns = {
        "eval_group": defaultdict(lambda: defaultdict(float)),
        "difficulty": defaultdict(lambda: defaultdict(float)),
    }
    hits = {
        "hit@3": 1,
        "hit@5": 1,
        "hit@10": 1,
        "hit@30": 1,
        "mrr": 0.5,
        "multi_section_recall@3": 0.5,
    }

    accumulate_breakdown_metrics(
        breakdowns,
        hits,
        {
            "eval_group": "patient_natural",
            "difficulty": "easy",
            "answer_mode": "multi_required",
        },
    )
    accumulate_breakdown_metrics(
        breakdowns,
        {**hits, "hit@3": 0, "mrr": 0.25},
        {
            "eval_group": "patient_natural",
            "difficulty": "hard",
            "answer_mode": "single",
        },
    )
    accumulate_breakdown_metrics(
        breakdowns,
        hits,
        {"eval_group": "", "difficulty": None, "answer_mode": "single"},
    )

    patient = breakdowns["eval_group"]["patient_natural"]
    assert patient["count"] == 2
    assert patient["hit@3"] == 1
    assert patient["mrr"] == 0.75
    assert breakdowns["difficulty"]["easy"]["count"] == 1
    assert breakdowns["difficulty"]["hard"]["count"] == 1
    assert breakdowns["eval_group"]["unknown"]["count"] == 1
    assert breakdowns["difficulty"]["unknown"]["count"] == 1


def test_markdown_breakdown_tables_renders_only_compact_metrics():
    breakdowns = {
        "eval_group": {
            "chunk_risk": {
                "count": 2,
                "hit@3": 1,
                "hit@5": 2,
                "hit@10": 2,
                "hit@30": 2,
                "mrr": 0.75,
                "multi_count": 1,
            },
            "ankhang": {
                "count": 1,
                "hit@3": 1,
                "hit@5": 1,
                "hit@10": 1,
                "hit@30": 1,
                "mrr": 1.0,
            },
        },
        "difficulty": {
            "easy": {
                "count": 1,
                "hit@3": 0,
                "hit@5": 1,
                "hit@10": 1,
                "hit@30": 1,
                "mrr": 0.2,
            },
        },
    }

    rendered = markdown_breakdown_tables(breakdowns)

    assert "## Breakdown by eval_group" in rendered
    assert "| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |" in rendered
    assert (
        "| brand_product_qa | 1 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |"
        in rendered
    )
    assert (
        "| chunk_level_retrieval | 2 | 50.00% | 100.00% | 100.00% | 100.00% | 0.3750 |"
        in rendered
    )
    assert "## Breakdown by difficulty" in rendered
    assert "| ankhang |" not in rendered
    assert "| chunk_risk |" not in rendered
    assert "Multi-section" not in rendered
    assert "Multi-all-hit" not in rendered
