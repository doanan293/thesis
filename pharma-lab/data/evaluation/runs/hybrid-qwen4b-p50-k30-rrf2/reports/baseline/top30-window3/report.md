# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 84.28% |
| Hit@5 | 90.65% |
| Hit@10 | 95.89% |
| Hit@30 | 99.17% |
| MRR | 0.7331 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 67.80% |
| Multi-all-hit@3 | 41.00% |
| Multi-section Recall@5 | 81.80% |
| Multi-all-hit@5 | 66.20% |
| Multi-section Recall@10 | 96.00% |
| Multi-all-hit@10 | 92.40% |
| Multi-section Recall@30 | 99.90% |
| Multi-all-hit@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 67.40% | 82.00% | 92.70% | 99.00% | 0.5325 |
| formulary | 5000 | 83.76% | 90.90% | 96.26% | 99.20% | 0.7020 |
| brand_product_qa | 2500 | 93.48% | 96.04% | 98.44% | 99.72% | 0.8893 |
| multi_intent | 500 | 94.60% | 97.40% | 99.60% | 100.00% | 0.7284 |
| noisy_confuser | 500 | 51.80% | 62.40% | 78.00% | 94.80% | 0.4376 |
| patient_natural | 500 | 99.40% | 100.00% | 100.00% | 100.00% | 0.9642 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 89.24% | 93.91% | 97.16% | 99.90% | 0.6860 |
| hard | 568 | 94.01% | 96.48% | 98.77% | 99.65% | 0.7475 |
| medium | 8447 | 83.05% | 89.88% | 95.55% | 99.05% | 0.7376 |
