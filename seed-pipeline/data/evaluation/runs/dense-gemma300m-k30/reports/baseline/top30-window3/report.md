# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 67.36% |
| Hit@5 | 75.41% |
| Hit@10 | 83.80% |
| Hit@30 | 90.61% |
| MRR | 0.5902 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 55.40% |
| Multi-all-hit@3 | 24.80% |
| Multi-section Recall@5 | 64.00% |
| Multi-all-hit@5 | 36.80% |
| Multi-section Recall@10 | 75.70% |
| Multi-all-hit@10 | 54.80% |
| Multi-section Recall@30 | 85.10% |
| Multi-all-hit@30 | 71.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 63.00% | 76.40% | 89.90% | 97.90% | 0.5014 |
| formulary | 5000 | 61.02% | 70.64% | 80.72% | 89.38% | 0.5226 |
| brand_product_qa | 2500 | 82.60% | 86.48% | 90.40% | 93.28% | 0.7655 |
| multi_intent | 500 | 86.00% | 91.20% | 96.60% | 98.40% | 0.7280 |
| noisy_confuser | 500 | 40.00% | 50.80% | 62.20% | 76.40% | 0.3477 |
| patient_natural | 500 | 72.00% | 74.60% | 78.20% | 81.40% | 0.6715 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 78.38% | 82.54% | 87.41% | 92.39% | 0.6925 |
| hard | 568 | 84.86% | 89.96% | 94.89% | 96.65% | 0.7223 |
| medium | 8447 | 64.90% | 73.60% | 82.63% | 90.00% | 0.5694 |
