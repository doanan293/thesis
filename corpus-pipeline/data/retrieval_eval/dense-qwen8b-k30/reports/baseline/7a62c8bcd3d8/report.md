# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 86.33% |
| Hit@5 | 90.83% |
| Hit@10 | 95.14% |
| Hit@30 | 98.17% |
| MRR | 0.7948 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 85.40% |
| Multi-all-hit@3 | 72.60% |
| Multi-section Recall@5 | 93.70% |
| Multi-all-hit@5 | 88.20% |
| Multi-section Recall@10 | 98.50% |
| Multi-all-hit@10 | 97.40% |
| Multi-section Recall@30 | 99.40% |
| Multi-all-hit@30 | 99.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 91.44% | 94.32% | 97.16% | 98.76% | 0.8720 |
| chunk_level_retrieval | 1000 | 61.20% | 76.00% | 88.70% | 98.80% | 0.4996 |
| formulary | 5000 | 89.34% | 93.00% | 96.28% | 98.28% | 0.8144 |
| multi_intent | 500 | 98.20% | 99.20% | 99.60% | 99.80% | 0.9159 |
| noisy_confuser | 500 | 58.60% | 66.80% | 80.20% | 92.20% | 0.5252 |
| patient_natural | 500 | 96.80% | 97.00% | 97.00% | 97.20% | 0.9506 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 93.20% | 96.04% | 98.68% | 99.70% | 0.8841 |
| hard | 568 | 97.18% | 98.06% | 98.59% | 99.47% | 0.9106 |
| medium | 8447 | 84.80% | 89.74% | 94.50% | 97.90% | 0.7765 |
