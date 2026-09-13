# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 66.91% |
| Hit@5 | 75.04% |
| Hit@10 | 83.12% |
| Hit@30 | 89.74% |
| MRR | 0.5879 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 55.20% |
| Multi-all-hit@3 | 24.40% |
| Multi-section Recall@5 | 64.60% |
| Multi-all-hit@5 | 38.20% |
| Multi-section Recall@10 | 75.20% |
| Multi-all-hit@10 | 54.60% |
| Multi-section Recall@30 | 84.20% |
| Multi-all-hit@30 | 70.40% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 82.00% | 85.68% | 89.52% | 92.60% | 0.7623 |
| chunk_level_retrieval | 1000 | 63.10% | 76.20% | 89.80% | 98.00% | 0.5023 |
| formulary | 5000 | 60.66% | 70.66% | 80.38% | 88.60% | 0.5202 |
| multi_intent | 500 | 86.00% | 91.00% | 95.80% | 98.00% | 0.7375 |
| noisy_confuser | 500 | 40.60% | 49.80% | 61.00% | 74.60% | 0.3483 |
| patient_natural | 500 | 68.80% | 72.60% | 74.60% | 77.20% | 0.6532 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 78.38% | 83.05% | 88.12% | 92.89% | 0.6939 |
| hard | 568 | 84.33% | 89.44% | 94.19% | 96.30% | 0.7285 |
| medium | 8447 | 64.40% | 73.14% | 81.79% | 88.93% | 0.5660 |
