# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 85.47% |
| Hit@5 | 90.36% |
| Hit@10 | 95.39% |
| Hit@30 | 98.59% |
| MRR | 0.7813 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 83.20% |
| Multi-all-hit@3 | 69.60% |
| Multi-section Recall@5 | 92.90% |
| Multi-all-hit@5 | 87.20% |
| Multi-section Recall@10 | 99.20% |
| Multi-all-hit@10 | 98.40% |
| Multi-section Recall@30 | 100.00% |
| Multi-all-hit@30 | 100.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 58.30% | 71.80% | 87.20% | 98.30% | 0.4763 |
| formulary | 5000 | 88.74% | 93.06% | 97.02% | 99.06% | 0.7967 |
| brand_product_qa | 2500 | 90.44% | 93.36% | 96.68% | 98.64% | 0.8671 |
| multi_intent | 500 | 96.80% | 98.60% | 100.00% | 100.00% | 0.9059 |
| noisy_confuser | 500 | 59.60% | 70.80% | 82.60% | 94.00% | 0.5275 |
| patient_natural | 500 | 96.80% | 96.80% | 97.20% | 97.40% | 0.9377 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 94.42% | 96.24% | 98.88% | 100.00% | 0.8923 |
| hard | 568 | 95.95% | 97.54% | 99.12% | 99.82% | 0.9022 |
| medium | 8447 | 83.72% | 89.19% | 94.73% | 98.34% | 0.7603 |
