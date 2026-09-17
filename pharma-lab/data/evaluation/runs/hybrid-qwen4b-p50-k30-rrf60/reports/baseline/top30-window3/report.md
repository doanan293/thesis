# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 76.83% |
| Hit@5 | 85.06% |
| Hit@10 | 93.74% |
| Hit@30 | 98.89% |
| nDCG@10 | 0.7470 |
| MRR@10 | 0.6941 |
| MRR@30 | 0.6976 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 54.50% |
| Complete-evidence rate@3 | 29.60% |
| Recall@5 | 69.20% |
| Complete-evidence rate@5 | 48.20% |
| Recall@10 | 86.60% |
| Complete-evidence rate@10 | 75.60% |
| Recall@30 | 99.80% |
| Complete-evidence rate@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 65.40% | 80.20% | 92.40% | 98.50% | 0.6205 | 0.5235 | 0.5275 |
| formulary | 5000 | 71.76% | 81.66% | 92.34% | 98.86% | 0.7057 | 0.6370 | 0.6414 |
| brand_product_qa | 2500 | 92.92% | 95.56% | 98.36% | 99.68% | 0.9115 | 0.8882 | 0.8892 |
| multi_intent | 500 | 79.40% | 90.20% | 97.60% | 100.00% | 0.6340 | 0.6729 | 0.6746 |
| noisy_confuser | 500 | 45.00% | 56.60% | 77.20% | 93.80% | 0.4777 | 0.3877 | 0.3989 |
| patient_natural | 500 | 99.20% | 99.60% | 100.00% | 100.00% | 0.9727 | 0.9633 | 0.9633 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 59.29% | 73.81% | 90.96% | 99.59% | 0.6249 | 0.5369 | 0.5429 |
| hard | 568 | 80.63% | 90.14% | 97.18% | 99.47% | 0.6653 | 0.6978 | 0.6994 |
| medium | 8447 | 78.62% | 86.03% | 93.83% | 98.77% | 0.7667 | 0.7122 | 0.7155 |
