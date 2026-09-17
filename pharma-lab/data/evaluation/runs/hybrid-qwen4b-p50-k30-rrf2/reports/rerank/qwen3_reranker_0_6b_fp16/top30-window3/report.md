# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 87.36% |
| Hit@5 | 92.04% |
| Hit@10 | 96.75% |
| Hit@30 | 99.17% |
| nDCG@10 | 0.8283 |
| MRR@10 | 0.7888 |
| MRR@30 | 0.7905 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 74.90% |
| Complete-evidence rate@3 | 51.60% |
| Recall@5 | 89.80% |
| Complete-evidence rate@5 | 79.80% |
| Recall@10 | 98.20% |
| Complete-evidence rate@10 | 96.40% |
| Recall@30 | 99.90% |
| Complete-evidence rate@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 77.70% | 87.00% | 94.60% | 99.00% | 0.7184 | 0.6449 | 0.6478 |
| formulary | 5000 | 86.76% | 92.06% | 97.22% | 99.20% | 0.8217 | 0.7729 | 0.7743 |
| brand_product_qa | 2500 | 95.48% | 97.68% | 99.40% | 99.72% | 0.9186 | 0.8935 | 0.8938 |
| multi_intent | 500 | 98.20% | 99.80% | 100.00% | 100.00% | 0.8187 | 0.8726 | 0.8726 |
| noisy_confuser | 500 | 48.60% | 58.00% | 76.60% | 94.80% | 0.5137 | 0.4366 | 0.4484 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9862 | 0.9813 | 0.9813 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 86.19% | 91.47% | 96.65% | 99.90% | 0.8265 | 0.7813 | 0.7835 |
| hard | 568 | 97.36% | 98.77% | 99.47% | 99.65% | 0.8294 | 0.8750 | 0.8752 |
| medium | 8447 | 86.82% | 91.65% | 96.58% | 99.05% | 0.8284 | 0.7839 | 0.7856 |
