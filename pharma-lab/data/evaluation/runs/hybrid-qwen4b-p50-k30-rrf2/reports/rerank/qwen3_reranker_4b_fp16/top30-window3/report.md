# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 90.64% |
| Hit@5 | 94.41% |
| Hit@10 | 97.52% |
| Hit@30 | 99.17% |
| nDCG@10 | 0.8586 |
| MRR@10 | 0.8268 |
| MRR@30 | 0.8280 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 73.20% |
| Complete-evidence rate@3 | 51.80% |
| Recall@5 | 86.30% |
| Complete-evidence rate@5 | 73.80% |
| Recall@10 | 96.20% |
| Complete-evidence rate@10 | 92.60% |
| Recall@30 | 99.90% |
| Complete-evidence rate@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 87.30% | 92.10% | 96.40% | 99.00% | 0.8328 | 0.7900 | 0.7917 |
| formulary | 5000 | 90.10% | 94.68% | 98.08% | 99.20% | 0.8416 | 0.7955 | 0.7964 |
| brand_product_qa | 2500 | 97.44% | 98.28% | 99.08% | 99.72% | 0.9584 | 0.9476 | 0.9481 |
| multi_intent | 500 | 94.60% | 98.80% | 99.80% | 100.00% | 0.7615 | 0.8145 | 0.8146 |
| noisy_confuser | 500 | 55.40% | 67.00% | 81.60% | 94.80% | 0.5482 | 0.4642 | 0.4734 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9889 | 0.9850 | 0.9850 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 90.25% | 95.33% | 98.58% | 99.90% | 0.8478 | 0.8021 | 0.8031 |
| hard | 568 | 94.19% | 97.89% | 99.47% | 99.65% | 0.7810 | 0.8261 | 0.8263 |
| medium | 8447 | 90.45% | 94.07% | 97.27% | 99.05% | 0.8651 | 0.8298 | 0.8310 |
