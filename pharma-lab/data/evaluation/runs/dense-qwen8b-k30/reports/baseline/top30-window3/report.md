# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 86.87% |
| Hit@5 | 91.31% |
| Hit@10 | 95.36% |
| Hit@30 | 98.18% |
| nDCG@10 | 0.8341 |
| MRR@10 | 0.7978 |
| MRR@30 | 0.7997 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 87.80% |
| Complete-evidence rate@3 | 77.00% |
| Recall@5 | 95.20% |
| Complete-evidence rate@5 | 91.00% |
| Recall@10 | 99.00% |
| Complete-evidence rate@10 | 98.00% |
| Recall@30 | 99.70% |
| Complete-evidence rate@30 | 99.40% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 61.20% | 75.90% | 88.60% | 98.70% | 0.5876 | 0.4930 | 0.4994 |
| formulary | 5000 | 90.14% | 93.76% | 96.56% | 98.20% | 0.8562 | 0.8202 | 0.8213 |
| brand_product_qa | 2500 | 91.80% | 94.56% | 97.44% | 99.00% | 0.8983 | 0.8736 | 0.8747 |
| multi_intent | 500 | 98.60% | 99.40% | 100.00% | 100.00% | 0.9072 | 0.9261 | 0.9261 |
| noisy_confuser | 500 | 58.60% | 67.20% | 79.80% | 91.60% | 0.5844 | 0.5179 | 0.5259 |
| patient_natural | 500 | 97.40% | 97.40% | 97.40% | 97.60% | 0.9615 | 0.9570 | 0.9571 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 93.81% | 96.75% | 99.29% | 100.00% | 0.9144 | 0.8887 | 0.8893 |
| hard | 568 | 97.71% | 98.42% | 99.12% | 99.82% | 0.9054 | 0.9207 | 0.9213 |
| medium | 8447 | 85.33% | 90.20% | 94.65% | 97.86% | 0.8199 | 0.7790 | 0.7811 |
