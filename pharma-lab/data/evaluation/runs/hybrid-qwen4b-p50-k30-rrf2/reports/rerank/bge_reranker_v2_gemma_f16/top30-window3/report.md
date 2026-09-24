# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 85.23% |
| Hit@5 | 90.73% |
| Hit@10 | 95.65% |
| Hit@30 | 99.06% |
| nDCG@10 | 0.8031 |
| MRR@10 | 0.7593 |
| MRR@30 | 0.7616 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 67.10% |
| Complete-evidence rate@3 | 41.20% |
| Recall@5 | 81.10% |
| Complete-evidence rate@5 | 64.20% |
| Recall@10 | 93.10% |
| Complete-evidence rate@10 | 86.20% |
| Recall@30 | 99.90% |
| Complete-evidence rate@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 83.20% | 90.20% | 95.00% | 99.10% | 0.7694 | 0.7102 | 0.7129 |
| formulary | 5000 | 81.46% | 88.32% | 95.00% | 98.98% | 0.7603 | 0.6990 | 0.7017 |
| brand_product_qa | 2500 | 95.72% | 98.00% | 99.28% | 99.72% | 0.9323 | 0.9124 | 0.9127 |
| multi_intent | 500 | 93.00% | 98.00% | 100.00% | 100.00% | 0.7580 | 0.7979 | 0.7979 |
| noisy_confuser | 500 | 52.00% | 63.00% | 76.60% | 94.60% | 0.5053 | 0.4231 | 0.4344 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9956 | 0.9940 | 0.9940 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 70.46% | 79.39% | 91.37% | 99.90% | 0.6779 | 0.6032 | 0.6089 |
| hard | 568 | 92.96% | 97.54% | 99.47% | 99.65% | 0.7775 | 0.8113 | 0.8114 |
| medium | 8447 | 86.43% | 91.59% | 95.89% | 98.92% | 0.8195 | 0.7740 | 0.7761 |
