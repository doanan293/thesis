# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 78.72% |
| Hit@5 | 85.33% |
| Hit@10 | 92.17% |
| Hit@30 | 97.34% |
| nDCG@10 | 0.7526 |
| MRR@10 | 0.7042 |
| MRR@30 | 0.7075 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 65.00% |
| Complete-evidence rate@3 | 37.20% |
| Recall@5 | 78.50% |
| Complete-evidence rate@5 | 60.00% |
| Recall@10 | 92.50% |
| Complete-evidence rate@10 | 85.80% |
| Recall@30 | 98.90% |
| Complete-evidence rate@30 | 98.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 51.00% | 65.90% | 83.10% | 96.80% | 0.5049 | 0.4029 | 0.4117 |
| formulary | 5000 | 78.52% | 85.84% | 92.70% | 97.54% | 0.7399 | 0.6795 | 0.6827 |
| brand_product_qa | 2500 | 90.96% | 93.48% | 96.24% | 98.56% | 0.8923 | 0.8698 | 0.8713 |
| multi_intent | 500 | 92.80% | 97.00% | 99.20% | 99.80% | 0.7559 | 0.7920 | 0.7924 |
| noisy_confuser | 500 | 43.00% | 54.40% | 71.60% | 86.80% | 0.4611 | 0.3827 | 0.3925 |
| patient_natural | 500 | 96.60% | 97.60% | 98.20% | 98.40% | 0.9642 | 0.9584 | 0.9585 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.85% | 86.09% | 94.01% | 98.68% | 0.7366 | 0.6715 | 0.6746 |
| hard | 568 | 92.61% | 96.48% | 98.42% | 99.12% | 0.7729 | 0.8036 | 0.8040 |
| medium | 8447 | 78.00% | 84.49% | 91.54% | 97.06% | 0.7531 | 0.7013 | 0.7049 |
