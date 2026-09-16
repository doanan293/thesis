# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 78.72% |
| Hit@5 | 85.33% |
| Hit@10 | 92.17% |
| Hit@30 | 97.34% |
| MRR | 0.7075 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 65.00% |
| Multi-all-hit@3 | 37.20% |
| Multi-section Recall@5 | 78.50% |
| Multi-all-hit@5 | 60.00% |
| Multi-section Recall@10 | 92.50% |
| Multi-all-hit@10 | 85.80% |
| Multi-section Recall@30 | 98.90% |
| Multi-all-hit@30 | 98.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 51.00% | 65.90% | 83.10% | 96.80% | 0.4117 |
| formulary | 5000 | 78.52% | 85.84% | 92.70% | 97.54% | 0.6827 |
| brand_product_qa | 2500 | 90.96% | 93.48% | 96.24% | 98.56% | 0.8713 |
| multi_intent | 500 | 92.80% | 97.00% | 99.20% | 99.80% | 0.7924 |
| noisy_confuser | 500 | 43.00% | 54.40% | 71.60% | 86.80% | 0.3925 |
| patient_natural | 500 | 96.60% | 97.60% | 98.20% | 98.40% | 0.9585 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.85% | 86.09% | 94.01% | 98.68% | 0.6746 |
| hard | 568 | 92.61% | 96.48% | 98.42% | 99.12% | 0.8040 |
| medium | 8447 | 78.00% | 84.49% | 91.54% | 97.06% | 0.7049 |
