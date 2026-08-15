# Baseline Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 78.47% |
| Hit@5 | 83.60% |
| Hit@10 | 89.01% |
| Hit@30 | 93.14% |
| MRR | 0.7150 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 62.30% |
| Multi-all-hit@3 | 32.00% |
| Multi-section Recall@5 | 75.80% |
| Multi-all-hit@5 | 55.40% |
| Multi-section Recall@10 | 85.40% |
| Multi-all-hit@10 | 72.60% |
| Multi-section Recall@30 | 92.50% |
| Multi-all-hit@30 | 85.40% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 91.60% | 94.12% | 96.04% | 97.48% | 0.8830 |
| chunk_level_retrieval | 1000 | 68.50% | 78.40% | 90.10% | 97.20% | 0.5747 |
| formulary | 5000 | 75.06% | 80.78% | 86.86% | 91.58% | 0.6695 |
| multi_intent | 500 | 92.60% | 96.20% | 98.20% | 99.60% | 0.7878 |
| noisy_confuser | 500 | 35.40% | 44.60% | 56.00% | 68.60% | 0.3147 |
| patient_natural | 500 | 95.80% | 96.00% | 97.00% | 97.00% | 0.9380 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.65% | 83.25% | 89.04% | 94.52% | 0.6597 |
| hard | 568 | 92.25% | 95.42% | 97.36% | 98.59% | 0.7956 |
| medium | 8447 | 77.76% | 82.85% | 88.45% | 92.61% | 0.7160 |
