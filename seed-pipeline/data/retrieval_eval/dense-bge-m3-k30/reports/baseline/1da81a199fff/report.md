# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 81.03% |
| Hit@5 | 86.01% |
| Hit@10 | 90.95% |
| Hit@30 | 94.54% |
| MRR | 0.7333 |

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
| formulary | 5000 | 80.14% | 85.58% | 90.72% | 94.36% | 0.7056 |
| multi_intent | 500 | 92.60% | 96.20% | 98.20% | 99.60% | 0.7878 |
| noisy_confuser | 500 | 35.80% | 44.80% | 56.20% | 68.80% | 0.3196 |
| patient_natural | 500 | 95.80% | 96.00% | 97.00% | 97.00% | 0.9380 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.65% | 83.25% | 89.04% | 94.52% | 0.6597 |
| hard | 568 | 92.25% | 95.42% | 97.36% | 98.59% | 0.7956 |
| medium | 8447 | 80.79% | 85.70% | 90.74% | 94.27% | 0.7376 |
