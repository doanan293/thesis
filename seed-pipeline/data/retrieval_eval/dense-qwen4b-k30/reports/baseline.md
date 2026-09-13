# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 85.15% |
| Hit@5 | 90.15% |
| Hit@10 | 95.21% |
| Hit@30 | 98.53% |
| MRR | 0.7784 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 81.00% |
| Multi-all-hit@3 | 65.60% |
| Multi-section Recall@5 | 91.60% |
| Multi-all-hit@5 | 84.80% |
| Multi-section Recall@10 | 98.90% |
| Multi-all-hit@10 | 97.80% |
| Multi-section Recall@30 | 100.00% |
| Multi-all-hit@30 | 100.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 90.36% | 93.20% | 96.44% | 98.36% | 0.8668 |
| chunk_level_retrieval | 1000 | 58.30% | 71.90% | 87.10% | 98.40% | 0.4765 |
| formulary | 5000 | 88.20% | 92.70% | 96.86% | 99.06% | 0.7924 |
| multi_intent | 500 | 96.40% | 98.40% | 100.00% | 100.00% | 0.8910 |
| noisy_confuser | 500 | 59.20% | 70.80% | 82.20% | 94.40% | 0.5275 |
| patient_natural | 500 | 97.00% | 97.00% | 97.00% | 97.00% | 0.9380 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 94.42% | 96.24% | 98.68% | 100.00% | 0.8928 |
| hard | 568 | 95.60% | 97.36% | 99.12% | 99.82% | 0.8891 |
| medium | 8447 | 83.37% | 88.95% | 94.54% | 98.27% | 0.7576 |
