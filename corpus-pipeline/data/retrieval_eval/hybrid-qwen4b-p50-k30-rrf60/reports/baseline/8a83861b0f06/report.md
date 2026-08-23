# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 74.57% |
| Hit@5 | 82.99% |
| Hit@10 | 92.45% |
| Hit@30 | 98.61% |
| MRR | 0.6784 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 42.60% |
| Multi-all-hit@3 | 16.20% |
| Multi-section Recall@5 | 57.00% |
| Multi-all-hit@5 | 33.40% |
| Multi-section Recall@10 | 74.60% |
| Multi-all-hit@10 | 59.00% |
| Multi-section Recall@30 | 98.40% |
| Multi-all-hit@30 | 96.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 92.92% | 95.68% | 98.36% | 99.68% | 0.8860 |
| chunk_level_retrieval | 1000 | 65.60% | 80.50% | 92.40% | 98.60% | 0.5309 |
| formulary | 5000 | 68.20% | 78.36% | 90.58% | 98.28% | 0.6136 |
| multi_intent | 500 | 69.00% | 80.60% | 90.20% | 100.00% | 0.5823 |
| noisy_confuser | 500 | 45.00% | 56.40% | 76.40% | 93.80% | 0.3940 |
| patient_natural | 500 | 99.60% | 99.80% | 100.00% | 100.00% | 0.9640 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 56.04% | 69.54% | 87.61% | 97.97% | 0.5197 |
| hard | 568 | 71.48% | 81.69% | 90.67% | 99.47% | 0.6172 |
| medium | 8447 | 76.94% | 84.65% | 93.13% | 98.63% | 0.7010 |
