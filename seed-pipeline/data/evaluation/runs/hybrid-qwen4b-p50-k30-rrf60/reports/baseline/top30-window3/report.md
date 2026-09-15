# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 74.57% |
| Hit@5 | 82.97% |
| Hit@10 | 92.44% |
| Hit@30 | 98.61% |
| MRR | 0.6786 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 42.50% |
| Multi-all-hit@3 | 16.40% |
| Multi-section Recall@5 | 56.90% |
| Multi-all-hit@5 | 33.40% |
| Multi-section Recall@10 | 75.50% |
| Multi-all-hit@10 | 60.60% |
| Multi-section Recall@30 | 98.30% |
| Multi-all-hit@30 | 96.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 65.50% | 80.10% | 92.30% | 98.50% | 0.5287 |
| formulary | 5000 | 68.34% | 78.46% | 90.60% | 98.30% | 0.6137 |
| brand_product_qa | 2500 | 92.72% | 95.60% | 98.36% | 99.68% | 0.8874 |
| multi_intent | 500 | 68.60% | 80.40% | 90.40% | 100.00% | 0.5831 |
| noisy_confuser | 500 | 45.40% | 56.40% | 76.00% | 93.80% | 0.3958 |
| patient_natural | 500 | 99.40% | 99.80% | 100.00% | 100.00% | 0.9615 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 55.84% | 69.75% | 87.92% | 97.97% | 0.5197 |
| hard | 568 | 71.13% | 81.51% | 90.85% | 99.47% | 0.6188 |
| medium | 8447 | 76.99% | 84.61% | 93.07% | 98.63% | 0.7012 |
