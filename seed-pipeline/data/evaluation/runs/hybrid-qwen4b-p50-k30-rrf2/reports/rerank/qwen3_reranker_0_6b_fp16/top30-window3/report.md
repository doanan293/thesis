# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 87.36% |
| Hit@5 | 92.04% |
| Hit@10 | 96.75% |
| Hit@30 | 99.17% |
| MRR | 0.7905 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 74.90% |
| Multi-all-hit@3 | 51.60% |
| Multi-section Recall@5 | 89.80% |
| Multi-all-hit@5 | 79.80% |
| Multi-section Recall@10 | 98.20% |
| Multi-all-hit@10 | 96.40% |
| Multi-section Recall@30 | 99.90% |
| Multi-all-hit@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 77.70% | 87.00% | 94.60% | 99.00% | 0.6478 |
| formulary | 5000 | 86.76% | 92.06% | 97.22% | 99.20% | 0.7743 |
| brand_product_qa | 2500 | 95.48% | 97.68% | 99.40% | 99.72% | 0.8938 |
| multi_intent | 500 | 98.20% | 99.80% | 100.00% | 100.00% | 0.8726 |
| noisy_confuser | 500 | 48.60% | 58.00% | 76.60% | 94.80% | 0.4484 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9813 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 86.19% | 91.47% | 96.65% | 99.90% | 0.7835 |
| hard | 568 | 97.36% | 98.77% | 99.47% | 99.65% | 0.8752 |
| medium | 8447 | 86.82% | 91.65% | 96.58% | 99.05% | 0.7856 |
