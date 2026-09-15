# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 56.51% |
| Hit@5 | 71.47% |
| Hit@10 | 89.75% |
| Hit@30 | 99.09% |
| MRR | 0.4797 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 38.10% |
| Multi-all-hit@3 | 11.20% |
| Multi-section Recall@5 | 56.50% |
| Multi-all-hit@5 | 32.20% |
| Multi-section Recall@10 | 83.20% |
| Multi-all-hit@10 | 72.20% |
| Multi-section Recall@30 | 99.80% |
| Multi-all-hit@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 59.60% | 76.30% | 93.80% | 99.10% | 0.4762 |
| formulary | 5000 | 42.36% | 60.64% | 85.62% | 99.02% | 0.3493 |
| brand_product_qa | 2500 | 79.92% | 89.12% | 97.36% | 99.72% | 0.7159 |
| multi_intent | 500 | 65.00% | 80.80% | 94.20% | 100.00% | 0.4912 |
| noisy_confuser | 500 | 31.60% | 47.00% | 71.00% | 94.80% | 0.2837 |
| patient_natural | 500 | 91.20% | 97.00% | 99.20% | 100.00% | 0.7949 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 28.53% | 46.50% | 80.41% | 99.80% | 0.2868 |
| hard | 568 | 66.90% | 82.04% | 94.37% | 99.65% | 0.5163 |
| medium | 8447 | 59.07% | 73.67% | 90.53% | 98.97% | 0.4998 |
