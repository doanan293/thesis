# Baseline Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 84.19% |
| Hit@5 | 89.34% |
| Hit@10 | 94.68% |
| Hit@30 | 98.36% |
| MRR | 0.7720 |

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
| chunk_level_retrieval | 1000 | 58.30% | 71.90% | 87.10% | 98.40% | 0.4760 |
| formulary | 5000 | 86.28% | 91.08% | 95.82% | 98.74% | 0.7800 |
| multi_intent | 500 | 96.40% | 98.40% | 100.00% | 100.00% | 0.8910 |
| noisy_confuser | 500 | 59.20% | 70.80% | 82.00% | 94.20% | 0.5258 |
| patient_natural | 500 | 97.00% | 97.00% | 97.00% | 97.00% | 0.9380 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 94.42% | 96.24% | 98.68% | 100.00% | 0.8928 |
| hard | 568 | 95.60% | 97.36% | 99.12% | 99.82% | 0.8891 |
| medium | 8447 | 82.23% | 88.00% | 93.91% | 98.07% | 0.7501 |
