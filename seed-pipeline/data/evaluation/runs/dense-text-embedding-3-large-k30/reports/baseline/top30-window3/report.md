# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 82.63% |
| Hit@5 | 89.02% |
| Hit@10 | 94.85% |
| Hit@30 | 98.17% |
| MRR | 0.7506 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 82.00% |
| Multi-all-hit@3 | 67.40% |
| Multi-section Recall@5 | 92.80% |
| Multi-all-hit@5 | 86.00% |
| Multi-section Recall@10 | 98.50% |
| Multi-all-hit@10 | 97.40% |
| Multi-section Recall@30 | 99.40% |
| Multi-all-hit@30 | 99.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 56.40% | 72.30% | 88.40% | 98.80% | 0.4678 |
| formulary | 5000 | 84.80% | 90.88% | 96.10% | 98.28% | 0.7611 |
| brand_product_qa | 2500 | 90.20% | 93.84% | 97.12% | 98.76% | 0.8477 |
| multi_intent | 500 | 96.60% | 99.60% | 99.60% | 99.80% | 0.8787 |
| noisy_confuser | 500 | 49.00% | 61.60% | 77.00% | 92.20% | 0.4264 |
| patient_natural | 500 | 95.20% | 96.60% | 97.00% | 97.20% | 0.9217 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 88.22% | 93.50% | 98.38% | 99.70% | 0.8016 |
| hard | 568 | 95.25% | 98.42% | 98.77% | 99.47% | 0.8727 |
| medium | 8447 | 81.13% | 87.87% | 94.18% | 97.90% | 0.7364 |
