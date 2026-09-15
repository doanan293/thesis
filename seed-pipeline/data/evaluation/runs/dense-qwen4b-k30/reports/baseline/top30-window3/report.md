# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 85.18% |
| Hit@5 | 90.16% |
| Hit@10 | 95.22% |
| Hit@30 | 98.45% |
| MRR | 0.7786 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 81.10% |
| Multi-all-hit@3 | 65.80% |
| Multi-section Recall@5 | 91.60% |
| Multi-all-hit@5 | 84.60% |
| Multi-section Recall@10 | 98.80% |
| Multi-all-hit@10 | 97.60% |
| Multi-section Recall@30 | 99.90% |
| Multi-all-hit@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 58.30% | 71.80% | 87.10% | 98.20% | 0.4762 |
| formulary | 5000 | 88.20% | 92.62% | 96.68% | 98.92% | 0.7923 |
| brand_product_qa | 2500 | 90.40% | 93.32% | 96.60% | 98.48% | 0.8673 |
| multi_intent | 500 | 96.40% | 98.60% | 100.00% | 100.00% | 0.8924 |
| noisy_confuser | 500 | 59.20% | 70.80% | 82.80% | 93.40% | 0.5267 |
| patient_natural | 500 | 97.40% | 97.40% | 97.60% | 97.60% | 0.9409 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 94.42% | 96.24% | 98.68% | 100.00% | 0.8929 |
| hard | 568 | 95.60% | 97.54% | 99.12% | 99.82% | 0.8904 |
| medium | 8447 | 83.40% | 88.95% | 94.55% | 98.18% | 0.7577 |
