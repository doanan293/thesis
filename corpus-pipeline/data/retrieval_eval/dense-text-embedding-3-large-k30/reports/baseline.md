# Baseline Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 83.18% |
| Hit@5 | 88.66% |
| Hit@10 | 94.03% |
| Hit@30 | 97.93% |
| MRR | 0.7579 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 83.90% |
| Multi-all-hit@3 | 70.00% |
| Multi-section Recall@5 | 93.50% |
| Multi-all-hit@5 | 87.40% |
| Multi-section Recall@10 | 98.50% |
| Multi-all-hit@10 | 97.40% |
| Multi-section Recall@30 | 99.40% |
| Multi-all-hit@30 | 99.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 91.56% | 94.40% | 97.12% | 98.76% | 0.8648 |
| chunk_level_retrieval | 1000 | 58.50% | 73.60% | 88.50% | 98.80% | 0.4770 |
| formulary | 5000 | 84.34% | 89.54% | 94.46% | 97.82% | 0.7600 |
| multi_intent | 500 | 97.80% | 99.60% | 99.60% | 99.80% | 0.9014 |
| noisy_confuser | 500 | 50.80% | 62.00% | 76.80% | 92.00% | 0.4371 |
| patient_natural | 500 | 96.80% | 97.00% | 97.00% | 97.20% | 0.9416 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 91.37% | 95.33% | 98.48% | 99.70% | 0.8275 |
| hard | 568 | 96.65% | 98.42% | 98.77% | 99.47% | 0.8965 |
| medium | 8447 | 81.32% | 87.23% | 93.19% | 97.62% | 0.7405 |
