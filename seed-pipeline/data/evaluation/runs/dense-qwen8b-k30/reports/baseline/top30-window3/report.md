# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 86.38% |
| Hit@5 | 90.98% |
| Hit@10 | 95.26% |
| Hit@30 | 98.29% |
| MRR | 0.7952 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 85.30% |
| Multi-all-hit@3 | 72.20% |
| Multi-section Recall@5 | 93.80% |
| Multi-all-hit@5 | 88.20% |
| Multi-section Recall@10 | 98.70% |
| Multi-all-hit@10 | 97.60% |
| Multi-section Recall@30 | 99.60% |
| Multi-all-hit@30 | 99.20% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 61.20% | 75.80% | 88.60% | 98.70% | 0.4993 |
| formulary | 5000 | 89.34% | 93.18% | 96.38% | 98.40% | 0.8146 |
| brand_product_qa | 2500 | 91.60% | 94.56% | 97.44% | 99.00% | 0.8734 |
| multi_intent | 500 | 98.40% | 99.40% | 99.80% | 100.00% | 0.9159 |
| noisy_confuser | 500 | 58.60% | 67.00% | 80.20% | 92.40% | 0.5257 |
| patient_natural | 500 | 96.80% | 97.00% | 97.00% | 97.00% | 0.9504 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 93.50% | 96.24% | 98.98% | 100.00% | 0.8867 |
| hard | 568 | 97.36% | 98.24% | 98.77% | 99.65% | 0.9106 |
| medium | 8447 | 84.81% | 89.88% | 94.59% | 98.00% | 0.7768 |
