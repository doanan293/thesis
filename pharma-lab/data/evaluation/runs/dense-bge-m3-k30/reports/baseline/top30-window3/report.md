# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 81.78% |
| Hit@5 | 86.76% |
| Hit@10 | 91.62% |
| Hit@30 | 95.16% |
| MRR | 0.7394 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 65.40% |
| Multi-all-hit@3 | 37.60% |
| Multi-section Recall@5 | 79.90% |
| Multi-all-hit@5 | 63.20% |
| Multi-section Recall@10 | 89.60% |
| Multi-all-hit@10 | 80.40% |
| Multi-section Recall@30 | 96.10% |
| Multi-all-hit@30 | 92.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 68.50% | 78.40% | 90.00% | 97.10% | 0.5758 |
| formulary | 5000 | 81.00% | 86.48% | 91.46% | 95.08% | 0.7126 |
| brand_product_qa | 2500 | 92.60% | 94.88% | 96.92% | 98.20% | 0.8896 |
| multi_intent | 500 | 93.20% | 96.60% | 98.80% | 99.60% | 0.7934 |
| noisy_confuser | 500 | 35.80% | 46.00% | 57.40% | 70.40% | 0.3258 |
| patient_natural | 500 | 96.60% | 96.60% | 97.00% | 97.20% | 0.9428 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.75% | 83.86% | 89.64% | 95.33% | 0.6602 |
| hard | 568 | 92.78% | 95.77% | 97.89% | 98.59% | 0.8008 |
| medium | 8447 | 81.63% | 86.49% | 91.43% | 94.91% | 0.7445 |
