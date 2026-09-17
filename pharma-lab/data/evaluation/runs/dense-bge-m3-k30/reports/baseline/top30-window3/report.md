# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 81.78% |
| Hit@5 | 86.76% |
| Hit@10 | 91.62% |
| Hit@30 | 95.16% |
| nDCG@10 | 0.7758 |
| MRR@10 | 0.7371 |
| MRR@30 | 0.7394 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 65.40% |
| Complete-evidence rate@3 | 37.60% |
| Recall@5 | 79.90% |
| Complete-evidence rate@5 | 63.20% |
| Recall@10 | 89.60% |
| Complete-evidence rate@10 | 80.40% |
| Recall@30 | 96.10% |
| Complete-evidence rate@30 | 92.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 68.50% | 78.40% | 90.00% | 97.10% | 0.6504 | 0.5712 | 0.5758 |
| formulary | 5000 | 81.00% | 86.48% | 91.46% | 95.08% | 0.7605 | 0.7103 | 0.7126 |
| brand_product_qa | 2500 | 92.60% | 94.88% | 96.92% | 98.20% | 0.9085 | 0.8888 | 0.8896 |
| multi_intent | 500 | 93.20% | 96.60% | 98.80% | 99.60% | 0.7390 | 0.7929 | 0.7934 |
| noisy_confuser | 500 | 35.80% | 46.00% | 57.40% | 70.40% | 0.3784 | 0.3176 | 0.3258 |
| patient_natural | 500 | 96.60% | 96.60% | 97.00% | 97.20% | 0.9497 | 0.9427 | 0.9428 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.75% | 83.86% | 89.64% | 95.33% | 0.7153 | 0.6565 | 0.6602 |
| hard | 568 | 92.78% | 95.77% | 97.89% | 98.59% | 0.7546 | 0.8004 | 0.8008 |
| medium | 8447 | 81.63% | 86.49% | 91.43% | 94.91% | 0.7842 | 0.7423 | 0.7445 |
