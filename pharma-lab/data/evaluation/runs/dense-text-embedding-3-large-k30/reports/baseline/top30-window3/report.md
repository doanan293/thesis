# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 83.16% |
| Hit@5 | 89.35% |
| Hit@10 | 95.00% |
| Hit@30 | 98.12% |
| nDCG@10 | 0.7999 |
| MRR@10 | 0.7540 |
| MRR@30 | 0.7560 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 84.90% |
| Complete-evidence rate@3 | 72.60% |
| Recall@5 | 94.40% |
| Complete-evidence rate@5 | 89.20% |
| Recall@10 | 98.70% |
| Complete-evidence rate@10 | 97.80% |
| Recall@30 | 99.60% |
| Complete-evidence rate@30 | 99.20% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 56.40% | 72.30% | 88.40% | 98.80% | 0.5624 | 0.4612 | 0.4678 |
| formulary | 5000 | 85.80% | 91.54% | 96.40% | 98.16% | 0.8166 | 0.7688 | 0.7699 |
| brand_product_qa | 2500 | 90.20% | 93.84% | 97.12% | 98.76% | 0.8771 | 0.8466 | 0.8477 |
| multi_intent | 500 | 97.20% | 99.60% | 99.60% | 100.00% | 0.8868 | 0.8982 | 0.8984 |
| noisy_confuser | 500 | 49.00% | 61.60% | 77.00% | 92.20% | 0.5004 | 0.4169 | 0.4264 |
| patient_natural | 500 | 95.20% | 96.60% | 97.00% | 97.20% | 0.9338 | 0.9216 | 0.9217 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 88.83% | 94.01% | 98.68% | 99.70% | 0.8501 | 0.8059 | 0.8065 |
| hard | 568 | 95.77% | 98.42% | 98.77% | 99.65% | 0.8823 | 0.8894 | 0.8900 |
| medium | 8447 | 81.65% | 88.20% | 94.32% | 97.83% | 0.7885 | 0.7388 | 0.7411 |
