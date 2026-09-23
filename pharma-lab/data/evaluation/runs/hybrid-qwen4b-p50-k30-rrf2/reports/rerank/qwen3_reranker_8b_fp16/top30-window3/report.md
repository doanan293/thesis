# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 91.80% |
| Hit@5 | 95.19% |
| Hit@10 | 97.72% |
| Hit@30 | 99.17% |
| nDCG@10 | 0.8799 |
| MRR@10 | 0.8546 |
| MRR@30 | 0.8557 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Recall@3 | 71.50% |
| Complete-evidence rate@3 | 49.60% |
| Recall@5 | 85.80% |
| Complete-evidence rate@5 | 73.80% |
| Recall@10 | 95.80% |
| Complete-evidence rate@10 | 91.80% |
| Recall@30 | 99.90% |
| Complete-evidence rate@30 | 99.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 91.30% | 94.70% | 97.20% | 99.00% | 0.8419 | 0.7983 | 0.7996 |
| formulary | 5000 | 90.58% | 94.88% | 97.84% | 99.20% | 0.8685 | 0.8324 | 0.8334 |
| brand_product_qa | 2500 | 98.08% | 99.00% | 99.56% | 99.72% | 0.9706 | 0.9624 | 0.9625 |
| multi_intent | 500 | 93.40% | 97.80% | 99.80% | 100.00% | 0.7684 | 0.8255 | 0.8257 |
| noisy_confuser | 500 | 63.80% | 72.80% | 84.00% | 94.80% | 0.6158 | 0.5445 | 0.5518 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9926 | 0.9900 | 0.9900 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | nDCG@10 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 95.03% | 98.38% | 99.59% | 99.90% | 0.9032 | 0.8720 | 0.8723 |
| hard | 568 | 93.13% | 97.18% | 99.47% | 99.65% | 0.7864 | 0.8348 | 0.8349 |
| medium | 8447 | 91.33% | 94.68% | 97.38% | 99.05% | 0.8835 | 0.8540 | 0.8551 |
