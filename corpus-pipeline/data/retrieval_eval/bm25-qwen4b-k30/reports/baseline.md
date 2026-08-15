# Baseline Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 60.25% |
| Hit@5 | 67.29% |
| Hit@10 | 78.06% |
| Hit@30 | 90.71% |
| MRR | 0.5551 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 20.70% |
| Multi-all-hit@3 | 6.00% |
| Multi-section Recall@5 | 29.10% |
| Multi-all-hit@5 | 10.40% |
| Multi-section Recall@10 | 46.50% |
| Multi-all-hit@10 | 25.00% |
| Multi-section Recall@30 | 74.60% |
| Multi-all-hit@30 | 60.20% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 93.08% | 95.36% | 97.68% | 99.44% | 0.8886 |
| chunk_level_retrieval | 1000 | 68.90% | 80.10% | 92.10% | 97.00% | 0.5578 |
| formulary | 5000 | 44.26% | 52.84% | 66.64% | 84.96% | 0.4016 |
| multi_intent | 500 | 35.40% | 47.80% | 68.00% | 89.00% | 0.3164 |
| noisy_confuser | 500 | 24.00% | 32.60% | 54.20% | 84.40% | 0.2468 |
| patient_natural | 500 | 99.80% | 100.00% | 100.00% | 100.00% | 0.9647 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 32.99% | 41.02% | 56.85% | 83.76% | 0.3181 |
| hard | 568 | 41.90% | 52.99% | 70.95% | 89.44% | 0.3822 |
| medium | 8447 | 64.66% | 71.32% | 81.01% | 91.61% | 0.5944 |
