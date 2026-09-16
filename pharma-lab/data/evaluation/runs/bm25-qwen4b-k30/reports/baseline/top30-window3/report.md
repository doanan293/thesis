# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 62.81% |
| Hit@5 | 70.44% |
| Hit@10 | 81.33% |
| Hit@30 | 92.64% |
| MRR | 0.5755 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 32.00% |
| Multi-all-hit@3 | 16.00% |
| Multi-section Recall@5 | 44.20% |
| Multi-all-hit@5 | 24.60% |
| Multi-section Recall@10 | 64.50% |
| Multi-all-hit@10 | 44.40% |
| Multi-section Recall@30 | 88.20% |
| Multi-all-hit@30 | 79.80% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 68.90% | 80.00% | 92.00% | 96.90% | 0.5586 |
| formulary | 5000 | 48.12% | 57.52% | 71.52% | 88.06% | 0.4319 |
| brand_product_qa | 2500 | 93.04% | 95.36% | 97.68% | 99.44% | 0.8888 |
| multi_intent | 500 | 48.00% | 63.80% | 84.60% | 96.60% | 0.4168 |
| noisy_confuser | 500 | 24.20% | 33.00% | 54.40% | 84.60% | 0.2489 |
| patient_natural | 500 | 99.80% | 100.00% | 100.00% | 100.00% | 0.9647 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 36.95% | 46.19% | 64.16% | 89.14% | 0.3464 |
| hard | 568 | 52.99% | 67.08% | 85.56% | 96.13% | 0.4705 |
| medium | 8447 | 66.49% | 73.49% | 83.05% | 92.81% | 0.6093 |
