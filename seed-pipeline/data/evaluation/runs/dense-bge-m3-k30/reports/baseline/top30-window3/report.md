# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 81.08% |
| Hit@5 | 85.96% |
| Hit@10 | 90.95% |
| Hit@30 | 94.55% |
| MRR | 0.7334 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 63.50% |
| Multi-all-hit@3 | 34.40% |
| Multi-section Recall@5 | 76.70% |
| Multi-all-hit@5 | 57.40% |
| Multi-section Recall@10 | 86.30% |
| Multi-all-hit@10 | 74.40% |
| Multi-section Recall@30 | 93.40% |
| Multi-all-hit@30 | 87.20% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 68.50% | 78.20% | 89.80% | 97.00% | 0.5742 |
| formulary | 5000 | 80.08% | 85.46% | 90.88% | 94.56% | 0.7051 |
| brand_product_qa | 2500 | 92.16% | 94.48% | 96.36% | 97.64% | 0.8870 |
| multi_intent | 500 | 92.60% | 96.00% | 98.20% | 99.60% | 0.7880 |
| noisy_confuser | 500 | 35.00% | 44.40% | 54.80% | 67.40% | 0.3147 |
| patient_natural | 500 | 95.40% | 95.40% | 95.80% | 96.20% | 0.9318 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.24% | 82.94% | 88.73% | 93.91% | 0.6581 |
| hard | 568 | 92.25% | 95.25% | 97.36% | 98.59% | 0.7961 |
| medium | 8447 | 80.89% | 85.69% | 90.78% | 94.35% | 0.7380 |
