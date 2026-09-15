# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 86.31% |
| Hit@5 | 91.21% |
| Hit@10 | 96.18% |
| Hit@30 | 99.09% |
| MRR | 0.7823 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 66.10% |
| Multi-all-hit@3 | 37.80% |
| Multi-section Recall@5 | 81.20% |
| Multi-all-hit@5 | 63.80% |
| Multi-section Recall@10 | 93.30% |
| Multi-all-hit@10 | 87.20% |
| Multi-section Recall@30 | 99.80% |
| Multi-all-hit@30 | 99.60% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 77.20% | 86.80% | 94.30% | 99.10% | 0.6508 |
| formulary | 5000 | 85.14% | 90.62% | 96.20% | 99.02% | 0.7607 |
| brand_product_qa | 2500 | 95.48% | 97.60% | 99.36% | 99.72% | 0.8935 |
| multi_intent | 500 | 94.40% | 98.60% | 99.40% | 100.00% | 0.8392 |
| noisy_confuser | 500 | 48.60% | 57.80% | 76.80% | 94.80% | 0.4462 |
| patient_natural | 500 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9840 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 84.57% | 89.64% | 95.03% | 99.80% | 0.7705 |
| hard | 568 | 94.01% | 97.71% | 98.94% | 99.65% | 0.8458 |
| medium | 8447 | 86.00% | 90.96% | 96.13% | 98.97% | 0.7793 |
