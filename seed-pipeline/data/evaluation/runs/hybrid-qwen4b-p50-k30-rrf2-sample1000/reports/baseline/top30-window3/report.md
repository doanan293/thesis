# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 1000.0 |
| Hit@3 | 84.00% |
| Hit@5 | 90.90% |
| Hit@10 | 95.60% |
| Hit@30 | 99.10% |
| MRR | 0.7398 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 50 |
| Multi-section Recall@3 | 70.00% |
| Multi-all-hit@3 | 44.00% |
| Multi-section Recall@5 | 86.00% |
| Multi-all-hit@5 | 72.00% |
| Multi-section Recall@10 | 95.00% |
| Multi-all-hit@10 | 90.00% |
| Multi-section Recall@30 | 100.00% |
| Multi-all-hit@30 | 100.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 100 | 61.00% | 75.00% | 88.00% | 99.00% | 0.4711 |
| formulary | 500 | 83.40% | 92.60% | 97.00% | 99.20% | 0.7131 |
| brand_product_qa | 250 | 95.60% | 97.60% | 99.60% | 100.00% | 0.9108 |
| multi_intent | 50 | 96.00% | 100.00% | 100.00% | 100.00% | 0.7657 |
| noisy_confuser | 50 | 50.00% | 54.00% | 68.00% | 92.00% | 0.4538 |
| patient_natural | 50 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9500 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 107 | 89.72% | 92.52% | 96.26% | 100.00% | 0.7171 |
| hard | 61 | 93.44% | 96.72% | 96.72% | 98.36% | 0.7758 |
| medium | 832 | 82.57% | 90.26% | 95.43% | 99.04% | 0.7401 |
