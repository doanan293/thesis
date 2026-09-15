# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 1000.0 |
| Hit@3 | 85.00% |
| Hit@5 | 90.60% |
| Hit@10 | 95.60% |
| Hit@30 | 99.10% |
| MRR | 0.7633 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 50 |
| Multi-section Recall@3 | 71.00% |
| Multi-all-hit@3 | 44.00% |
| Multi-section Recall@5 | 85.00% |
| Multi-all-hit@5 | 70.00% |
| Multi-section Recall@10 | 96.00% |
| Multi-all-hit@10 | 92.00% |
| Multi-section Recall@30 | 100.00% |
| Multi-all-hit@30 | 100.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 100 | 74.00% | 85.00% | 89.00% | 99.00% | 0.6182 |
| formulary | 500 | 83.20% | 90.00% | 96.40% | 99.20% | 0.7367 |
| brand_product_qa | 250 | 96.40% | 97.60% | 99.60% | 100.00% | 0.8911 |
| multi_intent | 50 | 98.00% | 100.00% | 100.00% | 100.00% | 0.8250 |
| noisy_confuser | 50 | 40.00% | 54.00% | 72.00% | 92.00% | 0.3926 |
| patient_natural | 50 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9900 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 107 | 71.03% | 79.44% | 92.52% | 100.00% | 0.6197 |
| hard | 61 | 95.08% | 96.72% | 98.36% | 98.36% | 0.8256 |
| medium | 832 | 86.06% | 91.59% | 95.79% | 99.04% | 0.7772 |
