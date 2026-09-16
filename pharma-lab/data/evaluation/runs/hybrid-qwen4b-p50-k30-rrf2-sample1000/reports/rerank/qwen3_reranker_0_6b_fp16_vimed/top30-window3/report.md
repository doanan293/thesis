# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 1000.0 |
| Hit@3 | 87.60% |
| Hit@5 | 91.80% |
| Hit@10 | 96.20% |
| Hit@30 | 99.10% |
| MRR | 0.8078 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 50 |
| Multi-section Recall@3 | 77.00% |
| Multi-all-hit@3 | 56.00% |
| Multi-section Recall@5 | 85.00% |
| Multi-all-hit@5 | 70.00% |
| Multi-section Recall@10 | 99.00% |
| Multi-all-hit@10 | 98.00% |
| Multi-section Recall@30 | 100.00% |
| Multi-all-hit@30 | 100.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 100 | 77.00% | 84.00% | 91.00% | 99.00% | 0.6511 |
| formulary | 500 | 87.40% | 92.60% | 97.20% | 99.20% | 0.8005 |
| brand_product_qa | 250 | 96.80% | 98.40% | 99.20% | 100.00% | 0.9129 |
| multi_intent | 50 | 98.00% | 100.00% | 100.00% | 100.00% | 0.8683 |
| noisy_confuser | 50 | 42.00% | 50.00% | 74.00% | 92.00% | 0.4264 |
| patient_natural | 50 | 100.00% | 100.00% | 100.00% | 100.00% | 0.9900 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 107 | 85.98% | 92.52% | 96.26% | 100.00% | 0.8263 |
| hard | 61 | 95.08% | 96.72% | 98.36% | 98.36% | 0.8613 |
| medium | 832 | 87.26% | 91.35% | 96.03% | 99.04% | 0.8015 |
