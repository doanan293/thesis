# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 78.24% |
| Hit@5 | 84.90% |
| Hit@10 | 91.85% |
| Hit@30 | 97.15% |
| MRR | 0.7042 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 63.90% |
| Multi-all-hit@3 | 36.60% |
| Multi-section Recall@5 | 77.40% |
| Multi-all-hit@5 | 58.80% |
| Multi-section Recall@10 | 91.30% |
| Multi-all-hit@10 | 83.60% |
| Multi-section Recall@30 | 99.00% |
| Multi-all-hit@30 | 98.20% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brand_product_qa | 2500 | 91.00% | 93.64% | 96.36% | 98.72% | 0.8724 |
| chunk_level_retrieval | 1000 | 51.10% | 66.00% | 83.20% | 97.00% | 0.4121 |
| formulary | 5000 | 77.64% | 84.98% | 92.02% | 97.12% | 0.6763 |
| multi_intent | 500 | 91.20% | 96.00% | 99.00% | 99.80% | 0.7829 |
| noisy_confuser | 500 | 43.00% | 54.20% | 71.40% | 86.20% | 0.3916 |
| patient_natural | 500 | 97.00% | 97.80% | 98.20% | 98.20% | 0.9597 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.45% | 85.48% | 93.10% | 98.17% | 0.6699 |
| hard | 568 | 91.20% | 95.60% | 98.24% | 99.12% | 0.7956 |
| medium | 8447 | 77.58% | 84.11% | 91.28% | 96.90% | 0.7020 |
