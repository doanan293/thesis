# Retrieval Metrics

| Metric | Value |
| --- | ---: |
| Total Queries | 10000.0 |
| Hit@3 | 78.28% |
| Hit@5 | 84.85% |
| Hit@10 | 91.88% |
| Hit@30 | 97.18% |
| MRR | 0.7042 |

## Multi-required queries

| Metric | Value |
| --- | ---: |
| Total Multi-required Queries | 500 |
| Multi-section Recall@3 | 63.80% |
| Multi-all-hit@3 | 36.40% |
| Multi-section Recall@5 | 77.10% |
| Multi-all-hit@5 | 58.20% |
| Multi-section Recall@10 | 91.10% |
| Multi-all-hit@10 | 83.20% |
| Multi-section Recall@30 | 98.90% |
| Multi-all-hit@30 | 98.00% |

## Breakdown by eval_group

| eval_group | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk_level_retrieval | 1000 | 51.00% | 65.90% | 83.10% | 96.80% | 0.4117 |
| formulary | 5000 | 77.76% | 84.96% | 92.16% | 97.24% | 0.6767 |
| brand_product_qa | 2500 | 90.92% | 93.44% | 96.12% | 98.48% | 0.8710 |
| multi_intent | 500 | 91.20% | 96.00% | 99.00% | 99.80% | 0.7826 |
| noisy_confuser | 500 | 43.20% | 54.40% | 71.60% | 86.60% | 0.3937 |
| patient_natural | 500 | 97.00% | 98.00% | 98.60% | 98.80% | 0.9615 |

## Breakdown by difficulty

| difficulty | Count | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 985 | 76.45% | 85.38% | 93.20% | 98.27% | 0.6698 |
| hard | 568 | 91.20% | 95.60% | 98.24% | 99.12% | 0.7953 |
| medium | 8447 | 77.63% | 84.07% | 91.30% | 96.92% | 0.7020 |
